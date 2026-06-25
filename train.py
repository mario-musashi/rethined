import os
import csv
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.models import vgg16, VGG16_Weights
import numpy as np
from PIL import Image, ImageDraw
from einops import rearrange
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model import InpaintingModel

# ── Config ────────────────────────────────────────────────────────────────────

TRAIN_DIR = "data/DF2K/HR"          # HR images for training
VAL_DIR = "data/DF2K/val_hr"            # or separate val set
MASK_DIR = "test_masks/DF8K-Inpainting/masks/test"      # validation masks only
IMAGE_SIZE = 512
PATCH_SIZE = 8
BATCH_SIZE = 4                      # adjust to your GPU (8GB → 4, 16GB → 8, 24GB → 16)
LR = 1e-3
TOTAL_STEPS = 600_000
WARMUP_STEPS = 5_000
NUM_WORKERS = 4
CHECKPOINT_DIR = "checkpoints"
DEBUG_DIR = "debug_images"
LOG_INTERVAL = 100
CHECKPOINT_INTERVAL = 5000
VAL_INTERVAL = 5000
PATIENCE = 10                       # validations without improvement before stopping
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Mask Generation (LaMa-style irregular masks) ──────────────────────────────

def random_brush_mask(shape, max_brush_width=24, max_length=120, max_vertices=8):
    """Generate irregular brush-stroke mask using PIL."""
    h, w = shape
    mask_img = Image.new('L', (w, h), 0)
    draw = ImageDraw.Draw(mask_img)
    num_strokes = random.randint(5, 20)
    for _ in range(num_strokes):
        x, y = random.randint(0, w - 1), random.randint(0, h - 1)
        brush_width = random.randint(3, max_brush_width)
        vertices = random.randint(2, max_vertices)
        points = [(x, y)]
        for _ in range(vertices):
            angle = random.uniform(0, 2 * math.pi)
            length = random.randint(10, max_length)
            x = int(x + length * math.cos(angle))
            y = int(y + length * math.sin(angle))
            x = max(0, min(w - 1, x))
            y = max(0, min(h - 1, y))
            points.append((x, y))
        draw.line(points, fill=255, width=brush_width)
    return np.array(mask_img, dtype=np.float32) / 255.0


def generate_mask(h, w):
    """Generate a random irregular mask at (h, w) resolution."""
    mask = np.zeros((h, w), dtype=np.float32)
    num_shapes = random.randint(3, 12)

    for _ in range(num_shapes):
        shape_type = random.choice(['rectangle', 'brush'])
        if shape_type == 'rectangle':
            x0 = random.randint(0, w-1)
            y0 = random.randint(0, h-1)
            x1 = min(w, x0 + random.randint(16, w//2))
            y1 = min(h, y0 + random.randint(16, h//2))
            mask[y0:y1, x0:x1] = 1.0
        else:
            brush = random_brush_mask((h//2, w//2), max_brush_width=12, max_length=60, max_vertices=6)
            brush_img = Image.fromarray((brush * 255).astype(np.uint8))
            brush_img = brush_img.resize((w, h), Image.NEAREST)
            mask = np.maximum(mask, np.array(brush_img, dtype=np.float32) / 255.0)

    # ensure mask covers 20-50% of image
    coverage = mask.mean()
    target = random.uniform(0.2, 0.5)
    if coverage > 0 and coverage < target:
        scale = target / coverage
        mask = np.clip(mask * scale, 0, 1)
    elif coverage == 0:
        mask[random.randint(0, h-1), random.randint(0, w-1)] = 1

    return torch.from_numpy(mask).float().unsqueeze(0)


# ── Dataset ──────────────────────────────────────────────────────────────────

class InpaintingDataset(Dataset):
    """Load HR images, generate random masks at each epoch."""

    def __init__(self, root_dir, image_size=512):
        self.root_dir = root_dir
        self.image_size = image_size
        self.files = sorted([
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert('RGB')
        w, h = img.size
        load_size = self.image_size
        if w > h:
            new_w = 2 * load_size
            new_h = int(h * (2 * load_size) / w)
        else:
            new_h = 2 * load_size
            new_w = int(w * (2 * load_size) / h)
        img = img.resize((max(new_w, 1), max(new_h, 1)), Image.BICUBIC)
        img = transforms.CenterCrop((2 * load_size, 2 * load_size))(img)
        img_tensor = self.to_tensor(img)

        mask = generate_mask(2 * load_size, 2 * load_size)
        return img_tensor, mask


def collate_fn(batch):
    images, masks = zip(*batch)
    images = torch.stack(images, 0)
    masks = torch.stack(masks, 0)
    return images, masks


# ── Validation dataset with fixed masks ──────────────────────────────────────

class ValDataset(Dataset):
    """Load HR images + precomputed masks for validation."""

    def __init__(self, img_root, mask_root, image_size=512, split='div2k'):
        self.image_size = image_size
        self.to_tensor = transforms.ToTensor()
        mask_dir = os.path.join(mask_root, split)
        mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.png')])
        self.pairs = []
        for mf in mask_files:
            base = os.path.splitext(mf)[0]
            # try common naming patterns
            for ext in ['.png', '.jpg', '.jpeg']:
                img_path = os.path.join(img_root, base + ext)
                if os.path.exists(img_path):
                    mask_path = os.path.join(mask_dir, mf)
                    self.pairs.append((img_path, mask_path))
                    break

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        # crop/pad to square at 2x training size then downsample
        w, h = img.size
        s = 2 * self.image_size
        if w > h:
            new_w = s
            new_h = int(h * s / w)
        else:
            new_h = s
            new_w = int(w * s / h)
        img = img.resize((max(new_w, 1), max(new_h, 1)), Image.BICUBIC)
        mask = mask.resize((max(new_w, 1), max(new_h, 1)), Image.NEAREST)
        img = transforms.CenterCrop((s, s))(img)
        mask = transforms.CenterCrop((s, s))(mask)

        img_tensor = self.to_tensor(img)
        mask_tensor = torch.from_numpy(np.array(mask, dtype=np.float32) / 255.0).unsqueeze(0)
        mask_tensor = (mask_tensor > 0.5).float()
        return img_tensor, mask_tensor


# ── Debug Image Saving ───────────────────────────────────────────────────────

@torch.no_grad()
def save_debug_images(images_hr, masks_hr, output, coarse, step, epoch):
    os.makedirs(DEBUG_DIR, exist_ok=True)
    b = min(images_hr.size(0), 4)
    images_lr = F.interpolate(images_hr[:b], size=IMAGE_SIZE, mode='bicubic', antialias=True)
    masks_lr = F.interpolate(masks_hr[:b], size=IMAGE_SIZE)
    masked = images_lr * (1 - masks_lr)

    grid = torch.cat([
        masked.cpu(),
        images_lr.cpu(),
        coarse[:b].cpu(),
        output[:b].cpu(),
    ], dim=0)
    save_image(grid, f"{DEBUG_DIR}/epoch_{epoch}_step_{step}.png",
               nrow=b, normalize=True, value_range=(0, 1))


# ── Results Logger (CSV + plots, Ultralytics-style) ──────────────────────────

class ResultsLogger:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.csv_path = os.path.join(save_dir, "results.csv")
        self.keys = ["step", "epoch", "train/loss", "train/l1", "train/perceptual", "val/l1", "lr"]
        self._write_header()

    def _write_header(self):
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.keys)

    def log(self, step, epoch, train_loss, train_l1, train_perc, val_l1, lr):
        def fmt(v):
            return f"{v:.6f}" if v is not None else ""
        row = [step, epoch, fmt(train_loss), fmt(train_l1),
               fmt(train_perc), fmt(val_l1), f"{lr:.2e}"]
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def plot(self):
        import pandas as pd
        try:
            df = pd.read_csv(self.csv_path)
        except Exception:
            return
        if len(df) < 2:
            return

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        metrics = [
            ("train/loss", "Loss", ["train/loss"], ['blue']),
            ("train/l1", "L1", ["train/l1", "val/l1"], ['blue', 'orange']),
            ("train/perceptual", "Perceptual", ["train/perceptual"], ['green']),
        ]
        for ax, (_, title, cols, colors) in zip(axes, metrics):
            for col, color in zip(cols, colors):
                valid = df[col].dropna()
                if len(valid) > 0:
                    ax.plot(df.loc[valid.index, "step"], valid.values,
                            label=col, color=color, linewidth=1)
            ax.set_title(title)
            ax.set_xlabel("Step")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "results.png"), dpi=150)
        plt.close()


# ── Perceptual Loss ──────────────────────────────────────────────────────────

class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features.eval()
        for p in vgg.parameters():
            p.requires_grad_(False)
        blocks = []
        for name, layer in vgg.named_children():
            blocks.append(layer)
            if name == '8':   # block1_conv2 → relu1_2
                self.block1 = nn.Sequential(*blocks)
                blocks = []
            elif name == '15':  # block2_conv2 → relu2_2
                self.block2 = nn.Sequential(*blocks)
                break
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred, target):
        pred = (pred - self.mean) / self.std
        target = (target - self.mean) / self.std
        p = self.block1(pred)
        f1_p = p
        p = self.block2(p)
        f2_p = p
        t = self.block1(target)
        f1_t = t
        t = self.block2(t)
        f2_t = t
        loss = F.l1_loss(f1_p, f1_t) + F.l1_loss(f2_p, f2_t)
        return loss


# ── Training ─────────────────────────────────────────────────────────────────

def train():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(DEBUG_DIR, exist_ok=True)

    model = InpaintingModel({
        'coarse_model': {
            'class': 'MobileOneCoarse',
            'parameters': {'variant': 's4'}
        },
        'generator': {
            'generator_class': 'PatchInpainting',
            'params': {
                'kernel_size': PATCH_SIZE,
                'nheads': 1,
                'stem_out_stride': 1,
                'stem_out_channels': 3,
                'merge_mode': 'all',
                'image_size': IMAGE_SIZE,
                'embed_dim': 576,
                'use_qpos': None,
                'use_kpos': None,
                'dropout': 0.1,
                'feature_i': 3,
                'concat_features': True,
                'final_conv': True,
                'feature_dim': 896,
                'attention_type': 'MultiHeadAttention',
                'compute_v': False,
                'use_argmax': False
            }
        }
    }).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = None  # manual LR schedule

    criterion_l1 = nn.L1Loss()
    criterion_perc = PerceptualLoss().to(DEVICE)

    exp_id = 1
    while os.path.exists(os.path.join(CHECKPOINT_DIR, "..", "runs", "train", f"exp{exp_id}")):
        exp_id += 1
    run_dir = os.path.join(CHECKPOINT_DIR, "..", "runs", "train", f"exp{exp_id}")
    logger = ResultsLogger(run_dir)

    train_dataset = InpaintingDataset(TRAIN_DIR, IMAGE_SIZE)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=NUM_WORKERS, pin_memory=True,
                              collate_fn=collate_fn, drop_last=True)

    val_dataset = ValDataset(TRAIN_DIR, MASK_DIR, IMAGE_SIZE)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    def set_lr(step):
        if step < WARMUP_STEPS:
            lr = LR * (step + 1) / WARMUP_STEPS
        else:
            progress = (step - WARMUP_STEPS) / (TOTAL_STEPS - WARMUP_STEPS)
            lr = LR * 0.5 * (1 + math.cos(math.pi * progress))
        for g in optimizer.param_groups:
            g['lr'] = lr

    step = 0
    epoch = 0
    best_val_loss = float('inf')
    patience_counter = 0
    ckpt_path = os.path.join(CHECKPOINT_DIR, "latest.pt")
    best_path = os.path.join(CHECKPOINT_DIR, "best.pt")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        step = ckpt['step']
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        patience_counter = ckpt.get('patience_counter', 0)
        set_lr(step)
        print(f"Resumed from step {step} (best_val_loss={best_val_loss:.4f})")

    model.train()
    while step < TOTAL_STEPS:
        for images_hr, masks_hr in train_loader:
            images_hr = images_hr.to(DEVICE)
            masks_hr = masks_hr.to(DEVICE)

            # downsample to working resolution
            images_lr = F.interpolate(images_hr, size=IMAGE_SIZE,
                                      mode='bicubic', antialias=True)
            masks_lr = F.interpolate(masks_hr, size=IMAGE_SIZE)
            masked = images_lr * (1 - masks_lr)

            output, attn, coarse = model(masked, masks_lr)
            loss_l1 = criterion_l1(output, images_lr)
            loss_perc = criterion_perc(output, images_lr)
            loss = loss_l1 + 0.1 * loss_perc

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            set_lr(step)

            if step % LOG_INTERVAL == 0:
                lr_now = optimizer.param_groups[0]['lr']
                logger.log(step, epoch, loss.item(), loss_l1.item(), loss_perc.item(), None, lr_now)
                print(f"Step {step}/{TOTAL_STEPS} | loss={loss.item():.4f} "
                      f"(l1={loss_l1.item():.4f} perc={loss_perc.item():.4f}) | lr={lr_now:.2e}")

            if step % CHECKPOINT_INTERVAL == 0:
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'step': step,
                    'best_val_loss': best_val_loss,
                    'patience_counter': patience_counter,
                }, os.path.join(CHECKPOINT_DIR, f"step_{step}.pt"))
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'step': step,
                    'best_val_loss': best_val_loss,
                    'patience_counter': patience_counter,
                }, ckpt_path)

            if step % VAL_INTERVAL == 0:
                model.eval()
                val_loss = 0
                first_batch = True
                with torch.no_grad():
                    for images_hr, masks_hr in val_loader:
                        images_hr = images_hr.to(DEVICE)
                        masks_hr = masks_hr.to(DEVICE)
                        images_lr = F.interpolate(images_hr, size=IMAGE_SIZE,
                                                  mode='bicubic', antialias=True)
                        masks_lr = F.interpolate(masks_hr, size=IMAGE_SIZE)
                        masked = images_lr * (1 - masks_lr)
                        output, _, coarse = model(masked, masks_lr)
                        val_loss += criterion_l1(output, images_lr).item()
                        if first_batch:
                            save_debug_images(images_hr, masks_hr, output, coarse, step, epoch)
                            first_batch = False
                avg = val_loss / max(len(val_loader), 1)
                logger.log(step, epoch, None, None, None, avg, optimizer.param_groups[0]['lr'])
                logger.plot()
                print(f"Val L1: {avg:.4f}")

                # Early stopping
                if avg < best_val_loss:
                    best_val_loss = avg
                    patience_counter = 0
                    torch.save({
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'step': step,
                        'best_val_loss': best_val_loss,
                        'patience_counter': patience_counter,
                    }, best_path)
                    print(f"New best validation L1: {best_val_loss:.4f} (model saved to {best_path})")
                else:
                    patience_counter += 1
                    print(f"No improvement. Patience: {patience_counter}/{PATIENCE}")
                    if patience_counter >= PATIENCE:
                        print(f"Early stopping triggered after {step} steps. Best Val L1: {best_val_loss:.4f}")
                        model.train() # set back to train before breaking
                        break

                model.train()

            if step >= TOTAL_STEPS:
                break
        epoch += 1

    print("Training complete.")


if __name__ == '__main__':
    train()
