# HACKING — RETHINED

Modificaciones necesarias para hacer correr el modelo, ya que el repo oficial está incompleto (solo lo justo para la conferencia WACV 2025 Oral).

## Parches aplicados

### 1. `mobileone.py` — Backbone faltante

El `model.py` hace `from mobileone import MobileOne, mobileone` pero el archivo no existe en el repo. Descargué la implementación oficial de Apple (`apple/ml-mobileone/mobileone.py`) y la puse en la raíz del proyecto.

### 2. Canales del decoder en `MobileOneCoarse` (model.py:268-272)

El decoder tenía canales hardcodeados que no coincidían con las salidas reales de MobileOne-S4:

| Capa | Original (roto) | Corregido |
|------|-----------------|-----------|
| d4   | 2048 → 1792     | 2048 → 896 |
| d3   | 1792+1792 → 896 | 896+896 → 448 |
| d2   | 896+896 → 384   | 448+448 → 192 |
| d1   | 384+384 → 64    | 192+192 → 64 |
| d0   | 64+64 → 3       | 64+64 → 3 |

MobileOne-S4 produce: x0=64, x1=192, x2=448, x3=896, x4=2048. El decoder original asumía canales distintos (posiblemente de otra variante o de un error de transcripción).

### 3. `feature_i` en config (model.py:377)

El config tenía `feature_i: 2` pero `feature_dim: 896`. Con MobileOne-S4:
- features[2] = x2 = 448 canales
- features[3] = x3 = 896 canales

Cambié `feature_i: 2 → 3` para que el feature de skip connection (896) coincida con `feature_dim`.

## Lo que falta

### Pesos pre-entrenados
No hay checkpoints publicados. El modelo corre con pesos aleatorios. Sin los pesos oficiales hay que entrenar desde cero.

### Training script — `train.py`
Creado desde cero. Según el paper:
- Optimizer: Adam, lr=1e-3 con warmup (5K steps) + cosine decay
- Batch size objetivo: 128 (ajustar `BATCH_SIZE` a lo que quepa en tu GPU)
- Steps: 600K
- GPU: NVIDIA RTX 4090 (los autores)
- Máscaras: generación aleatoria tipo LaMa en cada iteración (brush strokes + rectángulos)
- Loss: L1 + Perceptual (VGG16 relu2_2 + relu3_3, peso 0.1)
- Dataset: carga imágenes HR de `data/DF2K/HR/`, las redimensiona a 1024 y hace center crop, luego genera máscara aleatoria, y en training las baja a 512×512 con antialiasing
- Validation: usa las máscaras fijas de `test_masks/DF8K-Inpainting/masks/test/`
- Checkpoints: se guardan en `checkpoints/` cada 5000 steps

Para correr:
```bash
source .venv/bin/activate
python train.py
```

Ajustes disponibles en el header de `train.py`: `BATCH_SIZE`, `TOTAL_STEPS`, `WARMUP_STEPS`, etc.

### Inference script
Solo hay un `__main__` de prueba con tensores dummy. Falta:
- Cargar imagen + máscara desde archivo
- Downsampling a 512×512 con antialiasing
- Forward pass por el modelo
- AttentionUpscaling a resolución original
- Guardar resultado

### Dataset DF8K-Inpainting
- Compuesto por DF2K (DIV2K + Flickr2K) + CAFHQ
- Máscaras free-form solo para test (Google Drive: `1BzTVrzZ5Z4rKPPp0K5SO5fRe01Fs4dw5`)
- No hay script de descarga funcional (`bin/download_dataset.sh` no existe)

### Export a mobile
El paper reporta latencia en iPhone, iPad, Snapdragon, Jetson Nano via ONNX/CoreML. El código ya tiene `unfolding_coreml` y `folding_coreml` preparados para esto, pero no hay script de exportación.

## Dependencias

```
torch, torchvision, numpy, einops, kornia
```
