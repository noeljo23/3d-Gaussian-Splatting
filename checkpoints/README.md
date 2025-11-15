# Model Checkpoints

## Quick Validation Model
- **Filename:** vae_quick_test.pt
- **Training:** 50 objects, 10 epochs on RTX 4060
- **Size:** ~50MB
- **Purpose:** Quick validation, pipeline testing

**Load:**
```python
from main import SplatsVAE
model = SplatsVAE()
model.load_state_dict(torch.load('vae_quick_test.pt'))
```

## Full Trained Model
- **Filename:** vae_final.pt
- **Training:** 1,000 objects, 100 epochs on H200 GPU
- **Size:** ~50MB
- **Status:** Training on  HPC

# Model Checkpoints

## Quick Validation Model
- **Filename:** vae_quick_test.pt (~50MB)
- **Training:** 50 objects, 10 epochs
- **Status:** Available locally, too large for GitHub
- **Download:** Train locally with `python quick_train.py`

## Full Trained Model
- **Filename:** vae_final.pt (~50MB)
- **Training:** 1,000 objects, 100 epochs on H200

## Training Logs
- `training_logs.json` - Loss/PSNR per epoch (from vae_final.pt)

