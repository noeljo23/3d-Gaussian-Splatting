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
- **Expected:** Available November 15, 2025
- **Status:** Training on Northeastern HPC

**Download when ready:**
```bash
# From HPC
sftp john.noe@login.discovery.neu.edu
cd projects/gaussian_splats_vae
get vae_final.pt
quit
```

## Training Logs
- `training_logs.json` - Loss/PSNR per epoch (from vae_final.pt)
