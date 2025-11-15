import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from PIL import Image
import json

# ============================================================================
# LOAD TRAINED MODEL & EVALUATE
# ============================================================================


class ShapeNetSplatsDataset(Dataset):
    def __init__(self, data_dir, num_views=24, limit=None):
        self.num_views = num_views
        self.data_dir = Path(data_dir)
        self.objects = sorted(
            [d for d in self.data_dir.iterdir() if d.is_dir()])
        if limit:
            self.objects = self.objects[:limit]

    def __len__(self):
        return len(self.objects)

    def __getitem__(self, idx):
        obj_dir = self.objects[idx]
        views = []
        for i in range(self.num_views):
            img_path = obj_dir / f'{i+1:03d}.png'
            img = Image.open(img_path).convert('L')
            img = torch.from_numpy(np.array(img)).float() / 255.0
            img = img.unsqueeze(0)
            views.append(img)
        return torch.stack(views)


class SplatsEncoder(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        resnet = models.resnet18(weights=None)
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7,
                                 stride=2, padding=3, bias=False)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)

    def forward(self, views):
        B, num_views, C, H, W = views.shape
        views_flat = views.reshape(B * num_views, C, H, W)
        features = self.backbone(views_flat).squeeze(-1).squeeze(-1)
        features = features.reshape(B, num_views, 512).mean(dim=1)
        return self.fc_mu(features), self.fc_logvar(features)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)


class SplatsDecoder(nn.Module):
    def __init__(self, latent_dim=64, num_splats=2048):
        super().__init__()
        hidden = 256
        self.geom = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, num_splats * 4)
        )
        self.app = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, num_splats * 2)
        )
        self.num_splats = num_splats

    def forward(self, z):
        B = z.shape[0]
        geom = self.geom(z).reshape(B, self.num_splats, 4)
        app = self.app(z).reshape(B, self.num_splats, 2)
        centers = geom[..., :3]
        radii = F.softplus(geom[..., 3:4])
        intensity = torch.sigmoid(app[..., 0:1])
        opacity = app[..., 1:2]
        return torch.cat([centers, radii, intensity, opacity,
                         torch.zeros(B, self.num_splats, 2, device=z.device)], dim=-1)


class SimpleGaussianRenderer(nn.Module):
    def __init__(self, image_size=256):
        super().__init__()
        self.size = image_size

    def forward(self, splats):
        B, K, _ = splats.shape
        grid = torch.linspace(-1, 1, self.size, device=splats.device)
        yy, xx = torch.meshgrid(grid, grid, indexing='ij')

        centers = splats[..., :3]
        radii = splats[..., 3:4]
        intensity = splats[..., 4:5]
        opacity = torch.sigmoid(splats[..., 5:6])

        x_2d = centers[..., 0:1]
        y_2d = centers[..., 1:2]

        rendered = torch.zeros(
            B, 1, self.size, self.size, device=splats.device)

        for b in range(B):
            img = torch.zeros(self.size, self.size, device=splats.device)
            for k in range(K):
                xc = x_2d[b, k, 0]
                yc = y_2d[b, k, 0]
                r = radii[b, k, 0].clamp(min=0.01)
                i = intensity[b, k, 0]
                o = opacity[b, k, 0]
                dist_sq = (xx - xc) ** 2 + (yy - yc) ** 2
                gaussian = torch.exp(-dist_sq / (2 * r ** 2))
                img = img + o * i * gaussian
            rendered[b, 0] = torch.clamp(img, 0, 1)

        return rendered


class SplatsVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = SplatsEncoder(64)
        self.decoder = SplatsDecoder(64, 2048)
        self.renderer = SimpleGaussianRenderer(256)

    def forward(self, views):
        mu, logvar = self.encoder(views)
        z = self.encoder.reparameterize(mu, logvar)
        splats = self.decoder(z)
        rendered = self.renderer(splats)
        return rendered, mu, logvar, splats

# ============================================================================
# EVALUATION METRICS
# ============================================================================


def compute_psnr(pred, target):
    """PSNR between 0-1 images."""
    # Ensure shapes match
    if pred.shape != target.shape:
        if target.dim() == 5:  # [B, 1, 1, H, W]
            target = target.squeeze(2)  # [B, 1, H, W]

    mse = F.mse_loss(pred, target)
    psnr = 10 * torch.log10(1.0 / (mse + 1e-10))
    return psnr.item()


def compute_ssim(pred, target, window_size=11):
    """Simplified SSIM."""
    if pred.shape != target.shape:
        if target.dim() == 5:
            target = target.squeeze(2)

    # Simple MSE-based metric for now
    mse = F.mse_loss(pred, target)
    return 1.0 / (1.0 + mse.item())

# ============================================================================
# EVALUATION
# ============================================================================


def evaluate(model, test_loader, device):
    """Evaluate on test set."""
    model.eval()

    psnr_scores = []
    ssim_scores = []

    print("Evaluating on test set...")
    with torch.no_grad():
        for batch_idx, views in enumerate(test_loader):
            views = views.to(device)
            rendered, mu, logvar, splats = model(views)

            # Compare with first view [B, 1, 256, 256]
            target = views[:, 0:1]

            psnr = compute_psnr(rendered, target)
            ssim = compute_ssim(rendered, target)

            psnr_scores.append(psnr)
            ssim_scores.append(ssim)

            if (batch_idx + 1) % 10 == 0:
                print(
                    f"  Batch {batch_idx+1} | PSNR: {psnr:.2f} | SSIM: {ssim:.4f}")

    avg_psnr = np.mean(psnr_scores)
    avg_ssim = np.mean(ssim_scores)

    print(f"\n✓ Test Results:")
    print(f"  Average PSNR: {avg_psnr:.2f} dB")
    print(f"  Average SSIM: {avg_ssim:.4f}")

    return {
        'avg_psnr': avg_psnr,
        'avg_ssim': avg_ssim,
        'psnr_scores': psnr_scores,
        'ssim_scores': ssim_scores
    }


def latent_interpolation(model, test_loader, device, num_steps=10):
    """Interpolate between two latent codes."""
    print("\nLatent Interpolation...")
    model.eval()

    # Get two random objects
    z1, z2 = None, None
    with torch.no_grad():
        for views in test_loader:
            views = views.to(device)
            mu, _ = model.encoder(views)
            if z1 is None:
                z1 = mu[0:1]
            else:
                z2 = mu[0:1]
                break

    # Interpolate
    z_interp = []
    for t in np.linspace(0, 1, num_steps):
        z_t = (1 - t) * z1 + t * z2
        z_interp.append(z_t)

        rendered = model.renderer(model.decoder(z_t))
        print(f"  t={t:.2f} | rendered shape: {rendered.shape}")

    return z_interp


def sample_novel_objects(model, device, num_samples=5):
    """Sample random objects from latent space."""
    print("\nSampling Novel Objects...")
    model.eval()

    samples = []
    with torch.no_grad():
        for i in range(num_samples):
            z = torch.randn(1, 64, device=device)
            splats = model.decoder(z)
            rendered = model.renderer(splats)
            samples.append(rendered)
            print(f"  Sample {i+1}: {rendered.shape}")

    return samples

# ============================================================================
# MAIN
# ============================================================================


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")

    # Load model
    print("Loading trained model...")
    model = SplatsVAE().to(device)
    model.load_state_dict(torch.load('vae_quick_test.pt', map_location=device))
    print("✓ Model loaded\n")

    # Load test data
    data_path = r"C:\Users\NJ\Desktop\DL\Shapenet\ShapeNetCore.v2\ShapeNetCore.v2\03001627_imgs\aligned"
    test_dataset = ShapeNetSplatsDataset(data_path, limit=100)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    # Evaluation
    print("=" * 60)
    print("CHECK-IN #4: EVALUATION & ANALYSIS")
    print("=" * 60 + "\n")

    # 1. Quantitative metrics
    metrics = evaluate(model, test_loader, device)

    # 2. Latent interpolation
    z_interp = latent_interpolation(model, test_loader, device)

    # 3. Novel samples
    samples = sample_novel_objects(model, device, num_samples=5)

    # Save results
    results = {
        'metrics': {
            'psnr': metrics['avg_psnr'],
            'ssim': metrics['avg_ssim']
        },
        'interpolation': 'See z_interp list',
        'samples': 'See samples list'
    }

    with open('evaluation_results.json', 'w') as f:
        json.dump({
            'avg_psnr': metrics['avg_psnr'],
            'avg_ssim': metrics['avg_ssim']
        }, f, indent=2)

    print("\n" + "=" * 60)
    print("✓ Evaluation complete!")
    print("Results saved: evaluation_results.json")
    print("=" * 60)
