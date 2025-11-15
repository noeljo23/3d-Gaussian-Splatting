import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import json
from datetime import datetime

# ============================================================================
# DATASET
# ============================================================================


class ShapeNetSplatsDataset(Dataset):
    """Load rendered multiview images."""

    def __init__(self, data_dir=None, num_views=24, limit=None):
        self.num_views = num_views
        if data_dir is None:
            data_dir = Path(
                r"C:\Users\NJ\Desktop\DL\Shapenet\ShapeNetCore.v2\ShapeNetCore.v2\03001627_imgs\aligned")

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

# ============================================================================
# ENCODER
# ============================================================================


class SplatsEncoder(nn.Module):
    """Encode 24 views into 64-dim latent."""

    def __init__(self, latent_dim=64):
        super().__init__()
        self.latent_dim = latent_dim

        resnet = models.resnet18(weights=None)
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7,
                                 stride=2, padding=3, bias=False)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.feat_dim = 512

        self.fc_mu = nn.Linear(self.feat_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.feat_dim, latent_dim)

    def forward(self, views):
        B, num_views, C, H, W = views.shape
        views_flat = views.reshape(B * num_views, C, H, W)
        features = self.backbone(views_flat)
        features = features.squeeze(-1).squeeze(-1)
        features = features.reshape(B, num_views, self.feat_dim)
        avg_feat = features.mean(dim=1)

        mu = self.fc_mu(avg_feat)
        logvar = self.fc_logvar(avg_feat)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

# ============================================================================
# DECODER
# ============================================================================


class SplatsDecoder(nn.Module):
    """Decode latent into 2048 Gaussian splats."""

    def __init__(self, latent_dim=64, num_splats=2048):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_splats = num_splats
        hidden_dim = 512

        self.geometry_mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_splats * 4)
        )

        self.appearance_mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_splats * 2)
        )

    def forward(self, z):
        B = z.shape[0]

        geom = self.geometry_mlp(z)
        geom = geom.reshape(B, self.num_splats, 4)
        centers = geom[..., :3]
        radii = F.softplus(geom[..., 3:4])

        app = self.appearance_mlp(z)
        app = app.reshape(B, self.num_splats, 2)
        intensity = torch.sigmoid(app[..., 0:1])
        opacity = app[..., 1:2]

        splats = torch.cat([centers, radii, intensity, opacity,
                           torch.zeros(B, self.num_splats, 2, device=z.device)], dim=-1)
        return splats

# ============================================================================
# SIMPLE RENDERER (placeholder for gsplat integration)
# ============================================================================


class SimpleGaussianRenderer(nn.Module):
    """Simple differentiable Gaussian renderer."""

    def __init__(self, image_size=256):
        super().__init__()
        self.size = image_size

    def forward(self, splats):
        """Render first view only (simplified for training)."""
        B, K, _ = splats.shape

        # Create grid
        grid = torch.linspace(-1, 1, self.size, device=splats.device)
        yy, xx = torch.meshgrid(grid, grid, indexing='ij')

        centers = splats[..., :3]  # [B, K, 3]
        radii = splats[..., 3:4]   # [B, K, 1]
        intensity = splats[..., 4:5]  # [B, K, 1]
        opacity = torch.sigmoid(splats[..., 5:6])  # [B, K, 1]

        # Project to 2D (simple orthographic projection)
        x_2d = centers[..., 0:1]  # [B, K, 1]
        y_2d = centers[..., 1:2]  # [B, K, 1]

        # Render each splat as Gaussian
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

# ============================================================================
# VAE MODEL
# ============================================================================


class SplatsVAE(nn.Module):
    """Full VAE pipeline."""

    def __init__(self, latent_dim=64, num_splats=2048):
        super().__init__()
        self.encoder = SplatsEncoder(latent_dim)
        self.decoder = SplatsDecoder(latent_dim, num_splats)
        self.renderer = SimpleGaussianRenderer(256)

    def forward(self, views):
        mu, logvar = self.encoder(views)
        z = self.encoder.reparameterize(mu, logvar)
        splats = self.decoder(z)
        rendered = self.renderer(splats)
        return rendered, mu, logvar, splats

# ============================================================================
# LOSS & METRICS
# ============================================================================


def compute_psnr(pred, target):
    """PSNR between 0-1 images."""
    mse = F.mse_loss(pred, target)
    psnr = 10 * torch.log10(1.0 / (mse + 1e-10))
    return psnr


def compute_ssim(pred, target, window_size=11):
    """Simple SSIM approximation."""
    return F.mse_loss(pred, target)  # Placeholder


def vae_loss(rendered, target_views, mu, logvar, splats, beta_kl=1.0,
             lambda_size=1e-4, lambda_opacity=1e-4):
    """Total loss with KL warmup."""

    # L1 photometric loss (compare first view)
    l1_loss = F.l1_loss(rendered, target_views[:, 0:1])

    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) -
                               logvar.exp()) / rendered.shape[0]

    # Size regularizer
    radii = splats[..., 3]
    size_reg = lambda_size * torch.sum(radii ** 2) / rendered.shape[0]

    # Opacity regularizer
    opacity = splats[..., 5]
    opacity_reg = lambda_opacity * \
        torch.sum(torch.sigmoid(opacity)) / rendered.shape[0]

    total = l1_loss + beta_kl * kl_loss + size_reg + opacity_reg
    return total, l1_loss, kl_loss

# ============================================================================
# TRAINING
# ============================================================================


def train_vae(model, train_loader, val_loader, epochs=100, lr=1e-3, device='cuda'):
    """Train VAE with KL warmup."""

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)

    logs = []

    for epoch in range(epochs):
        # KL warmup: 0→1 over 20 epochs
        if epoch < 20:
            beta_kl = epoch / 20.0
        else:
            beta_kl = 1.0

        # Training
        model.train()
        train_loss = 0
        for views in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            views = views.to(device)

            rendered, mu, logvar, splats = model(views)
            loss, l1, kl = vae_loss(
                rendered, views, mu, logvar, splats, beta_kl=beta_kl)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        val_psnr = 0
        with torch.no_grad():
            for views in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                views = views.to(device)
                rendered, mu, logvar, splats = model(views)
                loss, _, _ = vae_loss(
                    rendered, views, mu, logvar, splats, beta_kl=1.0)
                psnr = compute_psnr(rendered, views[:, 0:1])

                val_loss += loss.item()
                val_psnr += psnr.item()

        val_loss /= len(val_loader)
        val_psnr /= len(val_loader)

        log_entry = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_psnr': val_psnr,
            'beta_kl': beta_kl
        }
        logs.append(log_entry)

        print(
            f"\nEpoch {epoch+1}/{epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | PSNR: {val_psnr:.2f} | β_KL: {beta_kl:.3f}")

        # Save checkpoint every 40 epochs
        if (epoch + 1) % 40 == 0:
            ckpt_path = f'checkpoint_epoch_{epoch+1}.pt'
            torch.save({
                'epoch': epoch + 1,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'logs': logs
            }, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

    return model, logs

# ============================================================================
# MAIN
# ============================================================================


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")

    # Create datasets
    print("Loading training data...")
    train_dataset = ShapeNetSplatsDataset(limit=1000)  # Use 1000 chairs
    val_dataset = ShapeNetSplatsDataset(limit=100)

    train_loader = DataLoader(
        train_dataset, batch_size=8, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=8,
                            shuffle=False, num_workers=0)

    # Create model
    model = SplatsVAE(latent_dim=64, num_splats=2048)

    # Train
    print("\nStarting training...")
    model, logs = train_vae(model, train_loader, val_loader,
                            epochs=100, lr=1e-3, device=device)

    # Save final model
    torch.save(model.state_dict(), 'vae_final.pt')

    # Save logs
    with open('training_logs.json', 'w') as f:
        json.dump(logs, f, indent=2)

    print("\n✓ Training complete!")
    print(f"Final checkpoint: vae_final.pt")
    print(f"Logs saved: training_logs.json")
