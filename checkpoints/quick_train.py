import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from PIL import Image

# ============================================================================
# QUICK LOCAL TRAINING - 10 epochs on 50 objects
# ============================================================================


class ShapeNetSplatsDataset(Dataset):
    def __init__(self, data_dir, num_views=24, limit=50):
        self.num_views = num_views
        self.data_dir = Path(data_dir)
        self.objects = sorted(
            [d for d in self.data_dir.iterdir() if d.is_dir()])[:limit]
        print(f"Loaded {len(self.objects)} objects")

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


class SplatsVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = SplatsEncoder(64)
        self.decoder = SplatsDecoder(64, 2048)

    def forward(self, views):
        mu, logvar = self.encoder(views)
        z = self.encoder.reparameterize(mu, logvar)
        splats = self.decoder(z)
        return mu, logvar, splats

# ============================================================================
# TRAINING
# ============================================================================


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")

    # Update this path to your local images folder
    data_path = r"C:\Users\NJ\Desktop\DL\Shapenet\ShapeNetCore.v2\ShapeNetCore.v2\03001627_imgs\aligned"

    dataset = ShapeNetSplatsDataset(data_path, limit=50)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = SplatsVAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("Training 10 epochs on 50 objects...\n")

    for epoch in range(10):
        total_loss = 0
        for views in loader:
            views = views.to(device)
            mu, logvar, splats = model(views)

            # KL loss
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) -
                                  logvar.exp()) / views.shape[0]

            # Regularization
            reg = 0.0001 * \
                (torch.sum(splats[..., 3]**2) +
                 torch.sum(torch.sigmoid(splats[..., 5])))

            loss = kl + reg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/10 | Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), 'vae_quick_test.pt')
    print("\n✓ Done! Saved: vae_quick_test.pt")
