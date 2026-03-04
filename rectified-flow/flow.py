import torch
import torch.nn as nn
import torch.optim as optim
from einops import rearrange, reduce, repeat

device = "cuda" if torch.cuda.is_available() else "cpu"


# dummy encoder

class Encoder(nn.Module):
    def __init__(self, latent_channels=256):
        super().__init__()

        self.net = nn.Sequential(

            nn.Conv2d(3, 32, 4, 2, 1),   # 64 → 32
            nn.ReLU(),

            nn.Conv2d(32, 64, 4, 2, 1),  # 32 → 16
            nn.ReLU(),

            nn.Conv2d(64, 128, 4, 2, 1), # 16 → 8
            nn.ReLU(),

            nn.Conv2d(128, latent_channels, 3, 1, 1) # keep 8x8
        )

    def forward(self, x):
        return self.net(x)
    


# transformer decoder
class FlowTransformer(nn.Module):
    def __init__(self, latent_dim=256, nhead=8, num_layers=4):
        super().__init__()
        
        self.time_embed = nn.Sequential(
            nn.Linear(1, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=nhead,
            batch_first=True,
            norm_first=True 
        )

        # bidirectional transformer 
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, latent, t):
    
        b, c, h, w = latent.shape

        # patchify
        latent = rearrange(latent, 'b c h w -> b (h w) c')
        t = rearrange(t, 'b -> b 1') # [B, 1]

        t_embed = self.time_embed(t) # [B, C]
        t_embed = rearrange(t_embed, 'b c -> b 1 c') # [B, 1, C]

        latent = latent + t_embed 

        flow = self.transformer(latent)
        
        # unpatchify
        flow = rearrange(flow, 'b (h w) c -> b c h w', h=h, w=w)
        return flow



encoder = Encoder().to(device)
model = FlowTransformer().to(device)


optimizer = optim.Adam(model.parameters(), lr=1e-4)

def sample_images(batch):
    return torch.randn(batch,3,224,224).to(device)


batch_size = 32

# Training Loop 
for step in range(1000):
    images = sample_images(batch_size)

    with torch.no_grad():
        latent = encoder(images)  # x0 (Target/Data)
    
    noise = torch.randn_like(latent) # Noise

    t = torch.rand(batch_size).to(device) # Random time steps in [0, 1]

    # t=0 for noise and t=1 for data

    t_padded = rearrange(t, 'b -> b 1 1 1')

    noised_latent = noise * (1 - t_padded) + latent * t_padded

    target_flow = latent - noise

    pred_flow = model(noised_latent, t)

    # MSE
    loss = torch.nn.functional.mse_loss(pred_flow, target_flow)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Step {step}: Loss = {loss.item():.4f}")
    break