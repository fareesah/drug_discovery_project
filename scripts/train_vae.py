import torch
from torch.utils.data import DataLoader
from models.vae_model import VAE
from utils.data_utils import SMILESDataset

# Load data and initialize model
dataset = SMILESDataset(pd.read_csv("data/molecules.csv")['smiles'])
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
vae = VAE()
optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)

def train():
    for epoch in range(10):
        for batch in dataloader:
            batch = batch.view(-1, 1024)
            recon_batch, mu, logvar = vae(batch)
            loss = vae_loss(recon_batch, batch, mu, logvar)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch}, Loss: {loss.item()}")

def vae_loss(recon_x, x, mu, logvar):
    BCE = nn.BCELoss(reduction='sum')(recon_x, x)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

if __name__ == "__main__":
    train()
