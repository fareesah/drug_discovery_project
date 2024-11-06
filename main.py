from scripts.train_vae import train as train_vae
from scripts.train_affinity import train as train_affinity

if __name__ == "__main__":
    print("Training VAE Model...")
    train_vae()
    print("Training Affinity Prediction Model...")
    train_affinity()
