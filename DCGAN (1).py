# ===============================
# Full Colab GAN Script (DCGAN style)
# ===============================

# 1️⃣ Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 2️⃣ Install dependencies
!pip install torch torchvision torchaudio fastai lpips

# 3️⃣ Import libraries
import os
from fastai.vision.all import *
import torch
from torchvision.utils import save_image

# 4️⃣ Paths
DATASET = r"/content/drive/MyDrive/final_dataset"   # Your 5k-image folder
FAKE_IMAGES = "/content/drive/MyDrive/fake_images" # Where fake images will be saved
os.makedirs(FAKE_IMAGES, exist_ok=True)

# 5️⃣ Parameters
IMG_SIZE = 128
BATCH_SIZE = 64
EPOCHS = 20         # Increase for better results
LATENT_DIM = 100
NUM_FAKE_IMAGES = 500  # Number of images to generate

# 6️⃣ DataLoaders
dblock = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.0),
    get_y=lambda x: 0,
    item_tfms=Resize(IMG_SIZE)
)
dls = dblock.dataloaders(DATASET, bs=BATCH_SIZE)

# 7️⃣ Define simple DCGAN generator
class Generator(nn.Module):
    def __init__(self, latent_dim=100, img_size=128, channels=3):
        super().__init__()
        self.init_size = img_size // 8
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, channels, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

# 8️⃣ Define simple DCGAN discriminator
class Discriminator(nn.Module):
    def __init__(self, img_size=128, channels=3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(256*(img_size//8)*(img_size//8), 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        return self.model(img)

# 9️⃣ Initialize models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = Generator(LATENT_DIM, IMG_SIZE).to(device)
discriminator = Discriminator(IMG_SIZE).to(device)

#  🔟 Loss and optimizers
adversarial_loss = torch.nn.BCELoss()
optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# 1️⃣1️⃣ Training loop
for epoch in range(EPOCHS):
    for i, (imgs, _) in enumerate(dls.train):
        real_imgs = imgs.to(device)
        batch_size = real_imgs.size(0)

        # Adversarial ground truths
        valid = torch.ones(batch_size, 1, device=device)
        fake = torch.zeros(batch_size, 1, device=device)

        # -----------------
        #  Train Generator
        # -----------------
        optimizer_G.zero_grad()
        z = torch.randn(batch_size, LATENT_DIM, device=device)
        gen_imgs = generator(z)
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)
        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()

    print(f"[Epoch {epoch+1}/{EPOCHS}] D_loss: {d_loss.item():.4f} G_loss: {g_loss.item():.4f}")

# 1️⃣2️⃣ Generate fake images
generator.eval()
with torch.no_grad():
    z = torch.randn(NUM_FAKE_IMAGES, LATENT_DIM, device=device)
    gen_imgs = generator(z)
    gen_imgs = (gen_imgs + 1) / 2.0  # scale to [0,1]

    for idx, img in enumerate(gen_imgs):
        save_image(img, os.path.join(FAKE_IMAGES, f"fake_{idx+1}.png"))

print(f"✅ Generated {NUM_FAKE_IMAGES} fake images in {FAKE_IMAGES}")
