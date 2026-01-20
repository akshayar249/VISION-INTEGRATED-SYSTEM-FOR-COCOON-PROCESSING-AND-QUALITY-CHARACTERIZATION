# ===============================
# Colab-ready FastGAN-style GAN
# ===============================

# 1️⃣ Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 2️⃣ Install dependencies
!pip install torch torchvision torchaudio lpips

# 3️⃣ Import libraries
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

# 4️⃣ Paths
DATASET = "/content/drive/MyDrive/final_dataset"   # Your 5k-image folder
FAKE_IMAGES = "/content/drive/MyDrive/fake_images" # Where fake images will be saved
os.makedirs(FAKE_IMAGES, exist_ok=True)

# 5️⃣ Parameters
IMG_SIZE = 128
BATCH_SIZE = 64
EPOCHS = 20
LATENT_DIM = 100
NUM_FAKE_IMAGES = 500
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 6️⃣ Dataset class
class ImageFolderDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.files = [os.path.join(root, f) for f in os.listdir(root) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img

# 7️⃣ Transform
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])
dataset = ImageFolderDataset(DATASET, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

# 8️⃣ FastGAN-style Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.shortcut = nn.Identity()
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

# 9️⃣ Generator
class Generator(nn.Module):
    def __init__(self, latent_dim=100, img_size=128, channels=3):
        super().__init__()
        self.init_size = img_size // 8
        self.l1 = nn.Linear(latent_dim, 128*self.init_size*self.init_size)
        self.res_blocks = nn.Sequential(
            ResidualBlock(128,128),
            nn.Upsample(scale_factor=2),
            ResidualBlock(128,64),
            nn.Upsample(scale_factor=2),
            ResidualBlock(64,32),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(32, channels, 3, 1, 1),
            nn.Tanh()
        )
    def forward(self, z):
        out = self.l1(z).view(z.size(0),128,self.init_size,self.init_size)
        img = self.res_blocks(out)
        return img

# 🔟 Discriminator
class Discriminator(nn.Module):
    def __init__(self, img_size=128, channels=3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(channels,64,4,2,1),
            nn.LeakyReLU(0.2,True),
            nn.Conv2d(64,128,4,2,1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2,True),
            nn.Conv2d(128,256,4,2,1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2,True),
            nn.Flatten(),
            nn.Linear(256*(img_size//8)*(img_size//8),1),
            nn.Sigmoid()
        )
    def forward(self,x):
        return self.model(x)

# 1️⃣1️⃣ Initialize models
generator = Generator(LATENT_DIM, IMG_SIZE).to(DEVICE)
discriminator = Discriminator(IMG_SIZE).to(DEVICE)

# Loss and optimizers
criterion = nn.BCELoss()
optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5,0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5,0.999))

# 1️⃣2️⃣ Training loop
for epoch in range(EPOCHS):
    for i, imgs in enumerate(dataloader):
        real_imgs = imgs.to(DEVICE)
        batch_size = real_imgs.size(0)
        valid = torch.ones(batch_size,1,device=DEVICE)
        fake = torch.zeros(batch_size,1,device=DEVICE)

        # Train Generator
        optimizer_G.zero_grad()
        z = torch.randn(batch_size,LATENT_DIM,device=DEVICE)
        gen_imgs = generator(z)
        g_loss = criterion(discriminator(gen_imgs), valid)
        g_loss.backward()
        optimizer_G.step()

        # Train Discriminator
        optimizer_D.zero_grad()
        real_loss = criterion(discriminator(real_imgs), valid)
        fake_loss = criterion(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss)/2
        d_loss.backward()
        optimizer_D.step()

    print(f"[Epoch {epoch+1}/{EPOCHS}] D_loss: {d_loss.item():.4f} G_loss: {g_loss.item():.4f}")

# 1️⃣3️⃣ Generate fake images
generator.eval()
with torch.no_grad():
    z = torch.randn(NUM_FAKE_IMAGES,LATENT_DIM,device=DEVICE)
    gen_imgs = generator(z)
    gen_imgs = (gen_imgs + 1)/2  # Scale to [0,1]
    for idx,img in enumerate(gen_imgs):
        save_image(img, os.path.join(FAKE_IMAGES,f"fake_{idx+1}.png"))

print(f"✅ Generated {NUM_FAKE_IMAGES} fake images in {FAKE_IMAGES}")
