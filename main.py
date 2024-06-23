# train_dfgan.py

import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image, UnidentifiedImageError
from pycocotools.coco import COCO
from torchvision import datasets, transforms
from tqdm import tqdm
from transformers import BertModel, BertTokenizer


# Define COCODataset
class COCODataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, ann_file, transform=None):
        self.img_dir = img_dir
        self.coco = COCO(ann_file)
        self.ids = list(self.coco.anns.keys())
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        ann_id = self.ids[idx]
        caption = self.coco.anns[ann_id]['caption']
        img_id = self.coco.anns[ann_id]['image_id']
        img_info = self.coco.loadImgs(img_id)[0]
        path = img_info['file_name']
        img_path = os.path.join(self.img_dir, path)
        try:
            image = Image.open(img_path).convert('RGB')
        except (OSError, UnidentifiedImageError) as e:
            print(f"Error loading image {img_path}: {e}")
            return None, None
        if self.transform:
            image = self.transform(image)
        return image, caption

# Define TextEncoder
class TextEncoder(nn.Module):
    def __init__(self, embedding_dim):
        super(TextEncoder, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.linear = nn.Linear(self.bert.config.hidden_size, embedding_dim)

    def forward(self, text):
        with torch.no_grad():
            encoded_layers = self.bert(text)['last_hidden_state']
        features = self.linear(encoded_layers[:, 0, :])
        return features

# Define Generator
class Generator(nn.Module):
    def __init__(self, text_dim, noise_dim, img_size):
        super(Generator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(text_dim + noise_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, img_size * img_size * 3),
            nn.Tanh()
        )

    def forward(self, text, noise):
        x = torch.cat([text, noise], 1)
        img = self.fc(x)
        img = img.view(img.size(0), 3, img_size, img_size)
        return img

# Define Discriminator
class Discriminator(nn.Module):
    def __init__(self, img_size, text_dim):
        super(Discriminator, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.fc = nn.Linear(256 * 8 * 8 + text_dim, 1)

    def forward(self, img, text):
        x = self.conv(img)
        x = x.view(x.size(0), -1)
        x = torch.cat([x, text], 1)
        validity = self.fc(x)
        return validity

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

# Hyperparameters
text_dim = 768
noise_dim = 100
img_size = 64
lr = 0.0002
b1 = 0.5
b2 = 0.999
epochs = 100
save_model_path = "dfgan_model.pth"

# Data transformations
transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.CenterCrop(img_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Load dataset
dataset = COCODataset(img_dir='val201', ann_file='annotations/captions_val2017.json', transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)


# Initialize tokenizer and text encoder
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
text_encoder = TextEncoder(text_dim)

# Initialize models
generator = Generator(text_dim, noise_dim, img_size)
discriminator = Discriminator(img_size, text_dim)

generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# Loss function
adversarial_loss = torch.nn.BCEWithLogitsLoss()

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

# Initialize lists to track losses
G_losses = []
D_losses = []

# Training loop
for epoch in range(epochs):
    for i, (imgs, captions) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)):
        if imgs is None or captions is None:
            continue

        batch_size = imgs.size(0)

        # Adversarial ground truths
        valid = torch.ones(batch_size, 1, requires_grad=False)
        fake = torch.zeros(batch_size, 1, requires_grad=False)

        # Configure input
        real_imgs = imgs.type(torch.FloatTensor)
        encoded_text = tokenizer(captions, return_tensors='pt', padding=True, truncation=True)['input_ids']

        # -----------------
        #  Train Generator
        # -----------------
        optimizer_G.zero_grad()

        z = torch.randn(batch_size, noise_dim)
        gen_text = text_encoder(encoded_text)
        gen_imgs = generator(gen_text, z)

        g_loss = adversarial_loss(discriminator(gen_imgs, gen_text), valid)

        g_loss.backward(retain_graph=True)
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()

        real_loss = adversarial_loss(discriminator(real_imgs, gen_text), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach(), gen_text), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        # Save losses for plotting later
        G_losses.append(g_loss.item())
        D_losses.append(d_loss.item())

        tqdm.write(f"[Epoch {epoch}/{epochs}] [Batch {i}/{len(dataloader)}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")

# Save the trained model
torch.save({
    'generator_state_dict': generator.state_dict(),
    'discriminator_state_dict': discriminator.state_dict(),
    'optimizer_G_state_dict': optimizer_G.state_dict(),
    'optimizer_D_state_dict': optimizer_D.state_dict(),
}, save_model_path)

print("Training finished and model saved.")

# Plot the training losses
plt.figure(figsize=(10, 5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses, label="G")
plt.plot(D_losses, label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()
