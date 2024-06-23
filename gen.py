# generate_images.py

import matplotlib.pyplot as plt
import torch
from torch import nn
from torchvision.utils import make_grid
from transformers import BertModel, BertTokenizer

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Define TextEncoder, Generator, and other required classes (as in train_dfgan.py)
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

# Hyperparameters
text_dim = 768
noise_dim = 100
img_size = 64
save_model_path = "dfgan_model.pth"

# Load the model
checkpoint = torch.load(save_model_path, map_location=device)

# Initialize tokenizer and text encoder
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
text_encoder = TextEncoder(text_dim).to(device)

# Initialize models
generator = Generator(text_dim, noise_dim, img_size).to(device)
generator.load_state_dict(checkpoint['generator_state_dict'])
generator.eval()

def generate_image_from_text(text):
    encoded_text = tokenizer([text], return_tensors='pt', padding=True, truncation=True)['input_ids'].to(device)
    gen_text = text_encoder(encoded_text)
    z = torch.randn(1, noise_dim).to(device)
    gen_img = generator(gen_text, z)
    return gen_img

# Example usage
text_description = "text written"
generated_image = generate_image_from_text(text_description)

# Plot the generated image
plt.imshow(make_grid(generated_image, normalize=True).permute(1, 2, 0).detach().cpu().numpy())
plt.axis('off')
plt.show()
