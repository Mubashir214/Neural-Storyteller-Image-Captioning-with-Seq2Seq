import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pickle

# ------------------ CONFIG ------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HIDDEN_SIZE = 512
EMBED_SIZE = 512
NUM_LAYERS = 1
MAX_LEN = 40

st.set_page_config(page_title="Neural Storyteller", layout="centered")
st.title("🧠 Neural Storyteller - Image Captioning")

# ------------------ LOAD VOCAB ------------------
with open("word_to_id.pkl", "rb") as f:
    word_to_id = pickle.load(f)

with open("id_to_word.pkl", "rb") as f:
    id_to_word = pickle.load(f)

vocab_size = len(word_to_id)

# ------------------ MODEL CLASSES ------------------
class Encoder(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.fc = nn.Linear(2048, hidden_size)

    def forward(self, x):
        return self.fc(x)

class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=word_to_id['<pad>'])
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.num_layers = num_layers
        self.hidden_size = hidden_size

    def forward(self, features, captions):
        embeds = self.embed(captions[:, :-1])
        h0 = features.unsqueeze(0).repeat(self.num_layers, 1, 1)
        c0 = torch.zeros_like(h0)
        out, _ = self.lstm(embeds, (h0, c0))
        logits = self.fc(out)
        return logits

# ------------------ LOAD MODELS ------------------
encoder = Encoder(HIDDEN_SIZE).to(DEVICE)
decoder = Decoder(EMBED_SIZE, HIDDEN_SIZE, vocab_size, NUM_LAYERS).to(DEVICE)

encoder.load_state_dict(torch.load("encoder.pth", map_location=DEVICE))
decoder.load_state_dict(torch.load("decoder.pth", map_location=DEVICE))

encoder.eval()
decoder.eval()

# ------------------ RESNET FEATURE EXTRACTOR ------------------
resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
resnet = nn.Sequential(*list(resnet.children())[:-1])
resnet = resnet.to(DEVICE)
resnet.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))
])

# ------------------ GREEDY DECODING ------------------
@torch.no_grad()
def generate_caption(feature):
    feature = feature.unsqueeze(0).to(DEVICE)
    enc_out = encoder(feature)

    hidden = enc_out.unsqueeze(0).repeat(NUM_LAYERS, 1, 1)
    cell = torch.zeros_like(hidden)

    input_token = torch.tensor([[word_to_id['<start>']]], device=DEVICE)
    caption = []

    for _ in range(MAX_LEN):
        embed = decoder.embed(input_token)
        output, (hidden, cell) = decoder.lstm(embed, (hidden, cell))
        logit = decoder.fc(output.squeeze(1))
        pred = logit.argmax(-1).item()

        if pred == word_to_id['<end>']:
            break

        caption.append(id_to_word[pred])
        input_token = torch.tensor([[pred]], device=DEVICE)

    return " ".join(caption)

# ------------------ STREAMLIT UI ------------------
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        feature = resnet(img_tensor).squeeze()

    caption = generate_caption(feature)
    st.subheader("📝 Generated Caption:")
    st.success(caption)
