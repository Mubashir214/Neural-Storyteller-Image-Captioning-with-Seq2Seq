# 🧠 Neural Storyteller – Image Captioning with Seq2Seq

## README.md

# 📸 Image Captioning using Seq2Seq (Neural Storyteller)

This project implements a **multimodal deep learning system** that generates natural language descriptions for images using a **Sequence-to-Sequence (Seq2Seq)** architecture.

The model combines:

* Pretrained CNN (ResNet50) for feature extraction
* Recurrent Neural Network (LSTM/GRU) for caption generation
* Encoder-Decoder framework for image-to-text translation

---

# 🚀 Live Demo

### Streamlit Application

[Neural Storyteller Image Captioning App](https://neural-storyteller-image-captioning-with-seq2seq-afkhezegpeekr.streamlit.app/?utm_source=chatgpt.com)

Try the model live:

* Upload an image
* Generate captions instantly
* Compare predicted vs ground truth captions

---

# 📌 Project Objective

The main goals of this project are:

* Generate meaningful captions for images
* Learn image-to-text mapping using deep learning
* Use pretrained CNN features for efficiency
* Build Seq2Seq architecture from scratch
* Evaluate model using NLP metrics
* Deploy a real-time captioning system

---

# 🧠 Concepts Covered

* Image Captioning
* Sequence-to-Sequence Learning
* Encoder-Decoder Architecture
* Transfer Learning (ResNet50)
* LSTM / GRU Networks
* Word Embeddings
* Beam Search & Greedy Search
* NLP Evaluation Metrics

---

# 📂 Dataset Used

## Flickr30k Dataset

[Flickr30k Dataset](https://www.kaggle.com/datasets/adityajn105/flickr30k?utm_source=chatgpt.com)

Contains:

* 30,000 images
* Multiple captions per image
* Real-world image-text pairs

---

# ⚙️ Environment Setup

## Platform

* Kaggle Notebook

## Hardware

* GPU: Tesla T4 ×2

---

# 📦 Libraries Used

```bash id="x9k1ab"
torch
torchvision
numpy
pandas
matplotlib
nltk
tqdm
Pillow
streamlit
```

Install dependencies:

```bash id="gk7x01"
pip install torch torchvision nltk matplotlib pandas pillow tqdm streamlit
```

---

# 🏗️ Model Architecture

# 🔷 1. Feature Extraction (CNN - ResNet50)

A pretrained **ResNet50** model is used to extract image features.

## Output

* 2048-dimensional feature vector per image

## Why ResNet50?

* Strong visual representation
* Pretrained on ImageNet
* Efficient feature extraction

---

# 🔷 2. Encoder

## Architecture

* Single Linear Layer

## Function

Projects image features:

```text id="zv8p2q"
2048 → 512 (hidden size)
```

## Output

* Initial hidden state for decoder

---

# 🔷 3. Decoder (LSTM / GRU)

## Inputs

* Word embeddings of captions

## Process

* Receives previous word
* Uses hidden state from encoder
* Generates next word

## Output

* Probability distribution over vocabulary

---

# 📁 Project Structure

```bash id="m3q8xz"
Neural-Storyteller/
│
├── notebooks/
│   ├── captioning_training.ipynb
│
├── models/
│   ├── encoder.py
│   ├── decoder.py
│   ├── seq2seq.py
│
├── data/
│   ├── captions.txt
│   ├── flickr30k_features.pkl
│
├── vocabulary/
│   ├── vocab.pkl
│
├── outputs/
│   ├── predictions/
│   ├── loss_plots/
│
├── app/
│   ├── streamlit_app.py
│
├── requirements.txt
├── README.md
```

---

# 🧹 Part 1: Feature Extraction Pipeline

## Steps:

1. Load Flickr30k dataset
2. Resize images to 224×224
3. Apply normalization
4. Extract features using ResNet50
5. Save features as:

```text id="k8d9qz"
flickr30k_features.pkl
```

---

# 📝 Part 2: Text Preprocessing

Steps:

* Load captions.txt
* Clean text (lowercase, punctuation removal)
* Tokenize sentences
* Build vocabulary
* Convert words → indices
* Add special tokens:

  * `<start>`
  * `<end>`
  * `<pad>`
  * `<unk>`

---

# 🔄 Part 3: Seq2Seq Architecture

## Encoder Output

* Hidden representation of image

## Decoder Process

1. Take word embedding
2. Combine with hidden state
3. Predict next word
4. Repeat until `<end>` token

---

# 📉 Loss Function

## Cross Entropy Loss

```text id="8r1xqz"
Loss = CrossEntropy(predicted_words, target_words)
```

### Important:

* `ignore_index = padding_token`

---

# ⚙️ Optimizer

| Component     | Value        |
| ------------- | ------------ |
| Optimizer     | Adam         |
| Learning Rate | 0.0002       |
| Loss          | CrossEntropy |

---

# 🔍 Inference Methods

## 1️⃣ Greedy Search

* Select highest probability word at each step
* Fast but less diverse

---

## 2️⃣ Beam Search

* Keeps top-k sequences
* Produces better captions
* More computationally expensive

---

# 📊 Evaluation Metrics

## 1️⃣ BLEU-4 Score

Measures similarity between generated and reference captions.

Higher BLEU = better caption quality.

---

## 2️⃣ Precision / Recall / F1-score

Token-level evaluation of predicted captions.

---

## Optional Metrics:

* METEOR
* ROUGE

---

# 🖼️ Visualization Module

Displays:

* Input Image
* Ground Truth Caption
* Predicted Caption

Example:

| Image | Ground Truth             | Prediction                      |
| ----- | ------------------------ | ------------------------------- |
| 🖼️   | "A dog running in grass" | "A dog is running in the field" |

---

# 📈 Training Logs

Includes:

* Training Loss vs Epochs
* Validation Loss vs Epochs
* Caption quality improvements over time

---

# 📱 App Deployment

The system is deployed using Streamlit.

## Features:

✅ Upload image
✅ Generate caption instantly
✅ Compare predictions
✅ Real-time inference

Run locally:

```bash id="p9x7km"
streamlit run streamlit_app.py
```

---

# 🌐 Live Demo

[Open Neural Storyteller App](https://neural-storyteller-image-captioning-with-seq2seq-afkhezegpeekr.streamlit.app/?utm_source=chatgpt.com)

---

# 🔍 Results & Observations

## Strengths

* Good caption coherence
* Understands objects and scenes
* Works well on common images

## Limitations

* Struggles with rare objects
* Limited context understanding
* Depends heavily on dataset quality

---

# 🎯 Applications

* Image Search Engines
* Accessibility tools for visually impaired users
* Social media auto-captioning
* Content indexing
* AI storytelling systems

---

# 🔮 Future Improvements

* Transformer-based captioning (e.g., ViT + GPT)
* Attention mechanisms
* Larger vocabulary training
* Multi-modal pretraining
* Better beam search optimization

---

# 🎓 Conclusion

This project successfully demonstrates a **Seq2Seq-based image captioning system** using ResNet50 + LSTM/GRU.

It bridges computer vision and natural language processing by generating meaningful captions from images.

---

# 👨‍💻 Author

**Mubashir Siddique**

AI / Deep Learning / Computer Vision Enthusiast

---

# 📜 License

This project is developed for educational and research purposes.
