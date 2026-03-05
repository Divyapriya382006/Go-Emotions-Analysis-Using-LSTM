# Toxic Comment Classification using LSTM

## Project Overview

This project implements a **multi-label toxic comment classification system** using **TensorFlow and Keras**. The model processes textual comments and predicts multiple toxicity labels using a Long Short-Term Memory (LSTM) neural network.

The project demonstrates a complete Natural Language Processing (NLP) pipeline including text preprocessing, tokenization, sequence padding, model training, and evaluation.

---

## Model Architecture

The model is built using a sequential neural network architecture:

* **Embedding Layer** – Converts tokenized words into dense vector representations
* **LSTM Layer** – Captures contextual relationships in text sequences
* **Dense Output Layer** – Produces multi-label predictions using sigmoid activation

Example implementation:

```python
from tensorflow import keras

model = keras.Sequential([
    keras.layers.Embedding(input_dim=10000, output_dim=128, input_length=200),
    keras.layers.LSTM(64),
    keras.layers.Dense(29, activation="sigmoid")
])
```

---

## Data Preprocessing

Text data must be converted into numerical format before training the model.

### Steps

1. Tokenize the input text
2. Convert text into sequences of integers
3. Pad sequences to maintain uniform input length

Example preprocessing pipeline:

```python
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(x_train)

x_train_seq = tokenizer.texts_to_sequences(x_train)
x_train_pad = pad_sequences(x_train_seq, maxlen=200)
```

---

## Model Training

The model is trained using binary cross-entropy loss since this is a **multi-label classification problem**.

```python
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    x_train_pad,
    y_train,
    batch_size=100,
    epochs=5
)
```

---

## Training Results

Example training output:

```
Epoch 1/5
Accuracy: 0.33
Loss: 0.137

Epoch 5/5
Accuracy: 0.37
Loss: 0.125
```

The decreasing loss indicates that the model is learning meaningful patterns from the training data.

---

## Resource Utilization

Training was executed in a cloud notebook environment with the following resource usage:

| Resource         | Usage      |
| ---------------- | ---------- |
| Session Duration | 25 minutes |
| CPU Utilization  | ~383%      |
| RAM Usage        | ~3 GB      |
| Disk Usage       | ~305 MB    |

CPU-based training significantly increases training time for deep learning models.

---

## Recommended Hardware

For efficient model training, it is recommended to run this project on a system with higher computational resources.

### Minimum Requirements

* 8 GB RAM
* Multi-core CPU

### Recommended Configuration

* 16 GB RAM or higher
* Dedicated GPU (NVIDIA CUDA-enabled)

### Ideal Configuration

* NVIDIA GPU (RTX / GTX series)
* 16–32 GB RAM
* CUDA and cuDNN installed

Using a GPU can significantly reduce training time compared to CPU-based execution.

---

## Installation

Install the required dependencies before running the project.

```bash
pip install tensorflow pandas numpy scikit-learn
```

---

## Running the Project

Clone the repository and run the training script or notebook.

```bash
python train_model.py
```

---

## Future Improvements

Potential improvements for this project include:

* Implementing Bidirectional LSTM
* Adding dropout layers to prevent overfitting
* Using pretrained embeddings such as GloVe or Word2Vec
* Experimenting with transformer-based architectures such as BERT

---

## License

This project is intended for academic and educational purposes.
