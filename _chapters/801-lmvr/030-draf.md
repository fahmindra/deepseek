---
slug: lmvr-3
title: lmvr-3
---
# Bab 3: Arsitektur Jaringan Saraf untuk NLP

> 💡 **"Keberhasilan AI modern tidak hanya bergantung pada algoritma, tetapi pada arsitektur yang mendukungnya, yang memungkinkan kita membangun sistem yang dapat memahami dan menghasilkan bahasa manusia dengan akurasi yang luar biasa." — Yann LeCun**

> 📘 **Ringkasan Bab:**
> Bab 3 dari LMVR menggali berbagai arsitektur jaringan saraf yang membentuk tulang punggung pemrosesan bahasa alami (NLP). Bab ini dimulai dengan dasar-dasar jaringan saraf, menjelaskan keterbatasan jaringan feedforward untuk tugas-tugas NLP dan perlunya arsitektur yang lebih canggih seperti Recurrent Neural Networks (RNNs) dan Convolutional Neural Networks (CNNs). Bab ini kemudian mengeksplorasi mekanisme atensi dan Transformers, menyoroti kemampuan mereka untuk menangani ketergantungan jarak jauh dan melakukan penskalaan secara efektif. Model-model canggih seperti BERT dan GPT dibahas, dengan penekanan pada proses pre-training dan fine-tuning-nya, dan bab ini diakhiri dengan model hybrid dan pembelajaran multi-tugas, yang menunjukkan bagaimana penggabungan arsitektur yang berbeda dapat meningkatkan performa. Wawasan praktis di seluruh bab ini memandu pembaca dalam mengimplementasikan model-model ini, memastikan mereka dapat menerapkan teknik-teknik ini dalam tugas-tugas NLP di dunia nyata.

---

## 3.1. Pengantar Jaringan Saraf untuk NLP

Jaringan saraf telah menjadi alat fundamental untuk memproses dan memahami bahasa alami dalam kecerdasan buatan modern. Pada intinya, jaringan saraf dibangun dari neuron (juga dikenal sebagai unit), yang merupakan fungsi matematika yang dirancang untuk mensimulasikan perilaku neuron biologis. Setiap neuron menerima masukan (direpresentasikan sebagai vektor numerik), menerapkan transformasi berbobot, dan melewatkan hasilnya melalui fungsi aktivasi untuk menentukan keluaran. Neuron-neuron ini disusun menjadi lapisan-lapisan—biasanya mencakup lapisan masukan, satu atau lebih lapisan tersembunyi, dan lapisan keluaran—membentuk apa yang dikenal sebagai feedforward neural network (FNN) atau Multi Layer Perceptron (MLP). Meskipun FNN sangat kuat, mereka menghadapi keterbatasan saat diterapkan pada pemrosesan bahasa alami (NLP), terutama karena bahasa secara inheren melibatkan konteks dan urutan, yang tidak dirancang untuk ditangani oleh FNN secara efisien.

![Figure 1: Alat Pembelajaran Interaktif untuk arsitektur FNN atau MLP (https://deeperplayground.org).](https://lmvr.rantai.dev/images/L6A504iArDGBvUq8WSeO-oLn6fvAtaJcnox1LeTx1-v1.png)

FNN atau MLP adalah jenis jaringan saraf tiruan di mana koneksi antar node tidak membentuk siklus. Setiap lapisan terdiri dari neuron, dan setiap neuron terhubung ke setiap neuron di lapisan berikutnya. Dalam lingkungan Deeperplayground, kita dapat bereksperimen dengan data 2D untuk melihat bagaimana MLP mempelajari batas keputusan.

Arsitektur MLP dalam skenario ini dapat direpresentasikan secara matematis pada setiap lapisan $l$ sebagai:

$$ z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)} $$
$$ a^{(l)} = \sigma(z^{(l)}) $$

Di sini, $W^{(l)}$ mewakili matriks bobot, $b^{(l)}$ adalah vektor bias, $a^{(l-1)}$ adalah aktivasi dari lapisan sebelumnya, dan $\sigma$ adalah fungsi aktivasi (seperti ReLU, sigmoid, atau tanh). Lapisan keluaran biasanya menggunakan fungsi softmax untuk klasifikasi:

$$ \hat{y}_i = \frac{e^{z_i}}{\sum_j e^{z_j}} $$

MLP belajar dengan meminimalkan kerugian cross-entropy:

$$ L(y, \hat{y}) = - \sum_i y_i \log(\hat{y}_i) $$

Untuk meminimalkan kerugian, algoritma gradient descent menghitung gradien fungsi kerugian terhadap setiap bobot dan bias menggunakan algoritma backpropagation. Aturan pembaruan bobot didefinisikan sebagai:

$$ W_{ij}^{(l)} \leftarrow W_{ij}^{(l)} - \eta \frac{\partial L}{\partial W_{ij}^{(l)}} $$

Momentum adalah ekstensi dari gradient descent yang membantu mempercepat konvergensi dengan menambahkan sebagian dari pembaruan sebelumnya ke pembaruan saat ini:

$$ v_{ij}^{(l)} \leftarrow \gamma v_{ij}^{(l)} + \eta \frac{\partial L}{\partial W_{ij}^{(l)}} $$
$$ W_{ij}^{(l)} \leftarrow W_{ij}^{(l)} - v_{ij}^{(l)} $$

![Figure 2: Backpropagation dan gradient descent.](https://lmvr.rantai.dev/images/L6A504iArDGBvUq8WSeO-xnfZc9XDGxcxrWVR8iKl-v1.png)

Fungsi aktivasi seperti ReLU sangat penting untuk menangkap non-linearitas. Dropout diterapkan untuk mencegah overfitting dengan mematikan neuron secara acak selama pelatihan. Regularisasi $L_2$ juga digunakan untuk memberikan penalti pada bobot yang terlalu besar.

![Figure 3: Fungsi aktivasi MLP.](https://lmvr.rantai.dev/images/L6A504iArDGBvUq8WSeO-KCHmcGKFbedy035Cy4vs-v1.png)

Dalam NLP, bahasa bersifat sekuensial. Kata "anjing" dalam "Anjing mengejar kucing" memiliki peran yang berbeda jika urutannya diubah. FNN tradisional kesulitan menangkap ketergantungan urutan ini karena mereka memperlakukan masukan secara independen.

Berikut adalah implementasi MLP sederhana dalam Python menggunakan PyTorch:

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Fungsi untuk menghasilkan data sintetis
def generate_synthetic_data(num_samples, input_size, num_classes):
    inputs = torch.randn(num_samples, input_size)
    targets = torch.randint(0, num_classes, (num_samples,))
    return inputs, targets

# Definisi arsitektur MLP
class MLP(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size):
        super(MLP, self).__init__()
        layers = []
        in_dim = input_size
        
        for h_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            in_dim = h_dim
            
        layers.append(nn.Linear(in_dim, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    input_size = 100
    hidden_layers = [128, 64, 32]
    output_size = 5
    
    model = MLP(input_size, hidden_layers, output_size).to(device)
    
    # Data sintetis
    train_input, train_target = generate_synthetic_data(1000, input_size, output_size)
    test_input, test_target = generate_synthetic_data(200, input_size, output_size)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    epochs = 100
    batch_size = 32
    
    for epoch in range(1, epochs + 1):
        model.train()
        permutation = torch.randperm(train_input.size(0))
        
        epoch_loss = 0
        for i in range(0, train_input.size(0), batch_size):
            indices = permutation[i:i + batch_size]
            batch_x, batch_y = train_input[indices].to(device), train_target[indices].to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                test_outputs = model(test_input.to(device))
                test_loss = criterion(test_outputs, test_target.to(device))
                print(f"Epoch: {epoch}, Train Loss: {epoch_loss/(1000/32):.4f}, Test Loss: {test_loss.item():.4f}")

if __name__ == "__main__":
    main()
```

---

## 3.2. Recurrent Neural Networks (RNNs)

Recurrent Neural Networks (RNNs) telah menjadi landasan untuk memodelkan data sekuensial. Berbeda dengan FNN, RNN memperkenalkan koneksi berulang yang memungkinkan mereka mempertahankan status internal (*internal state*).

![Figure 4: RNN dengan hidden state (Kredit d2l.ai)](https://lmvr.rantai.dev/images/L6A504iArDGBvUq8WSeO-331xGMO52yVwjmoasF4y-v1.svg)

Secara matematis, RNN memperbarui *hidden state* $h_t$ pada setiap langkah waktu:

$$ h_t = \phi(W_{hh} h_{t-1} + W_{xh} x_t + b_h) $$

Namun, RNN tradisional menderita masalah *vanishing* dan *exploding gradient* selama pelatihan melalui Backpropagation Through Time (BPTT). Untuk mengatasinya, dikembangkan arsitektur seperti Long Short-Term Memory (LSTM) dan Gated Recurrent Units (GRU).

![Figure 5: Sel LSTM dengan hidden state (Kredit d2l.ai).](https://lmvr.rantai.dev/images/L6A504iArDGBvUq8WSeO-1IxiLdN5eSAQLDIkNVKQ-v1.svg)

LSTM menggunakan gerbang (*gates*): *forget gate* ($F_t$), *input gate* ($I_t$), dan *output gate* ($O_t$) untuk mengatur aliran informasi dalam *cell state* ($C_t$).

![Figure 6: Sel GRU dengan hidden state.](https://lmvr.rantai.dev/images/L6A504iArDGBvUq8WSeO-ooh1VyRQIMkfQPZgtfx1-v1.svg)

Berikut adalah implementasi model bahasa tingkat karakter menggunakan LSTM/GRU dalam Python:

```python
import torch
import torch.nn as nn
import requests
import os

# Download data Tiny Shakespeare
URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
if not os.path.exists("input.txt"):
    data = requests.get(URL).text
    with open("input.txt", "w") as f:
        f.write(data)

class CharRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, n_layers=1, model_type="lstm"):
        super(CharRNN, self).__init__()
        self.model_type = model_type
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        
        self.encoder = nn.Embedding(vocab_size, hidden_size)
        if model_type == "lstm":
            self.rnn = nn.LSTM(hidden_size, hidden_size, n_layers, batch_first=True)
        else:
            self.rnn = nn.GRU(hidden_size, hidden_size, n_layers, batch_first=True)
        self.decoder = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden):
        x = self.encoder(x)
        out, hidden = self.rnn(x, hidden)
        out = self.decoder(out)
        return out, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        if self.model_type == "lstm":
            return (weight.new_zeros(self.n_layers, batch_size, self.hidden_size),
                    weight.new_zeros(self.n_layers, batch_size, self.hidden_size))
        else:
            return weight.new_zeros(self.n_layers, batch_size, self.hidden_size)

# (Proses pelatihan disederhanakan untuk contoh)
```

---

## 3.3. Convolutional Neural Networks (CNNs) untuk NLP

CNN, yang awalnya untuk pengolahan gambar, telah diadaptasi untuk NLP guna menangkap pola lokal seperti n-gram melalui konvolusi 1D.

![Figure 7: Ilustrasi arsitektur CNN untuk tugas NLP (Ref: https://arxiv.org/pdf/1703.03091).](https://lmvr.rantai.dev/images/L6A504iArDGBvUq8WSeO-NArmtrklkgenplPQj944-v1.png)

Secara matematis, operasi konvolusi 1D dinyatakan sebagai:
$$ Y(i) = \sum_{j} K(j) \cdot X(i + j) $$

Berikut adalah implementasi ResNet sederhana untuk klasifikasi gambar (sebagai dasar pemahaman CNN) dalam Python:

```python
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

def main():
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    # Menggunakan model ResNet18 standar dari torchvision
    model = torchvision.models.resnet18(num_classes=10)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

    for epoch in range(1, 11):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f"Epoch {epoch}, Loss: {running_loss/len(trainloader):.4f}")

if __name__ == "__main__":
    main()
```

---

## 3.4. Mekanisme Atensi dan Transformers

Mekanisme atensi memungkinkan model untuk memberikan bobot penting yang berbeda pada setiap token dalam urutan. *Self-attention* menghitung skor relevansi menggunakan vektor *Query* ($Q$), *Key* ($K$), dan *Value* ($V$):

$$ \text{Attention}(Q, K, V) = \text{softmax}\left( \frac{QK^T}{\sqrt{d_k}} \right) V $$

![Figure 8: Memahami mekanisme self-attention (Kredit: Sebastian Raschka)](https://lmvr.rantai.dev/images/L6A504iArDGBvUq8WSeO-FKvJ0bEroAvw0XXnnsdW-v1.png)

Transformer menggunakan arsitektur *encoder-decoder* (atau hanya salah satunya) dan dapat diparalelkan secara efisien, tidak seperti RNN.

![Figure 9: Arsitektur Transformer.](https://lmvr.rantai.dev/images/L6A504iArDGBvUq8WSeO-0fgxigI0qh1xNcSS2A8t-v1.svg)

BERT (*Bidirectional Encoder Representations from Transformers*) adalah model *encoder-only* yang bersifat dua arah, sementara GPT (*Generative Pre-trained Transformer*) adalah model *decoder-only* untuk tugas generatif.

![Figure 10: Arsitektur BERT.](https://lmvr.rantai.dev/images/L6A504iArDGBvUq8WSeO-45st5xbkWJHjOxPa8aJA-v1.png)

Berikut adalah contoh mini-GPT sederhana (causal self-attention) dalam Python:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        self.n_head = n_head
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size))
                                     .view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.size()
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / (k.size(-1)**0.5))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)
```

---

## 3.5. Arsitektur Lanjutan: BERT, GPT, dan Lainnya

Model seperti BERT menggunakan *Masked Language Modeling* (MLM):
$$P(x_i | X_{-i})$$
Sedangkan GPT bersifat autoregresif:
$$ P(x_i | x_1, \dots, x_{i-1}) $$

LLaMA (Meta) dan T5 (Google) adalah evolusi lebih lanjut yang menekankan efisiensi skala dan format *text-to-text*.

Berikut adalah cara menjalankan inferensi LLaMA menggunakan Python dan library `transformers`:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def main():
    model_id = "meta-llama/Llama-2-7b-hf" # Memerlukan akses di HuggingFace
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")

    prompt = "My favorite theorem is "
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    output = model.generate(**inputs, max_new_tokens=50)
    print(tokenizer.decode(output[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()
```

---

## 3.6. Model Hybrid dan Multi-Task Learning

Model hybrid menggabungkan kekuatan berbagai arsitektur, misalnya CNN untuk ekstraksi fitur lokal dan Transformer untuk hubungan global. LLaVA adalah contoh model multi-modal yang menggabungkan visi (CLIP) dan bahasa (LLaMA).

Berikut adalah contoh prapemrosesan gambar untuk model multi-modal (LLaVA) di Python:

```python
from PIL import Image
from torchvision import transforms

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.481, 0.457, 0.408], std=[0.268, 0.261, 0.275]),
    ])
    return preprocess(image).unsqueeze(0) # Tambahkan dimensi batch
```

---

## 3.7. Eksplanabilitas dan Interpretabilitas Model

Karena model seperti Transformer sering dianggap sebagai "kotak hitam", teknik visualisasi atensi digunakan untuk melihat kata mana yang dianggap penting oleh model.

```python
# Contoh ekstraksi bobot atensi (pseudocode)
def get_attention_example(model, input_ids):
    outputs = model(input_ids, output_attentions=True)
    attentions = outputs.attentions # Daftar tensor atensi per lapisan
    return attentions[0] # Ambil lapisan pertama
```

---

## 3.8. Kesimpulan

Bab 3 memberikan gambaran komprehensif tentang arsitektur jaringan saraf untuk NLP, dari prinsip dasar hingga model mutakhir seperti Transformers dan BERT.

### 3.8.1. Pembelajaran Lebih Lanjut dengan GenAI

- Jelaskan keterbatasan FNN dalam menangani data sekuensial.
- Bandingkan LSTM dan GRU dalam hal mekanisme internal.
- Bagaimana CNN 1D mengekstrak pola lokal dalam teks?
- Jelaskan konsep *self-attention* dan cara kerjanya secara matematis.
- Diskusikan pentingnya *pre-training* pada BERT dan GPT.

### 3.8.2. Praktik Mandiri

- **Latihan 3.1**: Implementasikan RNN dan Transformer sederhana, lalu bandingkan akurasinya pada tugas klasifikasi teks.
- **Latihan 3.2**: Lakukan *fine-tuning* pada model BERT yang sudah ada untuk tugas analisis sentimen.
- **Latihan 3.3**: Visualisasikan matriks atensi dari model Transformer untuk kalimat yang berbeda.
- **Latihan 3.4**: Bangun model *Multi-Task Learning* yang bisa melakukan klasifikasi topik dan analisis sentimen sekaligus.
- **Latihan 3.5**: Buat model hybrid CNN-LSTM untuk mendeteksi berita palsu.