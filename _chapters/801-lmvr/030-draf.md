---
slug: lmvr-3
title: lmvr-3
---
# Bab 3: Arsitektur Jaringan Saraf untuk NLP

> **"Keberhasilan AI modern tidak hanya bergantung pada algoritma, tetapi juga pada arsitektur yang mendukungnya, memungkinkan kita membangun sistem yang dapat memahami dan menghasilkan bahasa manusia dengan akurasi yang luar biasa." — Yann LeCun**

Bab 3 dari LMVR menggali berbagai arsitektur jaringan saraf yang menjadi tulang punggung pemrosesan bahasa alami (NLP). Bab ini dimulai dengan dasar-dasar jaringan saraf, menjelaskan keterbatasan jaringan feedforward untuk tugas-tugas NLP dan perlunya arsitektur yang lebih canggih seperti Recurrent Neural Networks (RNN) dan Convolutional Neural Networks (CNN). Bab ini kemudian mengeksplorasi mekanisme atensi dan Transformer, menyoroti kemampuan mereka untuk menangani dependensi jarak jauh dan menskalakan secara efektif. Model-model canggih seperti BERT dan GPT dibahas, dengan penekanan pada proses pra-pelatihan dan penyetelan halus (fine-tuning). Bab ini diakhiri dengan model hibrida dan pembelajaran multi-tugas (multi-task learning), menunjukkan bagaimana penggabungan arsitektur yang berbeda dapat meningkatkan performa.

## 3.1. Pengantar Jaringan Saraf untuk NLP

Jaringan saraf telah menjadi alat fundamental untuk memproses dan memahami bahasa alami dalam kecerdasan buatan modern. Pada intinya, jaringan saraf dibangun dari neuron, yang merupakan fungsi matematika yang dirancang untuk mensimulasikan perilaku neuron biologis. Setiap neuron menerima masukan (direpresentasikan sebagai vektor numerik), menerapkan transformasi berbobot, dan melewatkan hasilnya melalui fungsi aktivasi untuk menentukan keluaran. Neuron-neuron ini disusun menjadi lapisan-lapisan—biasanya mencakup lapisan masukan, satu atau lebih lapisan tersembunyi, dan lapisan keluaran—membentuk apa yang dikenal sebagai feedforward neural network (FNN) atau Multi Layer Perceptron (MLP). Meskipun FNN sangat kuat, mereka menghadapi keterbatasan saat diterapkan pada NLP karena bahasa pada dasarnya melibatkan konteks dan urutan, yang tidak dirancang untuk ditangani oleh FNN secara efisien.

![Gambar 1: Alat pembelajaran interaktif untuk arsitektur FNN atau MLP (https://deeperplayground.org).](https://lmvr.rantai.dev//images/L6A504iArDGBvUq8WSeO-oLn6fvAtaJcnox1LeTx1-v1.png)
**Gambar 1:** Alat pembelajaran interaktif untuk arsitektur FNN atau MLP (https://deeperplayground.org).

Dalam FNN, koneksi antar node tidak membentuk siklus. Setiap lapisan terdiri dari neuron, dan setiap neuron terhubung ke setiap neuron di lapisan berikutnya. Selama pelatihan jaringan saraf, proses pembaruan bobot dipandu oleh algoritma optimasi, seperti penurunan gradien (gradient descent), yang bertujuan untuk meminimalkan fungsi kerugian (loss function).

![Gambar 2: Backpropagation dan penurunan gradien.](https://lmvr.rantai.dev//images/L6A504iArDGBvUq8WSeO-xnfZc9XDGxcxrWVR8iKl-v1.png)
**Gambar 2:** Backpropagation dan penurunan gradien.

Algoritma backpropagation bekerja dengan menghitung gradien dari fungsi kerugian terhadap setiap bobot dan bias dalam jaringan dengan merambatkan kesalahan (error) ke belakang melalui jaringan. Momentum adalah ekstensi dari penurunan gradien yang membantu mempercepat konvergensi dan meratakan proses optimasi. Fungsi aktivasi seperti ReLU umum digunakan karena memperkenalkan non-linearitas tanpa menyebabkan gradien menghilang (vanishing gradient).

![Gambar 3: Fungsi aktivasi MLP.](https://lmvr.rantai.dev//images/L6A504iArDGBvUq8WSeO-KCHmcGKFbedy035Cy4vs-v1.png)
**Gambar 3:** Fungsi aktivasi MLP.

Dalam NLP, bahasa bersifat sekuensial dan sangat bergantung pada konteks. Makna sebuah kata sering berubah berdasarkan kata-kata di sekitarnya. FNN tradisional memperlakukan data masukan sebagai data independen dan statis, sehingga sulit bagi mereka untuk menangkap ketergantungan antar kata dalam suatu urutan.

### Contoh Python: Implementasi MLP Sederhana (PyTorch)

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Fungsi untuk menghasilkan dataset sintetis
def generate_synthetic_data(num_samples, input_size, num_classes):
    inputs = torch.randn(num_samples, input_size)
    targets = torch.randint(0, num_classes, (num_samples,))
    return inputs, targets

# Mendefinisikan arsitektur MLP
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
    
    train_input, train_target = generate_synthetic_data(1000, input_size, output_size)
    test_input, test_target = generate_synthetic_data(200, input_size, output_size)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    epochs = 100
    batch_size = 32
    
    for epoch in range(1, epochs + 1):
        model.train()
        permutation = torch.randperm(train_input.size(0))
        
        for i in range(0, train_input.size(0), batch_size):
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = train_input[indices].to(device), train_target[indices].to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                train_loss = criterion(model(train_input.to(device)), train_target.to(device))
                test_loss = criterion(model(test_input.to(device)), test_target.to(device))
                print(f"Epoch: {epoch}, Train Loss: {train_loss.item():.4f}, Test Loss: {test_loss.item():.4f}")

if __name__ == "__main__":
    main()
```

## 3.2. Jaringan Saraf Berulang (RNN)

Recurrent Neural Networks (RNN) telah menjadi landasan arsitektur jaringan saraf untuk pemodelan data sekuensial. Berbeda dengan FNN, RNN memperkenalkan koneksi berulang yang memungkinkan mereka mempertahankan keadaan internal (hidden state) yang berevolusi saat memproses urutan masukan.

![Gambar 4: Sebuah RNN dengan hidden state (Kredit d2l.ai)](https://lmvr.rantai.dev//images/L6A504iArDGBvUq8WSeO-331xGMO52yVwjmoasF4y-v1.svg)
**Gambar 4:** Sebuah RNN dengan hidden state (Kredit d2l.ai)

Secara matematis, RNN memperbarui hidden state $h_t$ di setiap langkah waktu:
$$ h_t = \phi(W_{hh} h_{t-1} + W_{xh} x_t + b_h) $$

Kelemahan utama RNN tradisional adalah masalah gradien menghilang (vanishing gradient) saat menangani urutan yang panjang. Untuk mengatasinya, dikembangkan arsitektur seperti Long Short-Term Memory (LSTM) dan Gated Recurrent Units (GRU).

![Gambar 5: Sel LSTM dengan hidden state (Kredit d2l.ai).](https://lmvr.rantai.dev//images/L6A504iArDGBvUq8WSeO-1IxiLdN5eSAQLDIkNVKQ-v1.svg)
**Gambar 5:** Sel LSTM dengan hidden state (Kredit d2l.ai).

LSTM menggunakan mekanisme gerbang (forget, input, output gate) untuk mengatur aliran informasi dalam sel memori. GRU menyederhanakan ini dengan hanya menggunakan gerbang pembaruan (update) dan pengaturan ulang (reset).

![Gambar 6: Sel GRU dengan hidden state.](https://lmvr.rantai.dev//images/L6A504iArDGBvUq8WSeO-ooh1VyRQIMkfQPZgtfx1-v1.svg)
**Gambar 6:** Sel GRU dengan hidden state.

### Contoh Python: Karakter-level Language Model (LSTM/GRU)

```python
import torch
import torch.nn as nn
import torch.optim as optim

class CharRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, n_layers=1, model_type='lstm'):
        super(CharRNN, self).__init__()
        self.model_type = model_type
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        
        self.encoder = nn.Embedding(vocab_size, hidden_size)
        if model_type == 'lstm':
            self.rnn = nn.LSTM(hidden_size, hidden_size, n_layers, batch_first=True)
        else:
            self.rnn = nn.GRU(hidden_size, hidden_size, n_layers, batch_first=True)
        self.decoder = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden):
        x = self.encoder(x)
        out, hidden = self.rnn(x, hidden)
        out = self.decoder(out.reshape(-1, self.hidden_size))
        return out, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        if self.model_type == 'lstm':
            return (weight.new(self.n_layers, batch_size, self.hidden_size).zero_(),
                    weight.new(self.n_layers, batch_size, self.hidden_size).zero_())
        else:
            return weight.new(self.n_layers, batch_size, self.hidden_size).zero_()

# Implementasi pelatihan dan sampling disingkat untuk kejelasan
```

## 3.3. Convolutional Neural Networks (CNN) untuk NLP

Convolutional Neural Networks (CNN), yang awalnya dikembangkan untuk pemrosesan gambar, telah diadaptasi untuk berbagai tugas NLP. Dalam NLP, filter CNN meluncur di atas data teks untuk mendeteksi pola lokal seperti n-gram atau frasa pendek.

![Gambar 7: Ilustrasi arsitektur CNN untuk tugas NLP (Ref: https://arxiv.org/pdf/1703.03091).](https://lmvr.rantai.dev//images/L6A504iArDGBvUq8WSeO-NArmtrklkgenplPQj944-v1.png)
**Gambar 7:** Ilustrasi arsitektur CNN untuk tugas NLP (Ref: https://arxiv.org/pdf/1703.03091).

Operasi konvolusi 1D didefinisikan sebagai:
$$ Y(i) = \sum_{j} K(j) \cdot X(i + j) $$

CNN sangat efektif untuk klasifikasi teks karena dapat dengan cepat mengidentifikasi frasa kunci yang bersifat indikatif terhadap suatu kelas melalui lapisan pooling (biasanya max pooling).

### Contoh Python: CNN untuk Klasifikasi Teks (PyTorch)

```python
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_filters, filter_sizes, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim, out_channels=n_filters, kernel_size=fs)
            for fs in filter_sizes
        ])
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, text):
        # text = [batch size, sent len]
        embedded = self.embedding(text).permute(0, 2, 1) # [batch size, embed dim, sent len]
        conved = [torch.relu(conv(embedded)) for conv in self.convs]
        pooled = [torch.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = self.dropout(torch.cat(pooled, dim=1))
        return self.fc(cat)
```

## 3.4. Mekanisme Atensi dan Transformer

Mekanisme atensi memungkinkan model untuk fokus pada bagian tertentu dari urutan masukan yang paling relevan. Self-attention, landasan model Transformer, memungkinkan setiap token dalam urutan untuk menghadiri (attend) setiap token lainnya.

![Gambar 8: Memahami mekanisme self-attention (Kredit: Sebastian Raschka)](https://lmvr.rantai.dev//images/L6A504iArDGBvUq8WSeO-FKvJ0bEroAvw0XXnnsdW-v1.png)
**Gambar 8:** Memahami mekanisme self-attention (Kredit: Sebastian Raschka).

Rumus dasar atensi:
$$ \text{Attention}(Q, K, V) = \text{softmax}\left( \frac{QK^T}{\sqrt{d_k}} \right) V $$

Arsitektur Transformer terdiri dari encoder dan decoder. Keuntungan utamanya adalah kemampuan untuk memproses urutan secara paralel, berbeda dengan RNN yang memproses secara sekuensial.

![Gambar 9: Arsitektur Transformer.](https://lmvr.rantai.dev//images/L6A504iArDGBvUq8WSeO-0fgxigI0qh1xNcSS2A8t-v1.svg)
**Gambar 9:** Arsitektur Transformer.

BERT (Bidirectional Encoder Representations from Transformers) adalah model bidirectional yang membaca teks dari dua arah secara bersamaan, sementara GPT adalah model autoregresif yang memprediksi kata berikutnya secara sekuensial.

![Gambar 10: Arsitektur BERT (Bidirectional Encoder Representations from Transformers).](https://lmvr.rantai.dev//images/L6A504iArDGBvUq8WSeO-45st5xbkWJHjOxPa8aJA-v1.png)
**Gambar 10:** Arsitektur BERT (Bidirectional Encoder Representations from Transformers).

![Gambar 11: Alat interaktif Dodrio untuk memahami cara kerja multi-head attention.](https://lmvr.rantai.dev//images/L6A504iArDGBvUq8WSeO-sL9fTWrjmPvOdQzxWVH1-v1.png)
**Gambar 11:** Alat interaktif Dodrio untuk memahami cara kerja multi-head attention.

![Gambar 12: Alat Transformer Explainer untuk memahami cara kerja transformer bagi model bahasa.](https://lmvr.rantai.dev//images/L6A504iArDGBvUq8WSeO-E5CxgiT0iCmwBCgtzk9M-v1.png)
**Gambar 12:** Alat Transformer Explainer untuk memahami cara kerja transformer bagi model bahasa.

### Contoh Python: Implementasi Transformer Sederhana (GPT-like)

```python
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        att = (q @ k.transpose(-2, -1)) * (1.0 / (k.size(-1)**0.5))
        # Masking untuk causal attention
        mask = torch.tril(torch.ones(T, T)).view(1, 1, T, T).to(x.device)
        att = att.masked_fill(mask == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)
```

## 3.5. Arsitektur Canggih: BERT, GPT, dan Seterusnya

Model bahasa pra-terlatih seperti BERT dan GPT telah merevolusi NLP dengan menetapkan paradigma baru dalam cara model dibangun dan diterapkan. Model-model ini dilatih pada korpus teks yang masif dan kemudian disetel halus untuk tugas-tugas tertentu.

![Gambar 13: Ilustrasi arsitektur GPT-2.](https://lmvr.rantai.dev//images/L6A504iArDGBvUq8WSeO-jJj04waaI54uyemzKACD-v1.png)
**Gambar 13:** Ilustrasi arsitektur GPT-2.

BERT menggunakan pendekatan Masked Language Model (MLM), sedangkan GPT menggunakan pendekatan autoregresif. Konsep transfer learning memungkinkan pengetahuan umum yang dipelajari selama pra-pelatihan untuk diadaptasi ke tugas-tugas khusus dengan dataset yang relatif kecil.

Evolusi model ini telah melahirkan arsitektur seperti LLaMA (Large Language Model Meta AI), T5 (Text-To-Text Transfer Transformer), dan GPT-3 yang mendorong batas kemampuan pemahaman bahasa manusia.

### Contoh Python: Inferensi LLaMA (Hugging Face Transformers)

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def generate_text(prompt, model_id="meta-llama/Llama-2-7b-hf"):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=50)
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# print(generate_text("The capital of France is"))
```

## 3.6. Model Hibrida dan Pembelajaran Multi-Tugas

Model hibrida menggabungkan arsitektur berbeda untuk menangkap aspek bahasa yang beragam. Misalnya, CNN dapat digunakan untuk menangkap fitur lokal, sementara Transformer menangani konteks global.

Multi-task learning (MTL) memungkinkan model untuk berbagi representasi di berbagai tugas NLP yang berbeda, yang mengarah pada generalisasi yang lebih baik.

LLaVA (Language and Vision Assistant) adalah contoh model multi-modal yang mengintegrasikan pemahaman bahasa dan gambar secara bersamaan.

### Contoh Python: Preprocessing Gambar LLaVA

```python
from PIL import Image
import torch
from torchvision import transforms

class ImageProcessor:
    def __init__(self, size=224):
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4814, 0.4578, 0.4082], std=[0.2686, 0.2613, 0.2757])
        ])

    def preprocess(self, image_path):
        image = Image.open(image_path).convert("RGB")
        return self.transform(image).unsqueeze(0) # [1, 3, 224, 224]
```

## 3.7. Eksplanabilitas dan Interpretabilitas Model

Seiring bertambahnya kompleksitas jaringan saraf, kebutuhan untuk memahami bagaimana model sampai pada prediksinya menjadi sangat penting. Teknik yang umum digunakan meliputi visualisasi atensi dan analisis kepentingan fitur (feature importance).

Salah satu metode populer adalah Integrated Gradients yang menghitung kontribusi setiap fitur input terhadap prediksi akhir secara matematis.

### Contoh Python: Ekstraksi Bobot Atensi

```python
def get_attention_weights(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs, output_attentions=True)
    
    # Mengambil bobot atensi dari lapisan terakhir, head pertama
    attention = outputs.attentions[-1][0, 0, :, :].detach().cpu().numpy()
    return attention

# Visualisasi dapat dilakukan menggunakan matplotlib atau library heatmap
```

## 3.8. Kesimpulan

Bab 3 memberikan tinjauan komprehensif tentang arsitektur jaringan saraf untuk NLP, dari prinsip dasar hingga model mutakhir seperti Transformer dan BERT. Penguasaan arsitektur ini sangat penting bagi praktisi AI untuk membangun sistem yang mampu menangani kekayaan dan kompleksitas bahasa manusia.

### 3.8.1. Pembelajaran Lebih Lanjut dengan GenAI

- Jelaskan keterbatasan FNN dalam menangani data sekuensial.
- Bandingkan mekanisme internal antara LSTM dan GRU.
- Diskusikan peran konvolusi 1D dalam ekstraksi fitur teks.
- Bedah mekanisme self-attention pada Transformer secara matematis.
- Analisis perbedaan antara model encoder-only (BERT) dan decoder-only (GPT).
- Bagaimana transfer learning mengubah lanskap pengembangan aplikasi AI?
- Apa keuntungan menggunakan model hibrida CNN-Transformer?
- Jelaskan konsep Integrated Gradients untuk interpretasi model.
- Diskusikan tantangan dalam menskalakan model bahasa besar.
- Bagaimana teknik kuantisasi dan distilasi membantu efisiensi model?

### 3.8.2. Latihan Praktis

#### Latihan Mandiri 3.1: Mengimplementasikan dan Membandingkan RNN dan Transformer
**Tujuan:** Memahami perbedaan penanganan data sekuensial antara RNN dan Transformer untuk tugas analisis sentimen.

#### Latihan Mandiri 3.2: Penyetelan Halus Model BERT Pra-Terlatih
**Tujuan:** Mendapatkan pengalaman praktis dalam menyesuaikan model BERT untuk tugas klasifikasi teks spesifik.

#### Latihan Mandiri 3.3: Visualisasi Bobot Atensi dalam Model Transformer
**Tujuan:** Mengeksplorasi interpretabilitas model dengan memvisualisasikan fokus model pada kata-kata tertentu dalam kalimat.

#### Latihan Mandiri 3.4: Mengimplementasikan Model Multi-Task Learning
**Tujuan:** Membangun satu model yang dapat melakukan klasifikasi topik dan analisis sentimen secara bersamaan.

#### Latihan Mandiri 3.5: Optimasi Model Hibrida
**Tujuan:** Menggabungkan kekuatan ekstraksi fitur lokal CNN dengan kemampuan sekuensial RNN atau Transformer.