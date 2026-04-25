---
slug: lmvr-4
title: lmvr-4
---
# Bab 4: Arsitektur Transformer

> **"Model Transformer mewakili pergeseran paradigma dalam pembelajaran mesin, membuktikan bahwa mekanisme atensi, jika diimplementasikan dengan benar, dapat secara signifikan mengungguli metode tradisional dalam menangani data sekuensial." — Geoffrey Hinton**

Bab ini memberikan eksplorasi komprehensif tentang komponen-komponen utama yang menjadikan model Transformer sebagai fondasi model bahasa besar modern. Bab ini dimulai dengan pengenalan arsitektur Transformer, menjelaskan bagaimana model ini merevolusi pemrosesan bahasa alami melalui kemampuan paralelisasi. Bab ini menggali mekanisme self-attention, yang memungkinkan model untuk menimbang pentingnya kata-kata yang berbeda dalam sebuah kalimat, diikuti oleh diskusi rinci tentang multi-head attention, di mana beberapa head atensi menangkap berbagai hubungan kontekstual. Bab ini selanjutnya menjelaskan positional encoding, yang krusial untuk merepresentasikan urutan kata, dan merinci arsitektur encoder-decoder yang menggerakkan tugas-tugas kompleks seperti terjemahan. Normalisasi lapisan (layer normalization) dan koneksi residual, yang penting untuk pelatihan yang stabil dan efisien, dibahas secara menyeluruh, bersama dengan teknik pelatihan dan optimasi yang meningkatkan kinerja. Akhirnya, bab ini diakhiri dengan aplikasi dunia nyata dari model Transformer, menyoroti dampaknya pada berbagai tugas NLP seperti terjemahan mesin, pembuatan teks, dan peringkasan.

## 4.1. Pengantar Model Transformer

Pengenalan model Transformer merevolusi Pemrosesan Bahasa Alami (NLP) dengan mengatasi keterbatasan utama arsitektur tradisional seperti Recurrent Neural Networks (RNN) dan Convolutional Neural Networks (CNN). Baik RNN maupun CNN, meskipun efektif dalam banyak tugas berbasis urutan, kesulitan menangani ketergantungan jarak jauh (long-range dependencies) dalam teks. Dalam RNN, data sekuensial harus diproses satu langkah waktu pada satu waktu, sehingga menantang untuk menangkap hubungan antara token yang berjauhan, terutama dalam urutan yang panjang. Sifat sekuensial ini juga membatasi paralelisasi selama pelatihan dan inferensi, yang secara signifikan mengurangi efisiensi. CNN, meskipun lebih baik dalam menangani komputasi paralel, dibatasi oleh ukuran filter konvolusional, membuatnya tidak cocok untuk memodelkan ketergantungan yang mencakup bagian besar dari sebuah teks. Keterbatasan ini menciptakan kebutuhan akan model yang dapat memproses ketergantungan jarak jauh secara efisien dan menskalakan dengan perangkat keras modern. Kebutuhan inilah yang menyebabkan pengembangan model Transformer.

Arsitektur Transformer, yang diperkenalkan dalam makalah *Attention is All You Need*, menjawab tantangan ini dengan mengganti rekurensi dan konvolusi dengan mekanisme yang disebut self-attention. Inovasi kunci ini memungkinkan Transformer untuk menangkap hubungan antar token tanpa memandang jarak mereka dalam urutan, semuanya sambil memproses urutan tersebut secara paralel. Arsitektur ini terdiri dari dua komponen utama: Encoder dan Decoder, keduanya dibangun menggunakan lapisan self-attention dan jaringan saraf feedforward. Encoder memproses urutan masukan dan menghasilkan sekumpulan representasi sadar-konteks, sementara decoder menghasilkan urutan keluaran dengan memperhatikan masukan yang dikodekan dan keluaran yang dihasilkan sebelumnya. Struktur modular ini sangat fleksibel dan dapat diadaptasi ke berbagai tugas NLP, seperti terjemahan, pembuatan teks, dan peringkasan.

Secara matematis, mekanisme self-attention membentuk inti dari model Transformer. Diberikan urutan masukan $X = [x_1, x_2, \dots, x_n]$, model pertama-tama mengubah setiap token menjadi tiga vektor: Query $Q$, Key $K$, dan Value $V$. Mekanisme atensi menghitung jumlah tertimbang dari vektor value, di mana bobotnya ditentukan oleh dot product dari vektor query dan key. Atensi dot-product berskala didefinisikan sebagai:

$$ \text{Attention}(Q, K, V) = \text{softmax}\left( \frac{QK^T}{\sqrt{d_k}} \right) V $$

Di sini, $d_k$ adalah dimensionalitas vektor key, dan fungsi softmax memastikan bahwa bobot atensi berjumlah 1, membentuk distribusi probabilitas. Hal ini memungkinkan model untuk lebih fokus pada bagian masukan yang relevan dengan token saat ini, menangkap ketergantungan jarak pendek dan jarak jauh dalam satu langkah. Dengan menghitung atensi untuk semua token secara paralel, Transformer dapat memproses seluruh urutan secara efisien, menjadikannya sangat terukur dan cocok untuk dataset besar.

Salah satu keuntungan utama model Transformer adalah peralihannya dari pemrosesan sekuensial ke paralel. RNN tradisional memproses satu token pada satu waktu, sehingga sulit untuk diparalelkan di beberapa unit pemrosesan. Sebaliknya, Transformer memproses semua token dalam urutan secara bersamaan, memungkinkan percepatan besar pada perangkat keras modern seperti GPU dan TPU. Paralelisasi ini sangat penting ketika bekerja dengan korpus teks yang besar atau urutan yang panjang, karena secara signifikan mengurangi waktu pelatihan sambil mempertahankan kinerja tinggi. Kemampuan untuk menangani urutan panjang secara efisien telah menjadikan model Transformer sebagai fondasi bagi banyak model bahasa mutakhir, seperti BERT dan GPT.

![Gambar 1: Arsitektur GPT vs BERT.](https://lmvr.rantai.dev//images/L6A504iArDGBvUq8WSeO-l27fwg4yZdgjT1n0eESy-v1.png)
**Gambar 1:** Arsitektur GPT vs BERT.

Mekanisme atensi dalam Transformer memainkan peran sentral dalam meningkatkan efisiensi dan efektivitas model. Berbeda dengan RNN, yang kesulitan menangkap ketergantungan jarak jauh karena masalah gradien menghilang, atensi memungkinkan Transformer untuk memperhatikan secara langsung bagian mana pun dari masukan, terlepas dari jaraknya. Hal ini membuat model lebih tangguh dalam tugas-tugas yang memerlukan pemahaman hubungan jarak jauh yang kompleks antar kata. Misalnya, dalam terjemahan mesin, Transformer dapat langsung memetakan kata-kata dalam kalimat sumber ke kata-kata yang sesuai dalam kalimat target, bahkan ketika kata-kata ini berjauhan.

Selain mengimplementasikan arsitektur Transformer, membandingkan kinerjanya pada dataset NLP kecil dapat memberikan wawasan tentang bagaimana model ini dibandingkan dengan RNN tradisional. Observasi utama adalah bahwa pemrosesan paralel Transformer memungkinkannya menangani urutan yang lebih panjang dan dataset yang lebih besar secara lebih efisien. Dalam tugas-tugas seperti pemodelan bahasa atau klasifikasi teks, Transformer umumnya mengungguli RNN baik dalam akurasi maupun kecepatan pelatihan karena kemampuannya untuk menangkap ketergantungan jarak jauh dan memproses urutan secara paralel.

Kode berikut mendemonstrasikan cara menggunakan model BERT pra-terlatih untuk analisis sentimen dengan Python.

### Contoh Python: Analisis Sentimen dengan BERT

```python
from transformers import pipeline

def main():
    # Inisialisasi pipeline analisis sentimen dengan model BERT default
    classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

    # Contoh teks untuk dianalisis
    sample_texts = [
        "I loved this movie!",
        "This was a terrible experience.",
    ]

    # Melakukan prediksi
    results = classifier(sample_texts)

    for text, result in zip(sample_texts, results):
        print(f"Teks: {text}")
        print(f"Label: {result['label']}, Skor: {result['score']:.4f}\n")

    # Simulasi pada dataset kecil (misal: 5 teks pertama)
    texts_dataset = [
        "The plot was very engaging.",
        "Acting was sub-par and boring.",
        "A masterpiece of modern cinema.",
        "I wouldn't recommend this to anyone.",
        "Highly original and exciting."
    ]
    
    dataset_results = classifier(texts_dataset)
    for i, res in enumerate(dataset_results):
        print(f"Dataset {i+1}: {res['label']} ({res['score']:.4f})")

if __name__ == "__main__":
    main()
```

Sekarang mari kita pelajari tugas NLP lain seperti pembuatan teks. Kode Python berikut mendemonstrasikan cara menggunakan model GPT-2 untuk pembuatan teks.

### Contoh Python: Pembuatan Teks dengan GPT-2

```python
from transformers import pipeline, set_seed

def main():
    # Inisialisasi generator teks dengan GPT-2
    generator = pipeline('text-generation', model='gpt2')
    set_seed(42) # Agar hasil reproduksibel

    prompts = [
        "The future of AI is",
        "In 2024, the world will",
        "Technology advancements will lead to",
    ]

    print("Pembuatan Teks Berbasis GPT-2:")
    for prompt in prompts:
        # Menghasilkan teks dengan panjang maksimum 50 token
        output = generator(prompt, max_length=50, num_return_sequences=1)
        print(f"Input: {prompt}")
        print(f"Teks Dihasilkan: {output[0]['generated_text']}\n")

if __name__ == "__main__":
    main()
```

BART (Bidirectional and Auto-Regressive Transformers) adalah model berbasis transformer yang dirancang untuk berbagai tugas NLP, termasuk peringkasan teks. BART menggabungkan kekuatan BERT dan GPT-2 dengan memanfaatkan komponen bidirectional dan autoregresif.

### Contoh Python: Peringkasan Teks dengan BART vs GPT-2

```python
from transformers import pipeline

def main():
    # Inisialisasi model peringkasan BART
    summarizer_bart = pipeline("summarization", model="facebook/bart-large-cnn")
    
    # GPT-2 tidak dirancang khusus untuk ringkasan terstruktur, 
    # tapi kita bisa menggunakannya untuk 'melanjutkan' teks sebagai ringkasan
    generator_gpt2 = pipeline("text-generation", model="gpt2")

    dataset = [
        "The quick brown fox jumps over the lazy dog. This is a sentence that represents a very common phrase used in typing tests. The fox is quick and brown, and the dog is lazy and tired.",
        "Artificial intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think and act like humans. AI is being used in various fields such as healthcare, finance, and technology."
    ]

    print("Peringkasan Berbasis BERT (BART):")
    for text in dataset:
        summary = summarizer_bart(text, max_length=20, min_length=5, do_sample=False)
        print(f"Input: {text[:50]}...")
        print(f"Ringkasan: {summary[0]['summary_text']}\n")

    print("Pembuatan 'Ringkasan' Berbasis GPT-2:")
    for text in dataset:
        # Menambahkan instruksi sederhana di akhir prompt
        prompt = text + "\n\nIn summary:"
        gen = generator_gpt2(prompt, max_new_tokens=20, num_return_sequences=1)
        print(f"GPT-2 Output: {gen[0]['generated_text'][len(prompt):].strip()}\n")

if __name__ == "__main__":
    main()
```

Penelitian terbaru tentang Transformer berfokus pada peningkatan skalabilitas dan efisiensi model. Varian seperti Sparse Transformers dan Longformers bertujuan untuk mengurangi kompleksitas komputasi self-attention, sehingga lebih layak untuk menerapkan Transformer pada urutan yang lebih panjang, seperti seluruh dokumen atau buku.

## 4.2. Mekanisme Self-Attention

Mekanisme self-attention terletak pada jantung arsitektur Transformer, yang memungkinkan model untuk memproses urutan secara efisien sambil menangkap hubungan yang kompleks antara kata-kata dalam sebuah kalimat. Model tradisional seperti RNN memproses urutan secara sekuensial, sehingga sulit untuk menangkap ketergantungan jarak jauh. Self-attention, di sisi lain, memungkinkan model untuk fokus pada berbagai bagian dari urutan masukan secara bersamaan.

![Gambar 2: Ilustrasi mekanisme self-attention encoder (Kredit Ketan Doshi).](https://lmvr.rantai.dev//images/L6A504iArDGBvUq8WSeO-VYj8ePTNWmhtqRGsGNXm-v1.png)
**Gambar 2:** Ilustrasi mekanisme self-attention encoder (Kredit Ketan Doshi).

Secara matematis, mekanisme self-attention bekerja dengan membuat vektor query ($Q$), key ($K$), dan value ($V$) untuk setiap token dalam urutan masukan. Skor atensi antara token $i$ dan token $j$ dihitung menggunakan dot product:

$$ \text{Score}(x_i, x_j) = Q_i \cdot K_j $$

Skor ini diskalakan oleh $\sqrt{d_k}$ dan dilewatkan melalui fungsi softmax untuk menghasilkan bobot atensi. Bobot ini kemudian digunakan untuk menghitung jumlah tertimbang dari vektor value:

$$ \text{Attention}(Q, K, V) = \text{softmax}\left( \frac{QK^T}{\sqrt{d_k}} \right) V $$

Salah satu keuntungan utama self-attention adalah kemampuannya untuk menangani hubungan antara semua token dalam urutan masukan secara bersamaan. Peningkatan besar dalam Transformer adalah penggunaan multi-head attention.

![Gambar 3: Multi Head Attention (Kredit Ketan Doshi).](https://lmvr.rantai.dev//images/L6A504iArDGBvUq8WSeO-NpCGTy0HgoOugVLNGZkW-v1.png)
**Gambar 3:** Multi Head Attention (Kredit Ketan Doshi).

### Contoh Python: Implementasi Dasar Scaled Dot-Product Attention

```python
import torch
import torch.nn.functional as F

def scaled_dot_product_attention(query, key, value):
    d_k = query.size(-1)
    # Menghitung skor: (batch, seq, seq)
    scores = torch.matmul(query, key.transpose(-2, -1)) / (d_k ** 0.5)
    # Softmax untuk mendapatkan bobot atensi
    attention_weights = F.softmax(scores, dim=-1)
    # Output adalah jumlah tertimbang dari value
    return torch.matmul(attention_weights, value), attention_weights

def main():
    # Contoh data: Batch=1, Seq_len=4, Embed_dim=8
    query = torch.randn(1, 4, 8)
    key = torch.randn(1, 4, 8)
    value = torch.randn(1, 4, 8)

    output, weights = scaled_dot_product_attention(query, key, value)
    print("Output Atensi:\n", output)
    print("Bobot Atensi:\n", weights)

if __name__ == "__main__":
    main()
```

Visualisasi skor atensi adalah aspek penting lainnya untuk memahami cara kerja self-attention. Berikut adalah kode Python untuk memvisualisasikan heatmap atensi menggunakan `matplotlib` dan `seaborn`.

### Contoh Python: Visualisasi Heatmap Atensi

```python
import matplotlib.pyplot as plt
import seaborn as sns
import torch

def plot_attention(weights, tokens, title="Attention Heatmap"):
    plt.figure(figsize=(10, 8))
    sns.heatmap(weights[0].detach().numpy(), xticklabels=tokens, yticklabels=tokens, annot=True, cmap='viridis')
    plt.title(title)
    plt.show()

def main():
    tokens = ["The", "cat", "sat", "on", "the", "mat"]
    seq_len = len(tokens)
    # Simulasi bobot atensi (random softmax)
    weights = torch.randn(1, seq_len, seq_len)
    weights = torch.softmax(weights, dim=-1)
    
    plot_attention(weights, tokens)

if __name__ == "__main__":
    main()
```

Untuk mengoptimalkan self-attention pada urutan besar, teknik seperti sparse attention sedang dieksplorasi. Sparse attention mengurangi kompleksitas kuadratik dengan membiarkan token hanya memperhatikan subset dari urutan masukan.

![Gambar 4: Ilustrasi mekanisme sparse attention.](https://lmvr.rantai.dev//images/L6A504iArDGBvUq8WSeO-DRuhbtDvveXg8unkLTtM-v1.png)
**Gambar 4:** Ilustrasi mekanisme sparse attention.

![Gambar 5: Ilustrasi arsitektur atensi Linformer.](https://lmvr.rantai.dev//images/L6A504iArDGBvUq8WSeO-IbQUH3xOgXE3mkUmm3c2-v1.png)
**Gambar 5:** Ilustrasi arsitektur atensi Linformer.

## 4.3. Multi-Head Attention

Multi-head attention adalah salah satu komponen paling krusial dari model Transformer. Dalam atensi single-head, model hanya dapat fokus pada satu aspek terbatas dari urutan masukan. Namun, multi-head attention memperluas kemampuan ini dengan membiarkan model secara bersamaan memperhatikan bagian yang berbeda dari urutan melalui beberapa "head" atensi, masing-masing menangkap hubungan dan pola yang berbeda dalam data.

Secara matematis, multi-head attention melibatkan penghitungan beberapa set vektor query, key, dan value. Output dari head atensi digabungkan (concatenated) dan diproyeksikan kembali ke dimensi asli:

$$ \text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O $$

di mana:
$$ \text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V) $$

### Contoh Python: Implementasi Multi-Head Attention (PyTorch)

```python
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert embed_dim % num_heads == 0, "embed_dim harus habis dibagi num_heads"
        
        self.wq = nn.Linear(embed_dim, embed_dim)
        self.wk = nn.Linear(embed_dim, embed_dim)
        self.wv = nn.Linear(embed_dim, embed_dim)
        self.wo = nn.Linear(embed_dim, embed_dim)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        # Linear projections
        Q = self.wq(q).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.wk(k).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.wv(v).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
            
        weights = torch.softmax(scores, dim=-1)
        attention = torch.matmul(weights, V)
        
        # Concat heads
        attention = attention.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        return self.wo(attention)

def main():
    mha = MultiHeadAttention(embed_dim=64, num_heads=8)
    x = torch.randn(1, 10, 64) # (batch, seq, dim)
    output = mha(x, x, x)
    print("Output Multi-Head Attention shape:", output.shape)

if __name__ == "__main__":
    main()
```

![Gambar 6: Faktorisasi peringkat rendah kompak untuk multi-head attention (Ref: https://arxiv.org/pdf/1912.00835v2).](https://lmvr.rantai.dev//images/L6A504iArDGBvUq8WSeO-sdQQWP5crbHJ0wXvVlQ5-v1.jpeg)
**Gambar 6:** Faktorisasi peringkat rendah kompak untuk multi-head attention (Ref: https://arxiv.org/pdf/1912.00835v2).

## 4.4. Positional Encoding

Transformer memproses urutan secara paralel, berbeda dengan RNN yang memproses token demi token. Paralelisme ini meningkatkan efisiensi tetapi menghilangkan kemampuan bawaan untuk memahami posisi relatif kata-kata. Untuk mengatasi ini, Transformer menyertakan positional encoding ke dalam embedding masukan.

Formulasi matematika untuk positional encoding yang digunakan adalah fungsi sinusoidal:

$$ PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right) $$
$$ PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right) $$

Di sini, $pos$ mewakili posisi token dan $i$ adalah dimensi dalam embedding. Sifat sinusoidal memastikan bahwa posisi yang berdekatan memiliki pengkodean yang serupa, yang membantu model mempelajari ketergantungan lokal.

### Contoh Python: Menghasilkan Positional Encoding

```python
import numpy as np
import torch

def get_positional_encoding(seq_len, d_model):
    pe = np.zeros((seq_len, d_model))
    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            pe[pos, i] = np.sin(pos / (10000 ** (2 * i / d_model)))
            if i + 1 < d_model:
                pe[pos, i + 1] = np.cos(pos / (10000 ** (2 * (i + 1) / d_model)))
    return torch.from_numpy(pe).float()

def main():
    seq_len = 50
    d_model = 512
    pe = get_positional_encoding(seq_len, d_model)
    print("Positional Encoding shape:", pe.shape)
    
    # Visualisasi sederhana baris pertama
    import matplotlib.pyplot as plt
    plt.plot(pe[0, :64].numpy())
    plt.title("Positional Encoding (pos=0, first 64 dims)")
    plt.show()

if __name__ == "__main__":
    main()
```

## 4.5. Arsitektur Encoder-Decoder

Arsitektur encoder-decoder adalah pusat desain model Transformer, terutama dalam tugas-tugas yang memerlukan pembuatan urutan keluaran berdasarkan urutan masukan. Encoder memproses seluruh urutan masukan dan menghasilkan representasi kaya konteks. Decoder kemudian menggunakan representasi ini untuk menghasilkan urutan keluaran token demi token.

![Gambar 7: Arsitektur transformer encoder dan decoder (Makalah Attention is all you need).](https://lmvr.rantai.dev//images/L6A504iArDGBvUq8WSeO-NvakgAoMNmgPRyQtEA34-v1.png)
**Gambar 7:** Arsitektur transformer encoder dan decoder (Makalah Attention is all you need).

Cross-attention adalah fitur utama dari interaksi encoder-decoder. Dalam cross-attention, decoder menggunakan output encoder sebagai matriks "key" dan "value", sementara status decoder saat ini berfungsi sebagai matriks "query".

### Contoh Python: Struktur Blok Encoder dan Decoder Sederhana

```python
import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, forward_expansion * embed_dim),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_dim, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask=None):
        # Self-attention
        attention, _ = self.attention(query, key, value, attn_mask=mask)
        # Add & Norm
        x = self.norm1(self.dropout(attention) + query)
        forward = self.feed_forward(x)
        # Add & Norm
        out = self.norm2(self.dropout(forward) + x)
        return out

def main():
    # Simulasi data masukan
    embed_size = 256
    seq_len = 10
    x = torch.randn(seq_len, 1, embed_size) # (seq, batch, dim)
    
    model = TransformerBlock(embed_size, heads=8, dropout=0.1, forward_expansion=4)
    out = model(x, x, x)
    print("Transformer block output shape:", out.shape)

if __name__ == "__main__":
    main()
```

## 4.6. Layer Normalization dan Residual Connections

Dua teknik yang berkontribusi signifikan pada stabilitas dan kinerja Transformer adalah layer normalization dan residual connections. Mekanisme ini membantu mengatasi tantangan dalam melatih jaringan saraf yang dalam, seperti gradien menghilang (vanishing) dan meledak (exploding).

![Gambar 8: (a) Lapisan Transformer Post-LN; (b) Lapisan Transformer Pre-LN (https://arxiv.org/pdf/2002.04745).](https://lmvr.rantai.dev//images/L6A504iArDGBvUq8WSeO-wQhgGcs195mtRKA8SzN4-v1.png)
**Gambar 8:** (a) Lapisan Transformer Post-LN; (b) Lapisan Transformer Pre-LN (https://arxiv.org/pdf/2002.04745).

Layer normalization menormalkan aktivasi setiap lapisan untuk memastikan mean nol dan varians unit. Berbeda dengan batch normalization, layer normalization beroperasi secara independen pada setiap contoh pelatihan.

$$ \text{LayerNorm}(x) = \frac{x - \mu}{\sigma + \epsilon} \cdot \gamma + \beta $$

![Gambar 9: Koneksi residual dalam arsitektur Transformer.](https://lmvr.rantai.dev//images/L6A504iArDGBvUq8WSeO-u8v0SZiwVq7XlektarlT-v1.png)
**Gambar 9:** Koneksi residual dalam arsitektur Transformer.

Residual connections memungkinkan model untuk "melompati" satu atau lebih lapisan, secara efektif membuat koneksi pintas yang melewatkan masukan langsung ke lapisan yang lebih dalam:
$$ \text{Output} = \text{Layer}(x) + x $$

### Contoh Python: Implementasi LayerNorm dan Residual

```python
import torch
import torch.nn as nn

class CustomTransformerLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.sublayer = nn.Linear(dim, dim)

    def forward(self, x):
        # Post-LN: x = norm(x + sublayer(x))
        # Pre-LN:  x = x + sublayer(norm(x))
        return self.norm(x + self.sublayer(x))

def main():
    x = torch.randn(1, 10, 64)
    layer = CustomTransformerLayer(64)
    print("Output shape:", layer(x).shape)

if __name__ == "__main__":
    main()
```

## 4.7. Teknik Pelatihan dan Optimasi

Melatih model Transformer skala besar menghadirkan tantangan signifikan dalam hal biaya komputasi dan penggunaan memori. Salah satu strategi optimasi utama adalah penjadwalan laju pembelajaran (learning rate scheduling). Transformer menggunakan kombinasi warm-up dan decay laju pembelajaran:

$$ \eta(t) = \frac{1}{\sqrt{d_{\text{model}}}} \min\left( \frac{1}{\sqrt{t}}, \frac{t}{\text{warmup\_steps}^{1.5}} \right) $$

Gradient clipping juga esensial untuk mencegah gradien meledak dengan membatasi norma gradien ke nilai maksimum tertentu. Teknik lainnya adalah mixed precision training, yang memanfaatkan operasi floating point 16-bit (half-precision) dan 32-bit (full-precision) untuk mengurangi penggunaan memori dan mempercepat komputasi tanpa mengorbankan akurasi.

### Contoh Python: Training Loop dengan LR Scheduler dan Gradient Clipping

```python
import torch
import torch.nn as nn
import torch.optim as optim

def main():
    model = nn.Linear(10, 1)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=...) 
    # Sederhana: StepLR
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    for epoch in range(20):
        # Simulasi forward/backward
        inputs = torch.randn(1, 10)
        target = torch.randn(1, 1)
        
        optimizer.zero_grad()
        output = model(inputs)
        loss = torch.mean((output - target)**2)
        loss.backward()
        
        # Gradient Clipping
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}, LR: {scheduler.get_last_lr()[0]}")

if __name__ == "__main__":
    main()
```

## 4.8. Aplikasi Model Transformer

Model Transformer telah merevolusi NLP dengan unggul dalam berbagai tugas. Dalam terjemahan mesin, arsitektur encoder-decoder sangat cocok untuk memetakan urutan sumber $X$ ke urutan target $Y$ dengan memaksimalkan probabilitas $P(Y|X)$. Dalam peringkasan teks, Transformer dapat secara efisien menangkap hubungan antara bagian-bagian dokumen yang berjauhan untuk menghasilkan ringkasan yang koheren.

Versatilitas Transformer datang dari kemampuannya untuk disetel halus (fine-tuned) pada tugas yang berbeda dengan perubahan arsitektur minimal. Pengembang kini berfokus pada pembuatan aplikasi Large Language Model (LLM) khusus yang memanfaatkan prompt engineering, retrieval-augmented generation (RAG), dan agen AI.

### Contoh Python: Implementasi Sederhana LangChain (RAG/Prompt Engineering)

```python
# Pastikan library terinstall: pip install langchain langchain-openai
import os
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def main():
    # Set API Key (simulasi)
    os.environ["OPENAI_API_KEY"] = "sk-..."
    
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    
    prompt = ChatPromptTemplate.from_template("Beritahukan saya tentang {topic} dalam 2 kalimat.")
    
    # Chain sederhana
    chain = prompt | llm | StrOutputParser()
    
    response = chain.invoke({"topic": "Pemrograman Rust"})
    print("Respons AI:", response)

if __name__ == "__main__":
    # main() # Memerlukan API key valid untuk berjalan
    pass
```

### Prompt Engineering, RAG, dan Agen AI

- **Zero-shot Learning:** Menggunakan prompt tanpa contoh, mengandalkan pengetahuan bawaan model.
- **Few-shot Learning:** Menyertakan beberapa contoh dalam prompt untuk memandu respons model.
- **Retrieval-Augmented Generation (RAG):** Menggabungkan LLM dengan sistem pengambilan informasi eksternal untuk memberikan respons yang lebih akurat dan berbasis fakta terbaru.
- **Agen AI:** Aplikasi di mana LLM melakukan alur kerja kompleks secara otonom dengan berinteraksi dengan API eksternal, database, atau sistem AI lainnya.

## 4.9. Kesimpulan

Bab 4 memberikan pemahaman komprehensif tentang arsitektur Transformer, menekankan pendekatan inovatifnya dalam memproses data sekuensial. Dengan menguasai konsep dan teknik yang dibahas, pembaca akan diperlengkapi untuk membangun dan mengoptimalkan model NLP yang kuat yang memanfaatkan potensi penuh dari Transformer.

### 4.9.1. Pembelajaran Lebih Lanjut dengan GenAI

- Jelaskan keterbatasan utama RNN dan CNN tradisional yang menyebabkan pengembangan Transformer.
- Deskripsikan mekanisme self-attention secara rinci (peran Q, K, V).
- Bandingkan konsep self-attention dengan mekanisme atensi tradisional.
- Diskusikan tujuan multi-head attention dalam meningkatkan kemampuan model menangkap pola beragam.
- Jelaskan signifikansi positional encoding dalam mengompensasi kurangnya urutan bawaan.
- Deskripsikan interaksi antara encoder dan decoder melalui cross-attention.
- Analisis peran normalisasi lapisan dan koneksi residual dalam stabilitas pelatihan.
- Diskusikan tantangan melatih model Transformer besar (biaya, memori, optimasi).
- Jelaskan konsep warm-up laju pembelajaran.
- Bandingkan penggunaan model encoder-only (BERT) vs decoder-only (GPT).
- Jelajahi keuntungan skalabilitas Transformer dibandingkan RNN.
- Deskripsikan dampak arsitektur terhadap interpretabilitas model.
- Diskusikan tantangan praktis implementasi Transformer dalam lingkungan produksi.
- Jelaskan konsep cross-attention dalam pembuatan urutan keluaran.
- Diskusikan penggunaan model pra-terlatih dalam transfer learning.
- Analisis kompleksitas komputasi mekanisme self-attention ($O(n^2)$).
- Jelajahi peran augmentasi data dalam meningkatkan kekokohan Transformer.
- Diskusikan pentingnya penyetelan hyperparameter.
- Jelaskan konsep mixed precision training.
- Deskripsikan proses integrasi model Transformer ke dalam lingkungan produksi (model serving).

### 4.9.2. Latihan Praktis

#### Latihan Mandiri 4.1: Mengimplementasikan dan Menganalisis Mekanisme Self-Attention
**Tujuan:** Mendapatkan pemahaman mendalam tentang self-attention dengan mengimplementasikannya dari awal dan menganalisis kinerjanya.

#### Latihan Mandiri 4.2: Mengimplementasikan Multi-Head Attention
**Tujuan:** Memahami manfaat multi-head attention dengan bereksperimen menggunakan jumlah head yang berbeda.

#### Latihan Mandiri 4.3: Penyetelan Halus Model Transformer untuk Tugas Kustom
**Tujuan:** Mempraktikkan fine-tuning model pra-terlatih untuk tugas spesifik seperti analisis sentimen pada dataset khusus.

#### Latihan Mandiri 4.4: Visualisasi Positional Encoding
**Tujuan:** Memahami bagaimana positional encoding memungkinkan model menangkap informasi urutan melalui visualisasi fungsi sinus dan kosinus.

#### Latihan Mandiri 4.5: Optimasi Pelatihan dengan Mixed Precision
**Tujuan:** Mengoptimalkan pelatihan model untuk mengurangi penggunaan memori dan mempercepat waktu pelatihan menggunakan teknik FP16.