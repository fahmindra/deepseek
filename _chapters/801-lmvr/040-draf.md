---
slug: lmvr-4
title: lmvr-4
---
# Bab 4: Arsitektur Transformer

> 💡 **"Model Transformer mewakili pergeseran paradigma dalam pembelajaran mesin, membuktikan bahwa mekanisme atensi, bila diimplementasikan dengan benar, dapat secara signifikan mengungguli metode tradisional dalam menangani data sekuensial." — Geoffrey Hinton**

Bab ini memberikan eksplorasi komprehensif tentang komponen-komponen utama yang menjadikan model Transformer sebagai fondasi bagi model bahasa besar modern. Bab ini dimulai dengan pengantar arsitektur Transformer, menjelaskan bagaimana arsitektur ini merevolusi pemrosesan bahasa alami melalui kemampuan paralelisasi. Bab ini mendalami mekanisme *self-attention*, yang memungkinkan model menimbang pentingnya kata-kata yang berbeda dalam sebuah kalimat, diikuti oleh diskusi mendalam tentang *multi-head attention*, di mana beberapa kepala atensi menangkap berbagai hubungan kontekstual. Selanjutnya, bab ini menjelaskan pengodean posisi (*positional encoding*), yang krusial untuk merepresentasikan urutan kata, dan membedah arsitektur encoder-decoder yang menggerakkan tugas-tugas kompleks seperti penerjemahan. Normalisasi lapisan (*layer normalization*) dan koneksi residual, yang esensial untuk pelatihan yang stabil dan efisien, dibahas secara menyeluruh, bersama dengan teknik pelatihan dan optimasi yang meningkatkan performa. Terakhir, bab ini diakhiri dengan aplikasi dunia nyata dari model Transformer, menyoroti dampaknya pada berbagai tugas NLP seperti penerjemahan mesin, pembuatan teks, dan peringkasan.

---

## 4.1. Pengantar Model Transformer

Pengenalan model Transformer merevolusi Pemrosesan Bahasa Alami (NLP) dengan mengatasi keterbatasan utama arsitektur tradisional seperti *Recurrent Neural Networks* (RNNs) dan *Convolutional Neural Networks* (CNNs). Baik RNN maupun CNN, meskipun efektif dalam banyak tugas berbasis urutan, kesulitan untuk menangani ketergantungan jarak jauh (*long-range dependencies*) dalam teks. Dalam RNN, data sekuensial harus diproses satu langkah waktu pada satu waktu, sehingga menantang untuk menangkap ketergantungan antara token yang berjauhan, terutama dalam urutan yang panjang. Sifat sekuensial ini juga membatasi paralelisasi selama pelatihan dan inferensi, yang secara signifikan mengurangi efisiensi. CNN, meskipun lebih baik dalam menangani komputasi paralel, dibatasi oleh ukuran filter konvolusional, membuatnya tidak cocok untuk memodelkan ketergantungan yang mencakup bagian teks yang luas. Keterbatasan ini menciptakan kebutuhan akan model yang dapat memproses ketergantungan jarak jauh secara efisien dan berskala dengan perangkat keras modern. Kebutuhan inilah yang menyebabkan pengembangan model Transformer.

Arsitektur Transformer, yang diperkenalkan dalam makalah *Attention is All You Need*, menjawab tantangan ini dengan mengganti rekurensi dan konvolusi dengan mekanisme yang disebut *self-attention*. Inovasi utama ini memungkinkan Transformer untuk menangkap hubungan antar token tanpa mempedulikan jarak mereka dalam urutan, sambil memproses urutan tersebut secara paralel. Arsitektur ini terdiri dari dua komponen utama: *Encoder* dan *Decoder*, yang keduanya dibangun menggunakan lapisan *self-attention* dan jaringan saraf *feedforward*. Encoder memproses urutan masukan dan menghasilkan sekumpulan representasi yang sadar konteks, sementara decoder menghasilkan urutan keluaran dengan memperhatikan masukan yang dikodekan dan keluaran yang dihasilkan sebelumnya. Struktur modular ini sangat fleksibel dan dapat diadaptasi ke berbagai tugas NLP, seperti penerjemahan, pembuatan teks, dan peringkasan.

Secara matematis, mekanisme *self-attention* membentuk inti dari model Transformer. Diberikan urutan masukan $X = [x_1, x_2, \dots, x_n]$, model pertama-tama mengubah setiap token menjadi tiga vektor: *Query* ($Q$), *Key* ($K$), dan *Value* ($V$). Mekanisme atensi menghitung jumlah tertimbang dari vektor *value*, di mana bobotnya ditentukan oleh hasil kali titik (*dot product*) dari vektor *query* dan *key*. Atensi *scaled dot-product* didefinisikan sebagai:

$$ \text{Attention}(Q, K, V) = \text{softmax}\left( \frac{QK^T}{\sqrt{d_k}} \right) V $$

Di sini, $d_k$ adalah dimensionalitas dari vektor *key*, dan fungsi softmax memastikan bahwa bobot atensi berjumlah 1, membentuk distribusi probabilitas. Hal ini memungkinkan model untuk lebih fokus pada bagian masukan yang relevan dengan token saat ini, menangkap ketergantungan jarak pendek dan jarak jauh dalam satu langkah. Dengan menghitung atensi untuk semua token secara paralel, Transformer dapat memproses seluruh urutan secara efisien, menjadikannya sangat skalabel dan cocok untuk dataset besar.

Salah satu keunggulan utama model Transformer adalah peralihannya dari pemrosesan sekuensial ke paralel. RNN tradisional memproses satu token pada satu waktu, sehingga sulit untuk diparalelkan di beberapa unit pemrosesan. Sebaliknya, Transformer memproses semua token dalam urutan secara bersamaan, memungkinkan percepatan besar-besaran pada perangkat keras modern seperti GPU dan TPU. Paralelisasi ini sangat krusial saat bekerja dengan korpus teks yang besar atau urutan yang panjang, karena secara signifikan mengurangi waktu pelatihan sambil mempertahankan kinerja tinggi. Kemampuan untuk menangani urutan panjang secara efisien telah menjadikan model Transformer sebagai fondasi bagi banyak model bahasa mutakhir, seperti BERT dan GPT.

![Figure 1: Arsitektur GPT vs BERT.](https://lmvr.rantai.dev/images/L6A504iArDGBvUq8WSeO-l27fwg4yZdgjT1n0eESy-v1.png)

Mekanisme atensi dalam Transformer memainkan peran sentral dalam meningkatkan efisiensi dan efektivitas model. Berbeda dengan RNN, yang kesulitan menangkap ketergantungan jarak jauh karena masalah *vanishing gradient*, atensi memungkinkan Transformer untuk memperhatikan secara langsung bagian mana pun dari masukan, terlepas dari jaraknya. Hal ini membuat model lebih kuat dalam tugas-tugas yang memerlukan pemahaman hubungan jarak jauh yang kompleks antar kata. Misalnya, dalam penerjemahan mesin, Transformer dapat secara langsung memetakan kata-kata dalam kalimat sumber ke kata-kata yang tepat dalam kalimat target, bahkan ketika kata-kata tersebut berjauhan.

Berikut adalah implementasi Python menggunakan perpustakaan `transformers` untuk mendemonstrasikan analisis sentimen dengan model BERT terlatih (padanan untuk contoh Rust):

```python
import torch
from transformers import pipeline

def main():
    # Menentukan perangkat (GPU jika tersedia)
    device = 0 if torch.cuda.is_available() else -1
    
    # Memuat pipeline analisis sentimen dengan model BERT default
    classifier = pipeline("sentiment-analysis", device=device)
    
    # Contoh teks untuk dianalisis
    sample_texts = [
        "I loved this movie!",
        "This was a terrible experience."
    ]
    
    # Melakukan prediksi
    results = classifier(sample_texts)
    
    print("Prediksi BERT:")
    for text, result in zip(sample_texts, results):
        print(f"Teks: {text} | Label: {result['label']} | Skor: {result['score']:.4f}")

if __name__ == "__main__":
    main()
```

Selanjutnya, mari kita lihat tugas NLP lainnya seperti pembuatan teks menggunakan GPT-2. Kode Python berikut menunjukkan penggunaan model GPT-2 untuk melengkapi teks:

```python
from transformers import pipeline, set_seed

def main():
    # Inisialisasi generator teks dengan GPT-2
    generator = pipeline('text-generation', model='gpt2')
    set_seed(42) # Agar hasil reproduksibel

    dataset = [
        "The future of AI is",
        "In 2024, the world will",
        "Technology advancements will lead to"
    ]

    print("Pembuatan Teks Berbasis GPT-2:")
    for prompt in dataset:
        output = generator(prompt, max_length=30, num_return_sequences=1)
        print(f"Input: {prompt}")
        print(f"Hasil: {output[0]['generated_text']}\n")

if __name__ == "__main__":
    main()
```

Terakhir, mari kita bandingkan peringkasan teks menggunakan BART (model berbasis BERT untuk urutan-ke-urutan) dan GPT-2:

```python
from transformers import pipeline

def main():
    # Muat model peringkasan BART
    summarizer_bart = pipeline("summarization", model="facebook/bart-large-cnn")
    
    # Muat model GPT-2 untuk simulasi peringkasan (melalui pembuatan teks)
    generator_gpt2 = pipeline("text-generation", model="gpt2")

    dataset = [
        "The quick brown fox jumps over the lazy dog. This is a sentence that represents a very common phrase used in typing tests. The fox is quick and brown, and the dog is lazy and tired.",
        "Artificial intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think and act like humans. AI is being used in various fields such as healthcare, finance, and technology."
    ]

    print("Peringkasan Berbasis BART:")
    summaries_bart = summarizer_bart(dataset, max_length=20, min_length=5, do_sample=False)
    for i, s in enumerate(summaries_bart):
        print(f"Hasil BART {i+1}: {s['summary_text']}")

    print("\nGenerasi GPT-2 (sebagai ringkasan):")
    for prompt in dataset:
        # Menambahkan instruksi ringkasan untuk GPT-2
        output = generator_gpt2(prompt + " TL;DR:", max_length=100, num_return_sequences=1)
        print(f"Hasil GPT-2: {output[0]['generated_text']}\n")

if __name__ == "__main__":
    main()
```

---

## 4.2. Mekanisme Self-Attention

Mekanisme *self-attention* memungkinkan model untuk fokus pada bagian tertentu dari urutan masukan yang paling relevan untuk tugas tertentu. Berbeda dengan model tradisional, *self-attention* memungkinkan model untuk menetapkan tingkat kepentingan (bobot) yang berbeda pada setiap token dalam urutan.

![Figure 2: Ilustrasi mekanisme self-attention pada encoder (Kredit Ketan Doshi).](https://lmvr.rantai.dev/images/L6A504iArDGBvUq8WSeO-VYj8ePTNWmhtqRGsGNXm-v1.png)

Secara matematis, proses ini melibatkan pembuatan vektor *query*, *key*, dan *value* melalui matriks bobot yang dipelajari. Untuk urutan masukan $X$, setiap token $x_i$ dipetakan menjadi $Q_i$, $K_i$, dan $V_i$. Skor atensi antara token $i$ dan $j$ dihitung dengan:

$$ \text{Score}(x_i, x_j) = Q_i \cdot K_j $$

Skor ini kemudian diskalakan dengan $\sqrt{d_k}$ dan dilewatkan ke fungsi softmax:

$$ \text{Attention}(Q, K, V) = \text{softmax}\left( \frac{QK^T}{\sqrt{d_k}} \right) V $$

![Figure 3: Multi Head Attention (Kredit Ketan Doshi).](https://lmvr.rantai.dev/images/L6A504iArDGBvUq8WSeO-NpCGTy0HgoOugVLNGZkW-v1.png)

Peningkatan utama dari *self-attention* adalah *multi-head attention*, yang menghitung beberapa set skor atensi secara paralel menggunakan matriks bobot yang berbeda. Hal ini memungkinkan model menangkap berbagai jenis hubungan (misalnya sintaksis vs semantik) secara bersamaan.

Berikut adalah implementasi sederhana *Scaled Dot-Product Attention* menggunakan PyTorch:

```python
import torch
import torch.nn.functional as F

def scaled_dot_product_attention(query, key, value):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / (d_k ** 0.5)
    attention_weights = F.softmax(scores, dim=-1)
    return torch.matmul(attention_weights, value)

# Contoh penggunaan
batch_size, seq_len, embed_dim = 10, 20, 64
q = torch.randn(batch_size, seq_len, embed_dim)
k = torch.randn(batch_size, seq_len, embed_dim)
v = torch.randn(batch_size, seq_len, embed_dim)

output = scaled_dot_product_attention(q, k, v)
print(f"Output shape: {output.shape}")
```

Untuk mengoptimalkan *self-attention* pada urutan yang sangat panjang, teknik seperti *sparse attention* dikembangkan. Model seperti *Longformer* memungkinkan token untuk hanya memperhatikan subset dari urutan masukan (misalnya, jendela lokal), sehingga mengurangi kompleksitas kuadratik menjadi linear.

![Figure 4: Ilustrasi mekanisme sparse attention.](https://lmvr.rantai.dev/images/L6A504iArDGBvUq8WSeO-DRuhbtDvveXg8unkLTtM-v1.png)

Linformer adalah varian lain yang memproyeksikan matriks atensi ke ruang dimensi yang lebih rendah. Berikut adalah simulasi arsitektur Linformer di Python:

```python
import torch
import torch.nn as nn

class LinformerAttention(nn.Module):
    def __init__(self, embed_dim, proj_dim):
        super().__init__()
        self.wq = nn.Linear(embed_dim, embed_dim)
        self.wk = nn.Linear(embed_dim, embed_dim)
        self.wv = nn.Linear(embed_dim, embed_dim)
        
        # Proyeksi linear untuk mengurangi dimensi sequence length
        self.proj = nn.Linear(embed_dim, proj_dim) 
        # (Catatan: Dalam Linformer asli, proyeksi dilakukan pada dimensi sequence length)

    def forward(self, x):
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)
        
        # Simulasi proyeksi Linformer (sederhana)
        # Di sini kita memproyeksikan embed_dim, bukan seq_len demi kemudahan contoh
        k_proj = self.proj(k) 
        
        return q, k_proj, v

# Contoh objek Linformer sederhana
```

---

## 4.3. Multi-Head Attention

*Multi-head attention* memperluas kemampuan model dengan membagi urutan menjadi $h$ kepala yang berbeda. Setiap kepala beroperasi pada subruang yang terpisah, memungkinkan model untuk fokus pada aspek urutan yang berbeda secara bersamaan.

$$ \text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O $$

Implementasi Python untuk *Multi-Head Attention* dari awal:

```python
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.out_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, q, k, v):
        bs = q.size(0)
        
        # Proyeksi dan bagi menjadi kepala-kepala
        q = self.q_linear(q).view(bs, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(k).view(bs, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(v).view(bs, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Atensi
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        weights = torch.softmax(scores, dim=-1)
        attn_out = torch.matmul(weights, v)

        # Gabungkan kembali
        attn_out = attn_out.transpose(1, 2).contiguous().view(bs, -1, self.num_heads * self.head_dim)
        return self.out_linear(attn_out)

# Inisialisasi
mha = MultiHeadAttention(64, 8)
input_tensor = torch.randn(10, 20, 64)
output = mha(input_tensor, input_tensor, input_tensor)
print(f"MHA Output: {output.shape}")
```

---

## 4.4. Positional Encoding

Karena Transformer tidak memiliki rekurensi, ia tidak tahu posisi relatif kata-kata. Pengodean posisi menyuntikkan informasi urutan ke dalam embedding masukan menggunakan fungsi sinusoidal:

$$ PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right) $$
$$ PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right) $$

Implementasi Python untuk *Positional Encoding*:

```python
import numpy as np
import torch

def get_positional_encoding(seq_len, d_model):
    pe = np.zeros((seq_len, d_model))
    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            div_term = np.exp(i * -(np.log(10000.0) / d_model))
            pe[pos, i] = np.sin(pos * div_term)
            pe[pos, i + 1] = np.cos(pos * div_term)
    return torch.FloatTensor(pe)

pe_matrix = get_positional_encoding(50, 512)
print(f"PE Matrix Shape: {pe_matrix.shape}")
```

---

## 4.5. Arsitektur Encoder-Decoder

Arsitektur *encoder-decoder* membagi model menjadi dua bagian: encoder memahami masukan, dan decoder menghasilkan keluaran token demi token. Keduanya dihubungkan melalui mekanisme *cross-attention*.

![Figure 7: Arsitektur encoder dan decoder transformer (Makalah Attention is all you need).](https://lmvr.rantai.dev/images/L6A504iArDGBvUq8WSeO-NvakgAoMNmgPRyQtEA34-v1.png)

Implementasi konseptual di Python menggunakan kelas bawaan PyTorch:

```python
import torch.nn as nn

# Menggunakan modul Transformer bawaan PyTorch
transformer_model = nn.Transformer(
    d_model=512, 
    nhead=8, 
    num_encoder_layers=6, 
    num_decoder_layers=6
)

# src: [batch, seq_len, embed]
# tgt: [batch, seq_len, embed]
src = torch.randn(10, 20, 512)
tgt = torch.randn(10, 20, 512)
out = transformer_model(src, tgt)
print(f"Transformer output shape: {out.shape}")
```

---

## 4.6. Normalisasi Lapisan dan Koneksi Residual

Koneksi residual memungkinkan gradien mengalir melewati lapisan tanpa mengecil, sementara normalisasi lapisan memastikan aktivasi memiliki rata-rata nol dan varians unit.

$$ \text{LayerNorm}(x) = \frac{x - \mu}{\sigma + \epsilon} \cdot \gamma + \beta $$
$$ \text{Output} = \text{Layer}(x) + x $$

Blok Transformer tingkat lanjut di Python:

```python
import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, dropout_prob=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=8)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        # Sublayer 1: Attention + Residual + Norm
        attn_out, _ = self.attn(x, x, x)
        x = self.ln1(x + self.dropout(attn_out))
        
        # Sublayer 2: FeedForward + Residual + Norm
        ff_out = self.ff(x)
        x = self.ln2(x + self.dropout(ff_out))
        return x
```

---

## 4.7. Teknik Pelatihan dan Optimasi

Melatih Transformer besar membutuhkan strategi khusus:
1.  **Learning Rate Scheduling:** Menggunakan *warm-up* di awal dan peluruhan (*decay*) kemudian.
2.  **Gradient Clipping:** Membatasi norma gradien untuk mencegah ledakan gradien.
3.  **Mixed Precision:** Menggunakan FP16 untuk kecepatan dan penghematan memori.

Berikut adalah simulasi loop pelatihan dengan scheduler di Python:

```python
import torch
import torch.optim as optim

model = nn.Linear(10, 2)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Scheduler sederhana dengan pemanasan (linear warmup)
def lr_lambda(current_step):
    warmup_steps = 100
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    return 1.0

scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# Loop pelatihan
for step in range(200):
    inputs = torch.randn(1, 10)
    target = torch.randn(1, 2)
    
    optimizer.zero_grad()
    output = model(inputs)
    loss = F.mse_loss(output, target)
    loss.backward()
    
    # Gradient Clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    scheduler.step()
```

---

## 4.8. Aplikasi Model Transformer

Model Transformer kini digunakan secara luas di berbagai industri, mulai dari Google Translate hingga asisten virtual. Pengembang kini fokus pada pembuatan aplikasi khusus menggunakan teknik seperti:

*   **Prompt Engineering:** Merancang input untuk memandu respon LLM (Zero-shot, Few-shot, Chain-of-thought).
*   **Retrieval-Augmented Generation (RAG):** Menggabungkan LLM dengan sistem pencarian informasi eksternal untuk jawaban yang lebih akurat.
*   **AI Agents:** Sistem otonom yang menggunakan LLM untuk menjalankan alur kerja kompleks melalui API.

Berikut adalah contoh penggunaan `LangChain` di Python untuk membuat rantai prompt sederhana (padanan untuk `anchor-chain` atau `langchain-rust`):

```python
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain
import os

# Membutuhkan API Key OpenAI
# os.environ["OPENAI_API_KEY"] = "sk-..."

def main():
    llm = OpenAI(temperature=0.7)
    prompt = PromptTemplate(
        input_variables=["product"],
        template="What is a good name for a company that makes {product}?",
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    
    # Menjalankan rantai
    # result = chain.run("eco-friendly water bottles")
    # print(result)
    print("Contoh LangChain dikonfigurasi.")

if __name__ == "__main__":
    main()
```

---

## 4.9. Kesimpulan

Bab 4 memberikan pemahaman komprehensif tentang arsitektur Transformer, menekankan pendekatan inovatifnya terhadap data sekuensial dan implementasi praktisnya. Dengan menguasai konsep-konsep ini, pembaca akan siap membangun model NLP yang kuat.

### 4.9.1. Pembelajaran Lebih Lanjut dengan GenAI

- Jelaskan keterbatasan utama RNN dan CNN tradisional yang memicu lahirnya Transformer.
- Deskripsikan mekanisme *self-attention* secara mendetail (Q, K, V).
- Apa kegunaan *multi-head attention* dalam menangkap pola data yang beragam?
- Mengapa pengodean posisi sangat penting dalam model Transformer?
- Analisis peran normalisasi lapisan dan koneksi residual dalam stabilitas pelatihan.

### 4.9.2. Praktik Mandiri

*   **Latihan 4.1:** Implementasikan mekanisme *self-attention* dari nol dan hitung skornya untuk kalimat pendek.
*   **Latihan 4.2:** Buat model *multi-head attention* dan bandingkan hasilnya dengan *single-head* pada dataset klasifikasi teks.
*   **Latihan 4.3:** Lakukan *fine-tuning* pada model Transformer (misalnya BERT) untuk dataset kustom analisis sentimen.
*   **Latihan 4.4:** Visualisasikan matriks pengodean posisi sinusoidal untuk berbagai panjang urutan.
*   **Latihan 4.5:** Optimalkan pelatihan Transformer menggunakan *mixed precision* dan amati penghematan memorinya.