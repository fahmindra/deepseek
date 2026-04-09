---
slug: lmvr-6
title: lmvr-6
---
# Bab 6: Model Generatif: GPT dan Seterusnya

> 💡 **"Model generatif seperti GPT mewakili lompatan besar dalam kemampuan kita untuk menyintesis teks yang menyerupai buatan manusia, mencerminkan potensi mendalam AI untuk memahami dan menghasilkan bahasa yang kompleks." — Yann LeCun**

> 📘 **Ringkasan Bab:**
> Bab 6 dari LMVR memberikan eksplorasi mendalam tentang model generatif, dengan fokus khusus pada GPT dan varian-variannya. Bab ini dimulai dengan memperkenalkan dasar-dasar model generatif, membedakannya dari model diskriminatif, dan mendiskusikan aplikasinya dalam pemrosesan bahasa alami (NLP). Kemudian, bab ini mendalami arsitektur GPT, menjelaskan sifat autoregresifnya, proses pelatihannya, dan manfaat memanfaatkan model Transformer. Bab ini juga mencakup kemajuan dalam varian GPT, termasuk GPT-2 dan GPT-3, memeriksa dampak penskalaan terhadap performa dan pertimbangan etis dalam menyebarkan model besar. Bagian praktis mencakup implementasi model generatif dasar, melatih model GPT, dan membandingkan varian GPT yang berbeda untuk memahami kekuatan dan keterbatasan mereka.

---

## 6.1. Pengantar Model Generatif

Model generatif memainkan peran krusial dalam pembelajaran mesin dengan memodelkan distribusi dasar data, memungkinkan mereka untuk menghasilkan instans baru yang menyerupai data tempat mereka dilatih. Ini berbeda dengan model diskriminatif, yang fokus pada membedakan antara kelas data yang berbeda. Model generatif bertujuan untuk memahami dan mereproduksi distribusi data itu sendiri, yang membuatnya sangat berguna dalam tugas-tugas seperti pembuatan teks, peringkasan, dan penerjemahan dalam domain pemrosesan bahasa alami (NLP). Sementara model diskriminatif seperti jaringan klasifikasi dirancang untuk menetapkan label atau kategori ke input, model generatif menangkap struktur dan pola dalam data untuk membuat output baru yang masuk akal.

![Figure 1: Model Diskriminatif vs Generatif.](https://lmvr.rantai.dev/images/L6A504iArDGBvUq8WSeO-rCRQPzy7GRYxSIgPEhvZ-v1.png)

Dalam konteks NLP, model generatif telah menjadi fundamental dalam menghasilkan teks berkualitas tinggi yang menyerupai buatan manusia. Aplikasinya berkisar dari menghasilkan kalimat yang koheren hingga meringkas informasi dalam jumlah besar menjadi teks yang ringkas dan bermakna, dan bahkan menerjemahkan bahasa dengan akurasi yang luar biasa. Model seperti GPT (*Generative Pretrained Transformer*) mencontohkan kemampuan model generatif untuk membuat teks yang cair dan sesuai konteks berdasarkan input atau perintah yang diberikan. Model-model ini tidak hanya menghasilkan urutan acak; sebaliknya, mereka memodelkan pola bahasa, tata bahasa, dan semantik.

Inti dari model generatif terletak pada kemampuan mereka untuk mempelajari distribusi data. Ini melibatkan penangkapan distribusi probabilitas $P(X)$, di mana $X$ mewakili data, seperti kalimat dalam korpus. Untuk model autoregresif, ide kuncinya adalah memfaktorkan distribusi probabilitas gabungan dari urutan kata menjadi produk dari probabilitas bersyarat. Sebagai contoh, dalam kasus pembuatan teks, probabilitas seluruh urutan dimodelkan sebagai:

$$P(X) = P(x_1)P(x_2 | x_1)P(x_3 | x_1, x_2) \dots P(x_n | x_1, x_2, \dots, x_{n-1}),$$

di mana $x_i$ mewakili setiap kata dalam urutan. Dengan mempelajari probabilitas bersyarat ini, model dapat menghasilkan teks baru kata demi kata, berdasarkan kata-kata yang dihasilkan sebelumnya.

Model generatif seperti GPT menggunakan metode autoregresif, di mana output pada setiap langkah dikondisikan pada token sebelumnya dalam urutan. Arsitektur Transformer di balik GPT telah merevolusi pemodelan generatif karena penanganannya yang efisien terhadap ketergantungan jarak jauh melalui mekanisme *self-attention*. Dalam mekanisme ini, setiap token dalam urutan input memperhatikan semua token lainnya, memungkinkan model untuk menangkap pola lokal dan global dalam data.

Konsep penting lainnya yang menggerakkan model generatif modern adalah *self-supervised learning*. Dalam pembelajaran mandiri ini, model dilatih pada tugas-tugas di mana bagian dari data input disamarkan atau dirusak, dan model diminta untuk memprediksi bagian yang hilang. Teknik ini membentuk fondasi dari banyak model mutakhir di NLP saat ini.

Salah satu tantangan awal yang dihadapi oleh model generatif adalah kesulitan dalam menghasilkan urutan jangka panjang yang koheren. RNN dan LSTM kesulitan mempertahankan konsistensi pada teks yang panjang karena keterbatasan inheren mereka. Namun, dengan pengenalan arsitektur Transformer, tantangan ini telah berkurang secara signifikan.

Tolok ukur (*benchmarking*) model generatif terhadap baseline sederhana sangat penting untuk mengevaluasi performa. Dalam kasus pembuatan teks, seseorang dapat membandingkan kualitas teks yang dihasilkan menggunakan metrik seperti *perplexity*, yang mengukur seberapa baik model memprediksi kata berikutnya dalam suatu urutan. *Perplexity* yang lebih rendah menunjukkan performa yang lebih baik.

Sejak 2018, model GPT OpenAI telah mengalami peningkatan pesat dalam jumlah parameter, mencerminkan kemajuan dalam kompleksitas, kapasitas, dan performa model. GPT asli (2018) memiliki sekitar 117 juta parameter. GPT-2 meningkat signifikan menjadi 1,5 miliar parameter. Pada tahun 2020, GPT-3 dirilis dengan 175 miliar parameter. Peningkatan ini didorong oleh kebutuhan model untuk menangkap pola bahasa dan pengetahuan dunia yang semakin kompleks.

![Figure 2: Jumlah parameter model-model GPT.](https://lmvr.rantai.dev/images/L6A504iArDGBvUq8WSeO-1yyOzCHsIb0oYBXNfNtR-v1.webp)

Berikut adalah contoh kode Python yang mendemonstrasikan berbagai teknik *prompt engineering* menggunakan perpustakaan `langchain` (padanan untuk `langchain-rust`):

```python
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.prompts import ChatPromptTemplate

def main():
    # Inisialisasi model (membutuhkan API Key OpenAI)
    llm = ChatOpenAI(model_name="gpt-4-turbo")

    # 1. Spesifikasi Peran (Role Specification)
    messages = [
        SystemMessage(content="Anda adalah penulis dokumentasi teknis kelas dunia dengan pengetahuan mendalam tentang pemrograman Rust."),
        HumanMessage(content="Jelaskan Rust dalam istilah sederhana.")
    ]
    response = llm(messages)
    print(f"Role Response: {response.content}\n")

    # 2. Format Respons (Few-Shot & Constraints)
    prompt = ChatPromptTemplate.from_template(
        "Anda adalah penulis teknis yang ringkas. Jawab dalam tiga poin bulet.\n"
        "Input: {input}"
    )
    formatted_prompt = prompt.format_messages(input="Apa manfaat utama dari Rust?")
    response = llm(formatted_prompt)
    print(f"Formatted Response: {response.content}\n")

    # 3. Konteks Historis (History)
    history_messages = [
        SystemMessage(content="Anda adalah asisten yang mengingat konteks."),
        HumanMessage(content="Nama saya Luis."),
        AIMessage(content="Halo Luis, senang bertemu denganmu!"),
        HumanMessage(content="Siapa penulis buku 20,000 Leagues Under the Sea?")
    ]
    response = llm(history_messages)
    print(f"History Response: {response.content}\n")

if __name__ == "__main__":
    main()
```

---

## 6.2. Arsitektur GPT

Arsitektur GPT (*Generative Pre-trained Transformer*) didasarkan pada arsitektur Transformer, yang berfokus pada pembuatan teks autoregresif. GPT menghasilkan teks satu token pada satu waktu, menggunakan distribusi probabilitas dari token berikutnya yang dikondisikan pada token sebelumnya.

![Figure 3: Ilustrasi arsitektur GPT-2.](https://lmvr.rantai.dev/images/L6A504iArDGBvUq8WSeO-u3gVcexpmTkzBQnrETnr-v1.png)

Secara matematis, proses pembuatan urutan $x_1, x_2, \dots, x_n$ dapat dijelaskan dengan rumus:

$$P(x_1, x_2, \dots, x_n) = \prod_{i=1}^{n} P(x_i | x_1, x_2, \dots, x_{i-1}),$$

Proses pelatihan GPT melibatkan dua tahap utama: *pre-training* (pelatihan awal) dan *fine-tuning* (penyesuaian). Tahap pre-training menggunakan tugas pemodelan bahasa dengan fungsi kerugian *negative log-likelihood*:

$$\mathcal{L} = - \sum_{i=1}^{n} \log P(x_i | x_1, x_2, \dots, x_{i-1}),$$

GPT biasanya menggunakan tokenisasi subkata seperti *Byte Pair Encoding* (BPE). Meskipun sangat lancar, model GPT memiliki batasan seperti "halusinasi"—menghasilkan informasi yang tampak benar tetapi sebenarnya salah secara faktual.

Berikut adalah implementasi Python sederhana untuk arsitektur GPT tingkat karakter menggunakan PyTorch (padanan untuk contoh `minGPT` di Rust):

```python
import torch
import torch.nn as nn
from torch.nn import functional as F

# Hyperparameters sederhana
batch_size = 64
block_size = 128
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_embd = 512
n_head = 8
n_layer = 8

class CausalSelfAttention(nn.Module):
    def __init__(self, n_head, n_embd):
        super().__init__()
        self.n_head = n_head
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size))
                             .view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / (C // self.n_head)**0.5)
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.sa = CausalSelfAttention(n_head, n_embd)
        self.ffwd = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
        )
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# Implementasi Model GPT Lengkap dapat dilanjutkan dari blok-blok di atas
```

![Figure 4: Pertumbuhan blok transformer pada berbagai model GPT-2.](https://lmvr.rantai.dev/images/L6A504iArDGBvUq8WSeO-oHvUxZ4KXIC2oSvMZAtG-v1.png)

---

## 6.3. Varian dan Ekstensi GPT

Penskalaan arsitektur dan data pelatihan telah menjadi faktor sentral kesuksesan GPT. GPT-2 meningkat menjadi 1,5 miliar parameter, sementara GPT-3 melonjak ke 175 miliar parameter. Tren ini sering disebut sebagai "Hukum Moore di NLP", di mana jumlah parameter meningkat secara eksponensial di setiap generasi.

![Figure 5: Hukum Moore di NLP.](https://lmvr.rantai.dev/images/L6A504iArDGBvUq8WSeO-I8HKwFaE7jawZvvgfEwc-v1.png)

Penskalaan ini didasarkan pada *scaling laws*, yang menyatakan bahwa performa meningkat secara terprediksi seiring bertambahnya ukuran model ($N$), ukuran dataset ($D$), dan komputasi ($C$):

$$L(N, D, C) = aN^{-\alpha} + bD^{-\beta} + cC^{-\gamma},$$

![Figure 6: Perbandingan ukuran model varian GPT.](https://lmvr.rantai.dev/images/L6A504iArDGBvUq8WSeO-mqVgJ3qL9pcY9rz0DS6w-v1.png)

Pertimbangan etis mencakup penyebaran bias dari data pelatihan dan dampak lingkungan dari konsumsi energi yang sangat besar selama pelatihan.

Berikut adalah contoh penggunaan Python untuk membandingkan kecepatan inferensi GPT-2 dan GPT-Neo (sebagai padanan contoh Rust):

```python
import time
from transformers import pipeline

def run_benchmark(model_name, prompt):
    print(f"Loading {model_name}...")
    generator = pipeline('text-generation', model=model_name, device=0)
    
    start = time.time()
    output = generator(prompt, max_length=50)
    end = time.time()
    
    print(f"{model_name} took {end - start:.2f} seconds.")
    print(f"Output: {output[0]['generated_text']}\n")

def main():
    prompt = "Once upon a time"
    run_benchmark("gpt2", prompt)
    run_benchmark("EleutherAI/gpt-neo-125M", prompt)

if __name__ == "__main__":
    main()
```

---

## 6.4. Pelatihan, Fine-Tuning, dan Adaptasi Tugas Model GPT

OpenAI menggunakan pendekatan terstruktur yang disebut *Reinforcement Learning with Human Feedback* (RLHF) untuk melatih model GPT. Proses ini melibatkan tiga tahap:

1.  **Supervised Fine-Tuning (SFT):** Melatih model dasar pada data yang dikurasi manusia.
    $$\mathcal{L}_{\text{SFT}}(\theta) = - \sum_{(X, Y) \in \text{Dataset}} \log P_\theta(Y | X)$$
2.  **Pelatihan Model Imbalan (Reward Model):** Melatih model $R_\phi$ untuk memberikan skor pada kualitas output berdasarkan peringkat manusia.
    $$\mathcal{L}_{\text{Reward}}(\phi) = \mathbb{E}_{(Y_1, Y_2) \sim D} \left[ \log \sigma\left(R_\phi(Y_1) - R_\phi(Y_2)\right) \right]$$
3.  **Proximal Policy Optimization (PPO):** Mengoptimalkan kebijakan model terhadap model imbalan agar respons sesuai dengan preferensi manusia.

![Figure 7: Cara OpenAI melatih model GPT 3.5.](https://lmvr.rantai.dev/images/L6A504iArDGBvUq8WSeO-J67zNjyqKxM6ioNt2Pf6-v1.png)

Berikut adalah simulasi sederhana logika pelatihan PPO di Python:

```python
import torch
import torch.nn as nn
import torch.optim as optim

def ppo_update_step(policy_model, states, actions, old_log_probs, advantages, eps=0.2):
    optimizer = optim.Adam(policy_model.parameters(), lr=1e-4)
    
    # Menghitung probabilitas baru
    logits = policy_model(states)
    new_log_probs = torch.log_softmax(logits, dim=-1).gather(1, actions)
    
    # Rasio probabilitas
    ratio = torch.exp(new_log_probs - old_log_probs)
    
    # Fungsi objektif PPO yang dipotong (clipped)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - eps, 1 + eps) * advantages
    loss = -torch.min(surr1, surr2).mean()
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()
```

---

## 6.5. Teknik Generatif Lanjutan di Luar GPT

Selain model autoregresif, terdapat teknik lain:

1.  **Variational Autoencoders (VAEs):** Memetakan data ke ruang laten Gaussian.
    $$\mathcal{L}_{\text{VAE}} = \mathbb{E}_{q(z|x)} \left[ \log p(x|z) \right] - \text{KL}(q(z|x) || p(z))$$
2.  **Generative Adversarial Networks (GANs):** Pertarungan antara Generator ($G$) dan Diskriminator ($D$).
3.  **Diffusion Models:** Menambahkan derau (*noise*) secara bertahap dan belajar membalikkannya.
    $$\mathcal{L}_{\text{diffusion}} = \mathbb{E}_{q} \left[ \| x_t - x \|^2 \right]$$

![Figure 8: Perbandingan antara model GAN, VAE, dan Diffusion.](https://lmvr.rantai.dev/images/L6A504iArDGBvUq8WSeO-u5mui8fwOw66TBwzmAJD-v1.webp)

Berikut adalah contoh implementasi VAE sederhana di Python:

```python
class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super().__init__()
        # Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim) # mu
        self.fc22 = nn.Linear(hidden_dim, latent_dim) # logvar
        # Decoder
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
```

Stable Diffusion adalah contoh implementasi model difusi yang menggunakan CLIP untuk pengodean teks, VAE untuk kompresi gambar, dan UNet untuk penghilangan derau (*denoising*).

---

## 6.6. Arsitektur GPT vs LLaMA

GPT dan LLaMA memiliki pendekatan yang berbeda meskipun keduanya menggunakan decoder-only Transformer.

-   **GPT:** Fokus pada penskalaan lapisan dan ukuran model secara padat. Menggunakan normalisasi lapisan standar.
-   **LLaMA:** Menggunakan *Rotary Positional Embeddings* (RoPE), RMSNorm untuk stabilitas, dan fungsi aktivasi SwiGLU. Penskalaan LLaMA lebih berfokus pada efisiensi parameter.

![Figure 9: Perbandingan Arsitektur LLAMA dan GPT-3 (decoder-only).](https://lmvr.rantai.dev/images/L6A504iArDGBvUq8WSeO-bq3r3JDy49iXKJooDWGD-v1.png)

Industri telah mengadopsi GPT-4 untuk tugas umum, sementara LLaMA sangat populer karena sifatnya yang *open-source* dan efisien untuk dijalankan pada perangkat keras dengan sumber daya terbatas.

---

## 6.7. Mengevaluasi Model Generatif

Evaluasi membutuhkan kombinasi metrik otomatis dan penilaian manusia.

-   **BLEU:** Mengukur presisi tumpang tindih n-gram dengan referensi.
-   **ROUGE:** Mengukur recall, sering digunakan untuk peringkasan.
-   **Perplexity:** Mengukur seberapa yakin model memprediksi data uji.
    $$\text{Perplexity} = \exp \left( - \frac{1}{N} \sum \log P(x_i | x_{<i}) \right)$$

Berikut adalah implementasi metrik evaluasi di Python:

```python
from nltk.translate.bleu_score import sentence_bleu

def calculate_metrics(reference, candidate):
    ref_list = [reference.split()]
    cand_list = candidate.split()
    
    # BLEU
    score = sentence_bleu(ref_list, cand_list)
    print(f"BLEU Score: {score:.4f}")

calculate_metrics("the cat sat on the mat", "the cat is on the mat")
```

---

## 6.8. Penerapan dan Optimasi Model Generatif

Menyebarkan model besar membutuhkan teknik optimasi:

1.  **Kuantisasi:** Mengurangi presisi berat (misal FP32 ke INT8).
2.  **Distilasi:** Melatih model "murid" yang lebih kecil untuk meniru model "guru".
3.  **Pruning:** Menghapus koneksi saraf yang tidak penting (aktivasi rendah).

Implementasi kuantisasi sederhana di Python menggunakan PyTorch:

```python
import torch

# Contoh kuantisasi dinamis sederhana
def quantize_model(model):
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    return quantized_model
```

---

## 6.9. Kesimpulan

Bab 6 menyoroti kemajuan signifikan dalam model generatif, terutama melalui GPT dan varian-variannya. Memahami arsitektur dan proses pelatihannya memberikan wawasan tentang bagaimana AI dapat menghasilkan teks yang koheren.

### 6.9.1. Pembelajaran Lebih Lanjut dengan GenAI

- Jelaskan perbedaan distribusi $P(X)$ pada model generatif vs $P(Y|X)$ pada model diskriminatif.
- Bagaimana arsitektur GPT-3 menangani ketergantungan urutan jangka panjang?
- Diskusikan peran RLHF dalam menyelaraskan output model dengan nilai-nilai manusia.
- Bandingkan mekanisme atensi pada Stable Diffusion vs GPT.
- Analisis dampak *Scaling Laws* terhadap strategi pengembangan AI di masa depan.

### 6.9.2. Praktik Mandiri

- **Latihan 6.1:** Implementasikan generator teks autoregresif sederhana dan amati pengaruh *temperature* pada outputnya.
- **Latihan 6.2:** Lakukan *fine-tuning* pada model GPT-2 untuk gaya bahasa tertentu (misal: gaya bahasa hukum).
- **Latihan 6.3:** Buat tokenizer BPE kustom dan evaluasi dampaknya terhadap ukuran kosakata.
- **Latihan 6.4:** Implementasikan distilasi model sederhana di mana model Linear kecil belajar dari model yang lebih dalam.
- **Latihan 6.5:** Gunakan teknik *Top-K* dan *Top-P sampling* untuk mengontrol diversitas teks yang dihasilkan.