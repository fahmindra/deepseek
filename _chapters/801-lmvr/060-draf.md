---
slug: lmvr-6
title: lmvr-6
---
# Bab 6: Model Generatif: GPT dan Seterusnya

> **"Model generatif seperti GPT mewakili loncatan besar ke depan dalam kemampuan kita untuk menyintesis teks yang menyerupai manusia, mencerminkan potensi mendalam AI untuk memahami dan menghasilkan bahasa yang kompleks." — Yann LeCun**

Bab 6 dari LMVR memberikan eksplorasi mendalam tentang model generatif, dengan fokus khusus pada GPT dan variannya. Bab ini dimulai dengan memperkenalkan dasar-dasar model generatif, membedakannya dari model diskriminatif, dan mendiskusikan aplikasinya dalam pemrosesan bahasa alami (NLP). Kemudian, bab ini menggali arsitektur GPT, menjelaskan sifat autoregresifnya, proses pelatihan, dan manfaat dari pemanfaatan model Transformer. Bab ini juga mencakup kemajuan dalam varian GPT, termasuk GPT-2 dan GPT-3, memeriksa dampak penskalaan terhadap kinerja dan pertimbangan etis dari penyebaran model besar. Bagian praktis mencakup implementasi model generatif dasar dalam Python, melatih model GPT, dan membandingkan berbagai varian GPT untuk memahami kekuatan dan keterbatasannya.

## 6.1. Pengantar Model Generatif

Model generatif memainkan peran krusial dalam pembelajaran mesin dengan memodelkan distribusi data yang mendasari, memungkinkan mereka untuk menghasilkan instansi baru yang menyerupai data tempat mereka dilatih. Ini berbeda dengan model diskriminatif, yang fokus pada membedakan antara berbagai kelas data. Model generatif bertujuan untuk memahami dan memproduksi distribusi data itu sendiri, yang membuatnya sangat berguna dalam tugas-tugas seperti pembuatan teks, peringkasan, dan terjemahan dalam domain pemrosesan bahasa alami (NLP). Sementara model diskriminatif seperti jaringan klasifikasi dirancang untuk menetapkan label atau kategori ke input, model generatif menangkap struktur dan pola dalam data untuk menciptakan output baru yang masuk akal.

![Gambar 1: Model Diskriminatif vs Generatif.](https://lmvr.rantai.dev//images/L6A504iArDGBvUq8WSeO-rCRQPzy7GRYxSIgPEhvZ-v1.png)
**Gambar 1:** Model Diskriminatif vs Generatif.

Dalam konteks NLP, model generatif telah menjadi fundamental dalam memproduksi teks berkualitas tinggi yang menyerupai buatan manusia. Aplikasinya berkisar dari menghasilkan kalimat yang koheren hingga meringkas informasi dalam jumlah besar menjadi teks yang ringkas dan bermakna, bahkan menerjemahkan bahasa dengan akurasi yang luar biasa. Model seperti GPT (Generative Pretrained Transformer) mencontohkan kemampuan model generatif untuk membuat teks yang mengalir dan sesuai konteks berdasarkan input atau perintah (prompt) yang diberikan. Model-model ini tidak sekadar menghasilkan urutan acak; sebaliknya, mereka memodelkan pola bahasa, tata bahasa, dan semantik, menghasilkan teks yang sering kali mencerminkan ekspresi manusia.

Inti dari model generatif terletak pada kemampuan mereka untuk mempelajari distribusi data. Ini melibatkan penangkapan distribusi probabilitas $P(X)$, di mana $X$ mewakili data, seperti kalimat dalam korpus. Untuk model autoregresif, ide kuncinya adalah memfaktorkan distribusi probabilitas gabungan dari urutan kata menjadi produk probabilitas bersyarat. Misalnya, dalam kasus pembuatan teks, probabilitas seluruh urutan dimodelkan sebagai:

$$P(X) = P(x_1)P(x_2 | x_1)P(x_3 | x_1, x_2) \dots P(x_n | x_1, x_2, \dots, x_{n-1}),$$

di mana $x_i$ mewakili setiap kata dalam urutan. Dengan mempelajari probabilitas bersyarat ini, model dapat menghasilkan teks baru kata demi kata, berdasarkan kata-kata yang dihasilkan sebelumnya.

Model generatif seperti GPT menggunakan metode autoregresif, di mana output pada setiap langkah dikondisikan pada token sebelumnya dalam urutan. Arsitektur Transformer di balik GPT telah merevolusi pemodelan generatif karena penanganannya yang efisien terhadap dependensi jarak jauh melalui mekanisme self-attention. Teknik pembelajaran mandiri (self-supervised learning) membentuk fondasi banyak model mutakhir saat ini.

Penerapan praktis model generatif secara tradisional difasilitasi oleh kerangka kerja seperti PyTorch dan TensorFlow. Program Python berikut menunjukkan penggunaan teknik prompt engineering dengan library `langchain` untuk mengoptimalkan respons dari model bahasa OpenAI.

### Contoh Python: Prompt Engineering dengan LangChain

```python
import os
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema import SystemMessage, HumanMessage, AIMessage

# Pastikan API Key sudah diset di environment variable
# os.environ["OPENAI_API_KEY"] = "your-api-key"

def main():
    # Inisialisasi model OpenAI
    llm = ChatOpenAI(model="gpt-4o-mini")

    # 1. Spesifikasi Peran (Role Specification)
    system_template = "Anda adalah penulis dokumentasi teknis kelas dunia dengan pengetahuan mendalam tentang pemrograman Rust."
    human_template = "{input}"
    
    chat_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template(human_template),
    ])

    messages = chat_prompt.format_messages(input="Jelaskan Rust dalam istilah sederhana.")
    response = llm.invoke(messages)
    print("Respons dengan Peran:\n", response.content)

    # 2. Contoh Few-Shot Learning
    few_shot_messages = [
        SystemMessage(content="Anda adalah programmer ahli. Jawab dengan nada ramah dan ringkas."),
        HumanMessage(content="Jelaskan perbedaan antara Rust dan C++."),
        AIMessage(content="Rust fokus pada keamanan memori tanpa garbage collector, sedangkan C++ memberikan kontrol lebih besar tetapi dengan risiko kesalahan memori yang lebih tinggi."),
        HumanMessage(content="Apa yang membuat Rust berbeda dari Python?")
    ]
    
    few_shot_response = llm.invoke(few_shot_messages)
    print("\nRespons Few-Shot:\n", few_shot_response.content)

    # 3. Konteks Sejarah (Conversation History)
    history = [
        SystemMessage(content="Anda adalah asisten yang membantu dan mengingat konteks."),
        HumanMessage(content="Nama saya adalah Luis."),
        AIMessage(content="Halo Luis, senang bertemu dengan Anda!"),
        HumanMessage(content="Siapa penulis buku 20.000 Mil di Bawah Laut?")
    ]
    
    history_response = llm.invoke(history)
    print("\nRespons Berbasis Riwayat:\n", history_response.content)

if __name__ == "__main__":
    main()
```

Sejak 2018, model GPT OpenAI telah mengalami peningkatan pesat dalam jumlah parameter. GPT asli memiliki 117 juta parameter, GPT-2 melonjak menjadi 1,5 miliar, dan GPT-3 meluas hingga 175 miliar parameter. Pertumbuhan ini didorong oleh kebutuhan model untuk menangkap pola bahasa yang semakin kompleks dan pengetahuan dunia.

![Gambar 2: Jumlah parameter model GPT.](https://lmvr.rantai.dev//images/L6A504iArDGBvUq8WSeO-1yyOzCHsIb0oYBXNfNtR-v1.webp)
**Gambar 2:** Jumlah parameter model GPT.

## 6.2. Arsitektur GPT

Arsitektur GPT (Generative Pre-trained Transformer) dibangun di atas arsitektur Transformer yang berfokus pada pembuatan teks autoregresif. GPT mengimplementasikan struktur transformer *decoder-only* di mana hanya atensi kausal (unidireksional) yang digunakan.

![Gambar 3: Ilustrasi arsitektur GPT-2.](https://lmvr.rantai.dev//images/L6A504iArDGBvUq8WSeO-u3gVcexpmTkzBQnrETnr-v1.png)
**Gambar 3:** Ilustrasi arsitektur GPT-2.

Proses pelatihan GPT melibatkan dua tahap utama: pra-pelatihan (pre-training) dan penyetelan halus (fine-tuning). Selama fase pra-pelatihan, model dilatih pada korpus teks besar menggunakan pembelajaran tanpa pengawasan (unsupervised). Fungsi kerugian (loss function) yang digunakan adalah log-likelihood negatif:

$$\mathcal{L} = - \sum_{i=1}^{n} \log P(x_i | x_1, x_2, \dots, x_{i-1})$$

Berikut adalah implementasi model mirip GPT untuk pembuatan teks tingkat karakter menggunakan PyTorch, yang dilatih pada dataset Tiny Shakespeare.

### Contoh Python: GPT Sederhana (Tingkat Karakter)

```python
import torch
import torch.nn as nn
from torch.nn import functional as F

# Hyperparameters
batch_size = 64
block_size = 128
max_iters = 5000
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

class Head(nn.Module):
    """ satu head self-attention """
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    """ beberapa head self-attention secara paralel """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """ lapisan linier sederhana diikuti oleh non-linearitas """
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: komunikasi diikuti komputasi """
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :] # (B, C)
            probs = F.softmax(logits, dim=-1) # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
```

Dalam beberapa tahun terakhir, aplikasi industri model GPT telah menjamur, terutama di bidang layanan pelanggan, pembuatan konten, dan asisten pengkodean seperti GitHub Copilot.

![Gambar 4: Pertumbuhan blok transformer dalam berbagai model GPT-2.](https://lmvr.rantai.dev//images/L6A504iArDGBvUq8WSeO-oHvUxZ4KXIC2oSvMZAtG-v1.png)
**Gambar 4:** Pertumbuhan blok transformer dalam berbagai model GPT-2.

## 6.3. Varian dan Ekstensi GPT

Penskalaan arsitektur model dan data pelatihan telah menjadi faktor sentral kesuksesan varian GPT. Pensksalaan ini sejalan dengan "Hukum Moore" eksponensial dalam NLP.

![Gambar 5: Hukum Moore dalam NLP.](https://lmvr.rantai.dev//images/L6A504iArDGBvUq8WSeO-I8HKwFaE7jawZvvgfEwc-v1.png)
**Gambar 5:** Hukum Moore dalam NLP.

Konsep hukum penskalaan (scaling laws) memainkan peran krusial dalam memahami mengapa model yang lebih besar berkinerja lebih baik secara signifikan. Secara formal:

$$L(N, D, C) = aN^{-\alpha} + bD^{-\beta} + cC^{-\gamma},$$

di mana $L$ adalah fungsi kerugian, $N$ jumlah parameter, $D$ ukuran dataset, dan $C$ komputasi.

![Gambar 6: Perbandingan ukuran model dari varian GPT.](https://lmvr.rantai.dev//images/L6A504iArDGBvUq8WSeO-mqVgJ3qL9pcY9rz0DS6w-v1.png)
**Gambar 6:** Perbandingan ukuran model dari varian GPT.

### Contoh Python: Menggunakan GPT-2 untuk Pembuatan Teks

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

def main():
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    prompt = "Once upon a time"
    inputs = tokenizer.encode(prompt, return_tensors="pt")

    # Generate text
    output = model.generate(
        inputs, 
        max_length=50, 
        num_return_sequences=1, 
        no_repeat_ngram_size=2,
        early_stopping=True
    )

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print("Teks yang dihasilkan:\n", generated_text)

if __name__ == "__main__":
    main()
```

## 6.4. Pelatihan, Penyetelan Halus, dan Adaptasi Tugas Model GPT

OpenAI menggunakan pendekatan Reinforcement Learning with Human Feedback (RLHF) yang terstruktur. Proses ini melibatkan tiga tahap kritis: Penyetelan Halus Terawasi (Supervised Fine-Tuning/SFT), pelatihan Model Hadiah (Reward Model), dan optimasi kebijakan menggunakan Proximal Policy Optimization (PPO).

![Gambar 7: Cara OpenAI melatih model GPT 3.5.](https://lmvr.rantai.dev//images/L6A504iArDGBvUq8WSeO-J67zNjyqKxM6ioNt2Pf6-v1.png)
**Gambar 7:** Cara OpenAI melatih model GPT 3.5.

Fungsi kerugian SFT:
$$ \mathcal{L}_{\text{SFT}}(\theta) = - \sum_{(X, Y) \in \text{Dataset}} \log P_\theta(Y | X) $$

Fungsi kerugian Model Hadiah:
$$\mathcal{L}_{\text{Reward}}(\phi) = \mathbb{E}_{(Y_1, Y_2) \sim D} \left[ \log \sigma\left(R_\phi(Y_1) - R_\phi(Y_2)\right) \right]$$

### Contoh Python: Simulasi Langkah SFT dan PPO sederhana

```python
import torch
import torch.nn.functional as F

# Simulasi SFT
def supervised_fine_tuning(model, inputs, targets, optimizer):
    model.train()
    optimizer.zero_grad()
    logits = model(inputs)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
    loss.backward()
    optimizer.step()
    return loss.item()

# Simulasi PPO Update (Konsep)
def ppo_update(policy_model, states, actions, old_log_probs, advantages, epsilon=0.2):
    logits = policy_model(states)
    new_log_probs = F.log_softmax(logits, dim=-1).gather(1, actions)
    ratio = torch.exp(new_log_probs - old_log_probs)
    
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon) * advantages
    
    loss = -torch.min(surr1, surr2).mean()
    return loss
```

## 6.5. Teknik Generatif Lanjutan di Luar GPT

Selain model autoregresif, teknik tingkat lanjut seperti Variational Autoencoders (VAEs), Generative Adversarial Networks (GANs), dan Model Difusi telah diperkenalkan.

![Gambar 8: Perbandingan antara model GAN, VAE, dan Difusi.](https://lmvr.rantai.dev//images/L6A504iArDGBvUq8WSeO-u5mui8fwOw66TBwzmAJD-v1.webp)
**Gambar 8:** Perbandingan antara model GAN, VAE, dan Difusi.

VAEs mengoptimalkan *evidence lower bound* (ELBO):
$$\mathcal{L}_{\text{VAE}} = \mathbb{E}_{q(z|x)} \left[ \log p(x|z) \right] - \text{KL}(q(z|x) || p(z))$$

GANs menggunakan minimax game:
$$\min_G \max_D \mathbb{E}_{x \sim p_{\text{data}}(x)} \left[ \log D(x) \right] + \mathbb{E}_{z \sim p_z(z)} \left[ \log (1 - D(G(z))) \right]$$

### Contoh Python: VAE Sederhana untuk MNIST

```python
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        # Encoder
        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20) # mu
        self.fc22 = nn.Linear(400, 20) # logvar
        # Decoder
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

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

## 6.6. Arsitektur GPT vs LLaMA

LLaMA (Large Language Model Meta AI) memperkenalkan beberapa penyempurnaan arsitektur seperti *rotary positional embeddings* (RoPE) dan normalisasi RMS. RoPE menerapkan matriks rotasi $R$ ke vektor query dan key:

$$Q' = Q \cdot R, \quad K' = K \cdot R$$

![Gambar 9: Perbandingan Arsitektur LLAMA dan GPT-3 (decoder-only).](https://lmvr.rantai.dev//images/L6A504iArDGBvUq8WSeO-bq3r3JDy49iXKJooDWGD-v1.png)
**Gambar 9:** Perbandingan Arsitektur LLAMA dan GPT-3 (decoder-only).

LLaMA mengoptimalkan efisiensi melalui parameter yang dikemas padat dan penggunaan dataset berkualitas tinggi yang lebih terarah.

### Contoh Python: Menggunakan LLaMA untuk Inferensi (via Hugging Face)

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def main():
    model_id = "meta-llama/Llama-2-7b-hf" # Memerlukan akses dari Meta
    # tokenizer = AutoTokenizer.from_pretrained(model_id)
    # model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")

    # prompt = "EDWARD: I wonder how our princely father 'scaped,"
    # inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    # output = model.generate(**inputs, max_new_tokens=100)
    # print(tokenizer.decode(output[0], skip_special_tokens=True))
    pass

if __name__ == "__main__":
    main()
```

## 6.7. Mengevaluasi Model Generatif

Mengevaluasi model generatif memerlukan kombinasi metrik otomatis dan evaluasi manusia. Metrik otomatis yang populer meliputi:

**BLEU (Bilingual Evaluation Understudy):**
$$\text{BLEU} = \text{BP} \cdot \exp \left( \sum_{n=1}^{N} w_n \log p_n \right)$$

**ROUGE (Recall-Oriented Understudy for Gisting Evaluation):** Fokus pada recall n-gram, sering digunakan untuk peringkasan.

**Perplexity:** Mencerminkan seberapa baik model memprediksi kata berikutnya:
$$\text{Perplexity} = \exp \left( - \frac{1}{N} \sum_{i=1}^{N} \log P(x_i | x_1, \dots, x_{i-1}) \right)$$

### Contoh Python: Menghitung BLEU dan ROUGE sederhana

```python
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer

def main():
    reference = "the cat sat on the mat"
    generated = "the cat is on the mat"

    # BLEU
    score = sentence_bleu([reference.split()], generated.split())
    print(f"BLEU Score: {score:.4f}")

    # ROUGE
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(reference, generated)
    print(f"ROUGE-L Score: {scores['rougeL'].fmeasure:.4f}")

if __name__ == "__main__":
    main()
```

## 6.8. Penyebaran dan Optimasi Model Generatif

Menyebarkan model besar memerlukan teknik optimasi seperti:

1.  **Kuantisasi (Quantization):** Mengurangi presisi bobot (misal dari FP32 ke INT8).
2.  **Distilasi Model (Model Distillation):** Melatih model "murid" yang lebih kecil untuk meniru model "guru" yang besar.
3.  **Pruning:** Menghapus neuron atau koneksi yang tidak perlu.

### Contoh Python: Kuantisasi Dinamis Sederhana di PyTorch

```python
import torch

def main():
    # Model linier sederhana
    model = torch.nn.Linear(128, 64)
    
    # Kuantisasi dinamis untuk lapisan linier
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    
    print("Model asli size (byte):", model.weight.nelement() * 4)
    print("Model terkuantisasi size (byte):", quantized_model.weight().nelement())

if __name__ == "__main__":
    main()
```

## 6.9. Kesimpulan

Bab 6 menyoroti kemajuan signifikan dalam model generatif, khususnya melalui GPT dan variannya. Memahami arsitektur, proses pelatihan, dan teknik optimasi sangat penting untuk membangun sistem AI yang mampu menghasilkan teks yang koheren dan relevan secara kontekstual.

### 6.9.1. Pembelajaran Lebih Lanjut dengan GenAI

- Jelaskan perbedaan mendasar antara model generatif dan diskriminatif.
- Deskripsikan arsitektur GPT dan sifat autoregresifnya.
- Diskusikan pentingnya pra-pelatihan skala besar pada GPT.
- Bandingkan tujuan pelatihan GPT dengan VAE dan GAN.
- Apa perbedaan arsitektur utama antara GPT-2 dan GPT-3?
- Jelajahi konsep pembelajaran mandiri (self-supervised learning).
- Diskusikan peran tokenisasi (BPE) dalam model GPT.
- Jelaskan konsep penyetelan halus (fine-tuning) model.
- Apa tantangan utama penyetelan halus pada data domain spesifik?
- Jelajahi konsep zero-shot dan few-shot learning.
- Diskusikan implikasi etis dari penyebaran model generatif besar.
- Bandingkan kualitas output GPT dengan teknik generatif lain.
- Apa metrik evaluasi utama (BLEU, ROUGE, Perplexity)?
- Jelajahi teknik generatif lanjutan seperti Model Difusi.
- Diskusikan tantangan skalabilitas di lingkungan produksi.
- Jelaskan proses distilasi model.
- Apa pertimbangan utama untuk aplikasi waktu nyata (latency vs accuracy)?
- Jelajahi konsep pembuatan teks terkontrol (attribute control).
- Diskusikan potensi transfer learning ke domain lain (coding, multimodal).
- Analisis dampak hukum penskalaan (scaling laws) pada kinerja model.

### 6.9.2. Latihan Praktis

#### Latihan Mandiri 6.1: Implementasi Pembuatan Teks Autoregresif
Tujuan: Memahami sifat autoregresif dengan melatih model GPT skala kecil untuk memprediksi karakter berikutnya.

#### Latihan Mandiri 6.2: Fine-Tuning GPT untuk Domain Spesifik
Tujuan: Melatih model GPT pada dataset khusus (misal: dokumen teknis) untuk meningkatkan relevansi teks.

#### Latihan Mandiri 6.3: Optimasi Tokenizer
Tujuan: Mengimplementasikan tokenizer kustom (BPE) dan mengevaluasi dampaknya terhadap memori.

#### Latihan Mandiri 6.4: Distilasi Model GPT
Tujuan: Melatih model yang lebih kecil untuk meniru output dari model yang lebih besar.

#### Latihan Mandiri 6.5: Pembuatan Teks Terkontrol
Tujuan: Memodifikasi proses sampling untuk mengarahkan output model ke sentimen tertentu.