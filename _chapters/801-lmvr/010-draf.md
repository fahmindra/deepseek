---
slug: lmvr-1
title: lmvr-1
---
# Bab 1: Pengantar Model Bahasa Besar (LLM)

> **"Persimpangan antara model bahasa canggih dan bahasa pemrograman sistem seperti Rust membuka batas baru dalam menciptakan sistem AI yang efisien, aman, dan terukur." — Yoshua Bengio**

Bab 1 menawarkan pengantar komprehensif tentang Model Bahasa Besar (LLM) dan implementasinya menggunakan Rust. Dimulai dengan penyelaman mendalam ke dasar-dasar LLM, menelusuri evolusinya, dan mengeksplorasi komponen utama serta aplikasinya. Bab ini kemudian memperkenalkan fitur bahasa Rust yang kuat, terutama yang meningkatkan efisiensi dan keamanan implementasi LLM. Ini memberikan panduan rinci tentang membangun arsitektur LLM di Rust, lengkap dengan contoh praktis dan studi kasus. Terakhir, bab ini membahas tantangan dan pertimbangan etis yang melekat dalam penerapan LLM, menyoroti peran Rust dalam mengatasi masalah ini dan membuka jalan bagi perkembangan masa depan di bidang ini.

## 1.1. Dasar-dasar Model Bahasa Besar

Model Bahasa Besar (LLM) seperti robot ajaib yang telah membaca jutaan buku dan mempelajari segala macam hal tentang cara kita berbicara, menulis, dan bercerita. Ia memahami kata-kata, kalimat, dan bahkan makna di baliknya. Jadi, ketika Anda mengajukan pertanyaan, ia dapat menggunakan semua yang telah dipelajarinya untuk memberi Anda jawaban, atau jika Anda sedang bercerita dan buntu, ia dapat membantu Anda memikirkan bagian selanjutnya! Ini seperti teman yang sangat membantu yang tahu banyak kata dan selalu bisa memberikan tanggapan, baik Anda memintanya menjelaskan sesuatu, menceritakan lelucon, atau bahkan membantu mengerjakan pekerjaan rumah. LLM tidak berpikir seperti kita, tetapi ia sangat pandai berpura-pura karena ia tahu begitu banyak tentang bahasa!

![Figure 1: Analogi LLM yang dihasilkan oleh DALL-E.](https://lmvr.rantai.dev//images/L6A504iArDGBvUq8WSeO-TkCePeIW9O4aijlZqCNH-v1.webp)
**Gambar 1:** Analogi LLM yang dihasilkan oleh DALL-E.

Sebenarnya, kemampuan dasar model bahasa terletak pada kemampuannya untuk memahami dan menghasilkan bahasa manusia dengan belajar dari data teks dalam jumlah besar. Pada intinya, model bahasa memprediksi kata berikutnya dalam suatu urutan berdasarkan kata-kata sebelumnya, yang memungkinkannya menghasilkan kalimat yang koheren, menanggapi pertanyaan, dan melakukan tugas-tugas seperti pelengkapan teks, terjemahan, atau peringkasan. Dengan mempelajari pola dan struktur bahasa selama pelatihan, model dapat mengenali konteks, sintaksis, dan hubungan semantik. Hal ini memungkinkan model untuk terlibat dalam tugas-tugas bahasa alami seperti menjawab pertanyaan atau menulis paragraf yang mencerminkan pemahaman mendalam tentang data masukan. Seiring perkembangan model bahasa, mereka terus menangani tugas-tugas yang semakin kompleks dalam pemrosesan bahasa alami (NLP).

Dalam beberapa tahun terakhir, LLM telah menjadi landasan kecerdasan buatan modern, merevolusi cara mesin memahami dan menghasilkan bahasa manusia. Model-model ini mewakili lompatan signifikan dari teknik Pemrosesan Bahasa Alami (NLP) tradisional, yang memungkinkan mesin tidak hanya memproses teks tetapi juga menghasilkan bahasa yang koheren dan kaya secara kontekstual. Pada intinya, LLM dirancang untuk menangani dan memodelkan urutan bahasa pada skala yang belum pernah terjadi sebelumnya, mempelajari pola, semantik, dan bahkan penalaran dari data dalam jumlah besar.

Pengembangan LLM berakar kuat dalam evolusi historis NLP. Model-model awal seperti sistem berbasis aturan dan model $n$-gram memiliki kemampuan terbatas dalam memahami ketergantungan jarak jauh dalam bahasa. Pengenalan model berbasis pembelajaran mesin seperti penyematan kata (misalnya, Word2Vec) menandai dimulainya NLP berbasis data. Namun, transformasi sebenarnya datang dengan pengembangan arsitektur berbasis Transformer, seperti yang diperkenalkan oleh Vaswani et al. dalam makalah seminal *Attention is All You Need* (2017). Arsitektur ini meletakkan dasar bagi model-model canggih seperti GPT (Generative Pre-trained Transformer), BERT (Bidirectional Encoder Representations from Transformers), dan model skala besar lainnya yang unggul dalam berbagai tugas NLP, mulai dari pembuatan teks hingga terjemahan mesin. Kita akan membahas lebih banyak tentang arsitektur ini di bab-bab berikutnya.

![Figure 2: Arsitektur Transformer (Kredit d2l.ai)](https://lmvr.rantai.dev//images/L6A504iArDGBvUq8WSeO-BVRsatHqIfo0oG2Gd6YO-v1.svg)
**Gambar 2:** Arsitektur Transformer (Kredit d2l.ai)

Secara matematis, LLM mengandalkan beberapa komponen inti yang membedakannya dari model NLP tradisional. Salah satu aspek mendasar dari model ini adalah tokenisasi, di mana teks dipecah menjadi unit-unit yang lebih kecil, seringkali berupa subkata atau karakter, yang dikenal sebagai token. Setiap token kemudian direpresentasikan sebagai vektor dalam ruang vektor kontinu menggunakan penyematan (embeddings). Vektor penyematan dari sebuah token dipelajari selama proses pelatihan dan menangkap informasi semantik serta sintaksis tentang kata tersebut.

Mekanisme atensi (attention mechanism) adalah inovasi kritis lainnya yang menggerakkan LLM. Secara formal, diberikan urutan token masukan $\mathbf{x} = [x_1, x_2, \dots, x_n]$, mekanisme atensi menghitung jumlah tertimbang dari representasi token berdasarkan relevansinya dengan token tertentu. Relevansi ini dikuantifikasi menggunakan matriks query ($Q$), key ($K$), dan value ($V$), di mana skor atensi dihitung sebagai:

$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V $$

Di sini, $d_k$ adalah dimensi dari vektor key, dan fungsi softmax memastikan bahwa skor atensi dinormalisasi di semua token. Mekanisme ini memungkinkan model untuk fokus pada bagian relevan dari urutan masukan saat membuat prediksi, memungkinkannya menangani ketergantungan jarak jauh secara efektif, yang merupakan batasan signifikan dalam model-model awal seperti RNN dan LSTM.

Panjang urutan adalah faktor lain yang membedakan LLM. Model tradisional kesulitan menangkap konteks di luar jendela token tertentu. Namun, arsitektur berbasis Transformer, melalui mekanisme self-attention, memungkinkan LLM untuk memproses urutan dengan panjang yang cukup besar tanpa mengalami masalah gradien hilang (vanishing gradient) atau kehilangan konteks. Kemampuan ini memungkinkan model seperti GPT-3 atau GPT-4 untuk menghasilkan potongan teks koheren yang panjang atau meringkas dokumen besar secara akurat.

Paradigma pra-pelatihan (pre-training) dan penyetelan halus (fine-tuning) sangat penting untuk mengembangkan LLM yang efektif. Pra-pelatihan biasanya dilakukan pada dataset tak berlabel yang sangat besar di mana model mempelajari pola bahasa umum, semantik, dan tata bahasa. Secara formal, ini sering dicapai menggunakan tujuan seperti pemodelan bahasa bertopeng (seperti pada BERT) atau pemodelan bahasa kausal (seperti pada GPT). Selama pra-pelatihan, model meminimalkan fungsi kerugian seperti log-likelihood negatif:

$$ L = - \sum_{i=1}^{n} \log P(x_i | x_{1}, x_{2}, \dots, x_{i-1}; \theta), $$

di mana $P(x_i | x_{1}, x_{2}, \dots, x_{i-1}; \theta)$ mewakili probabilitas memprediksi token $x_i$ dengan mempertimbangkan token sebelumnya dan parameter model $\theta$. Setelah pra-pelatihan selesai, model menjalani penyetelan halus pada tugas-tugas khusus domain menggunakan dataset berlabel, yang mengadaptasi model bahasa umum ke aplikasi spesifik seperti klasifikasi teks, pengenalan entitas bernama, atau menjawab pertanyaan.

Aplikasi LLM mencakup berbagai domain dan industri. Dalam pembuatan teks, model seperti GPT-4 dapat membuat teks yang mirip manusia, yang memiliki aplikasi dalam pembuatan konten, sistem dialog, dan penulisan kreatif. Terjemahan mesin juga mendapat manfaat signifikan dari LLM, dengan sistem seperti Google Translate menggunakan model berbasis transformer untuk memberikan terjemahan yang lebih akurat yang lebih baik dalam menangkap nuansa bahasa yang berbeda. Aplikasi kritis lainnya adalah peringkasan teks, di mana LLM memadatkan volume teks yang besar menjadi ringkasan yang lebih pendek dan bermakna, yang sangat berguna di bidang-bidang seperti analisis dokumen hukum, penelitian akademik, dan pelaporan berita.

Di industri, LLM mendorong inovasi di bidang-bidang seperti layanan kesehatan, di mana model sedang disetel halus untuk membantu dalam dokumentasi klinis, interaksi pasien, dan penemuan obat. Layanan keuangan telah memanfaatkan LLM untuk menganalisis sentimen pasar, mengotomatiskan pelaporan, dan mendeteksi aktivitas penipuan. Selain itu, dalam layanan pelanggan, LLM menggerakkan chatbot canggih dan asisten virtual yang mampu memberikan interaksi yang dipersonalisasi dan sadar konteks dalam skala besar.

![Figure 3: Berbagai aplikasi LLM di banyak industri.](https://lmvr.rantai.dev//images/L6A504iArDGBvUq8WSeO-CLIPZOjVBqm3Fi9uca8H-v1.webp)
**Gambar 3:** Berbagai aplikasi LLM di banyak industri.

Dalam hal tren, skala LLM terus tumbuh, dengan model multibahasa dan LLM khusus domain muncul sebagai bidang minat utama. Model seperti GPT-4 telah menunjukkan kemampuan yang kuat di berbagai bahasa, mendorong penelitian tentang pembelajaran transfer lintas bahasa dan memperluas jangkauan aplikasi LLM. Selain itu, LLM khusus domain, seperti yang berfokus pada teks biomedis (misalnya, PubMedBERT), sedang dilatih untuk memberikan kinerja unggul di bidang khusus.

Untuk memahami sepenuhnya konsep dalam Large Language Model via Rust (LMVR), pembaca disarankan untuk mengeksplorasi buku Deep Learning via Rust (DLVR) terlebih dahulu, karena model bahasa besar (LLM) dibangun di atas prinsip-prinsip inti deep learning. DLVR mencakup topik-topik mendasar seperti jaringan saraf, propagasi balik, dan teknik optimasi, yang sangat penting untuk memahami cara kerja LLM. Selain itu, topik lanjutan seperti arsitektur transformer dan mekanisme atensi, yang merupakan pusat dari LLM, dibahas secara mendalam di DLVR. Dengan mempelajari kedua buku tersebut, pembaca akan memperoleh pemahaman komprehensif tentang deep learning dan lebih siap untuk mengimplementasikan serta mengoptimalkan LLM menggunakan Rust.

Hugging Face adalah platform terkemuka di komunitas AI yang menyediakan akses ke koleksi besar model pra-terlatih, dataset, dan alat untuk tugas pemrosesan bahasa alami (NLP). Ini menawarkan antarmuka yang ramah pengguna dan pustaka yang luas, termasuk model seperti GPT, BERT, dan T5, yang merupakan kunci dalam memahami dan bekerja dengan Model Bahasa Besar (LLM). Bagi insinyur AI, Hugging Face mempermudah eksperimen dengan model-model tercanggih dengan menyediakan integrasi yang mudah melalui pustaka `transformers` mereka, memungkinkan mereka melakukan tugas-tugas seperti pembuatan teks, analisis sentimen, dan terjemahan tanpa perlu membangun model dari nol. Dengan memanfaatkan Hugging Face, insinyur dapat mempercepat pembelajaran mereka, bereksperimen dengan teknologi mutakhir, dan memperdalam pemahaman mereka tentang arsitektur serta aplikasi LLM di berbagai domain.

![Figure 4: Komunitas AI HuggingFace untuk model LLM pra-terlatih.](https://lmvr.rantai.dev//images/L6A504iArDGBvUq8WSeO-XQ871m5X5GyxZ5zdi1wI-v1.png)
**Gambar 4:** Komunitas AI HuggingFace untuk model LLM pra-terlatih.

Open LLM Leaderboard Hugging Face memeringkat model bahasa besar berdasarkan kinerjanya di berbagai tugas NLP. Ini mengevaluasi model pada beberapa tolok ukur, seperti pembuatan teks, menjawab pertanyaan, peringkasan, dan tugas-tugas lain yang umum digunakan dalam penelitian dan pengembangan AI. Papan peringkat tersebut memberikan transparansi dengan menunjukkan kekuatan dan kelemahan masing-masing model, menawarkan wawasan tentang kemampuan mereka di bidang-bidang seperti akurasi, kecepatan, dan efisiensi. Platform terbuka dan kolaboratif ini membantu insinyur dan peneliti AI membandingkan model, memilih yang berkinerja terbaik untuk kasus penggunaan mereka, dan tetap mendapatkan informasi terbaru tentang kemajuan terbaru dalam pengembangan LLM, mendorong lingkungan kompetitif dan inovatif untuk peningkatan model.

![Figure 5: HuggingFace Open LLM Leaderboard.](https://lmvr.rantai.dev//images/L6A504iArDGBvUq8WSeO-ZMPyJiHW3c4fCeQXILIk-v1.png)
**Gambar 5:** HuggingFace Open LLM Leaderboard.

Sebagai contoh praktis, kita akan mendemonstrasikan cara melakukan beberapa tugas NLP menggunakan LLM yang tersedia untuk publik dari HuggingFace. Kita akan menggunakan library Python `transformers` untuk melakukan terjemahan mesin. Tugas-tugas ini menyoroti kekuatan dan keserbagunaan LLM dalam menangani berbagai kebutuhan pemrosesan bahasa alami.

### Contoh Python: Terjemahan Mesin dengan Transformers

```python
from transformers import pipeline

# Inisialisasi pipeline terjemahan
# Secara default menggunakan model MarianMT untuk terjemahan
translator = pipeline("translation_en_to_es", model="Helsinki-NLP/opus-mt-en-es")

def main():
    input_text = "This is a sentence to be translated"
    
    # Melakukan terjemahan
    output = translator(input_text)
    
    # Mencetak hasil
    for translation in output:
        print(translation['translation_text'])

if __name__ == "__main__":
    main()
```

Kode ini menggunakan library `transformers` untuk membuat model terjemahan mesin yang dapat menerjemahkan teks dari bahasa Inggris ke bahasa Spanyol. Pipeline `translation_en_to_es` diinisialisasi dengan model yang telah ditentukan. Model kemudian digunakan untuk menerjemahkan kalimat masukan yang diberikan ("This is a sentence to be translated").

Sebagai penutup bagian ini, mari kita lihat implementasi model sederhana yang mirip GPT (Generative Pre-trained Transformer) untuk pemodelan bahasa tingkat karakter menggunakan PyTorch. Kode berikut mengimplementasikan versi sederhana dari arsitektur mirip GPT. Model ini mencakup komponen kunci seperti penyematan token (token embeddings), mekanisme self-attention, lapisan feed-forward, dan koneksi residual. Model ini dirancang untuk menghasilkan prediksi dalam tugas pemodelan bahasa tingkat karakter, di mana ia memproses urutan karakter dari sebuah kosakata dan belajar memprediksi karakter berikutnya dalam sebuah urutan.

### Contoh Python: Arsitektur GPT Sederhana (Tingkat Karakter)

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt

# Fungsi untuk menghasilkan data sintetis
def generate_synthetic_data(vocab, seq_len, num_samples):
    inputs = []
    targets = []
    
    for _ in range(num_samples):
        seq_indices = [vocab.index(random.choice(vocab)) for _ in range(seq_len)]
        inputs.append(torch.tensor(seq_indices[:-1], dtype=torch.long))
        targets.append(torch.tensor(seq_indices[1:], dtype=torch.long))
    
    return inputs, targets

class AdvancedGPT(nn.Module):
    def __init__(self, vocab_size, hidden_dim):
        super(AdvancedGPT, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.linear_q = nn.Linear(hidden_dim, hidden_dim)
        self.linear_k = nn.Linear(hidden_dim, hidden_dim)
        self.linear_v = nn.Linear(hidden_dim, hidden_dim)
        self.linear_out = nn.Linear(hidden_dim, hidden_dim)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        self.layer_norm_1 = nn.LayerNorm(hidden_dim)
        self.layer_norm_2 = nn.LayerNorm(hidden_dim)
        self.output_linear = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        # x shape: (seq_len)
        embeddings = self.token_embedding(x) # (seq_len, hidden_dim)
        
        q = self.linear_q(embeddings)
        k = self.linear_k(embeddings)
        v = self.linear_v(embeddings)
        
        # Self-attention sederhana
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (d_k ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attention_output = torch.matmul(attn_weights, v)
        
        attention_output = self.linear_out(attention_output)
        normed_attention = self.layer_norm_1(embeddings + attention_output)
        
        ff_output = self.feed_forward(normed_attention)
        normed_output = self.layer_norm_2(normed_attention + ff_output)
        
        logits = self.output_linear(normed_output)
        return logits

def evaluate(model, inputs, targets):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in zip(inputs, targets):
            logits = model(x)
            predictions = torch.argmax(logits, dim=-1)
            correct += (predictions == y).sum().item()
            total += y.size(0)
    return correct / total

def main():
    vocab = list("abcdefghijklmnopqrstuvwxyz ")
    vocab_size = len(vocab)
    seq_len = 5
    num_samples = 1000
    hidden_dim = 128
    
    inputs, targets = generate_synthetic_data(vocab, seq_len, num_samples)
    model = AdvancedGPT(vocab_size, hidden_dim)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    losses = []
    
    for epoch in range(1, 501):
        total_loss = 0
        model.train()
        for x, y in zip(inputs, targets):
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / num_samples
        losses.append(avg_loss)
        if epoch % 50 == 0:
            print(f"Epoch: {epoch}, Loss: {avg_loss:.4}")

    # Plot loss
    plt.plot(losses)
    plt.title("Training Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("loss_plot.png")
    
    accuracy = evaluate(model, inputs, targets)
    print(f"Evaluation accuracy: {accuracy:.4}")

if __name__ == "__main__":
    main()
```

Model dimulai dengan menyematkan token masukan, diikuti dengan menerapkan self-attention, yang memungkinkannya fokus pada bagian relevan dari urutan masukan. Mekanisme self-attention menghitung atensi dot-product berskala, mentransformasi query, key, dan value melalui lapisan linier untuk menangkap hubungan kontekstual dalam masukan. Setelah atensi, penyematan melewati jaringan feed-forward dengan koneksi residual dan normalisasi lapisan untuk menstabilkan pelatihan dan meningkatkan kinerja. Output akhir dihasilkan melalui lapisan linier, menghasilkan logit yang mewakili probabilitas token yang diprediksi.

![Figure 6: Plot loss dari model bahasa tingkat karakter menggunakan arsitektur mirip GPT-2.](https://lmvr.rantai.dev//images/L6A504iArDGBvUq8WSeO-o4MwjDyZXabb4UALaByg-v1.png)
**Gambar 6:** Plot loss dari model bahasa tingkat karakter menggunakan arsitektur mirip GPT-2.

Sebagai kesimpulan, Model Bahasa Besar telah membentuk kembali lanskap kecerdasan buatan, memungkinkan mesin untuk melakukan tugas-bahasa yang kompleks dengan akurasi yang luar biasa. Dari arsitektur Transformer dan mekanisme atensi hingga pra-pelatihan dan penyetelan halus, landasan matematis LLM memungkinkan kinerja mereka yang tak tertandingi.

## 1.2. Fitur Bahasa Rust untuk LLM

Dalam pengembangan dan penerapan Model Bahasa Besar (LLM), manajemen memori yang efisien, konkurensi, dan optimalisasi kinerja adalah faktor penting, terutama saat menangani data dan komputasi skala besar. Rust, bahasa pemrograman sistem yang dikenal dengan keamanan memori dan fitur konkurensinya, memberikan keunggulan unik dalam konteks implementasi LLM. Berbeda dengan bahasa pemrograman tradisional seperti C++ atau Python, model kepemilikan (ownership) Rust memastikan bahwa data ditangani dengan aman tanpa memerlukan pengumpul sampah (garbage collector). Hal ini sangat relevan untuk LLM, di mana model seringkali besar dan haus sumber daya, dan mengelola memori secara efisien dapat berdampak signifikan pada kinerja.

![Figure 7: Rust menawarkan fitur luar biasa untuk pengembangan dan penyajian LLM.](https://lmvr.rantai.dev//images/L6A504iArDGBvUq8WSeO-GIPGgO7vu23mMxp4HGJP-v1.png)
**Gambar 7:** Rust menawarkan fitur luar biasa untuk pengembangan dan penyajian LLM.

Inti dari pendekatan Rust terhadap manajemen memori adalah sistem kepemilikan dan peminjaman (borrowing), yang menjamin bahwa data dimiliki oleh variabel tunggal atau dipinjam dengan cara yang memastikan akses aman. Desain ini mencegah masalah seperti dangling pointer atau kebocoran memori, yang umum terjadi dalam bahasa tanpa mekanisme keamanan memori bawaan. Untuk implementasi LLM, di mana tensor besar dan struktur data yang kompleks dimanipulasi, kontrol ketat Rust atas memori dapat mencegah bug umum dan meningkatkan kinerja runtime.

Saat membangun LLM, operasi tensor dan komputasi matriks sangat penting. Rust menyediakan pustaka yang kuat seperti `tch-rs` (binding untuk PyTorch) dan `ndarray`. Manipulasi tensor melibatkan operasi seperti perkalian matriks:

$$C_{ij} = \sum_{k} A_{ik} B_{kj}$$

Dukungan Rust untuk konkurensi melalui fitur-fitur seperti threads, async/await, dan crate `Rayon` untuk paralelisme memungkinkan distribusi operasi tensor di berbagai inti CPU atau GPU. Keuntungan lain dari Rust untuk LLM terletak pada teknik manajemen memorinya, yang memungkinkan pengembang untuk mengoptimalkan alokasi dan dealokasi sumber daya secara eksplisit. Berbeda dengan bahasa seperti Python, di mana manajemen memori sering ditangani secara otomatis tetapi dapat menyebabkan siklus pengumpulan sampah yang tidak terduga, Rust memungkinkan penanganan memori deterministik.

Berikut adalah contoh kode Python yang mendemonstrasikan penggunaan model BERT pra-terlatih untuk tugas masked language modeling (memprediksi kata yang hilang).

### Contoh Python: Masked Language Modeling dengan BERT

```python
from transformers import BertTokenizer, BertForMaskedLM
import torch

def main():
    # Load model dan tokenizer
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForMaskedLM.from_pretrained(model_name)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Teks masukan dengan token [MASK]
    input_text = "Rust is a [MASK] programming language."
    
    # Tokenisasi
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)
        prediction_scores = outputs.logits
    
    # Mencari indeks [MASK]
    mask_token_index = torch.where(inputs.input_ids == tokenizer.mask_token_id)[1]
    
    # Mengambil token yang diprediksi
    predicted_index = torch.argmax(prediction_scores[0, mask_token_index]).item()
    predicted_token = tokenizer.decode([predicted_index])
    
    print(f"Predicted token: {predicted_token}")

if __name__ == "__main__":
    main()
```

Kode di atas menyiapkan model BERT dan tokenizer, memuat bobot pra-terlatih dari repositori Hugging Face. Teks masukan yang berisi token bertopeng (`[MASK]`) ditokenisasi dan diumpankan ke model BERT. Model memprediksi kata yang paling mungkin untuk posisi yang bertopeng tersebut.

Dalam hal komputasi performa tinggi (HPC), integrasi Rust dengan kerangka kerja HPC memungkinkan pelatihan model yang lebih cepat, pemanfaatan memori yang lebih baik, dan kemampuan untuk menangani model yang lebih besar tanpa hambatan kinerja. Tren terbaru menunjukkan bahwa seiring bertambahnya skala LLM—seperti GPT-4 dan LLaMA—ada kebutuhan yang lebih besar untuk pengoptimalan tingkat sistem. Penekanan Rust pada abstraksi tanpa biaya (zero-cost abstractions) memposisikannya sebagai pilihan menarik untuk membangun aplikasi LLM yang terukur dan tangguh.

## 1.3. Mengimplementasikan Arsitektur LLM di Rust

Mengimplementasikan arsitektur Model Bahasa Besar (LLM) memerlukan pemahaman mendalam tentang konsep pembelajaran mesin dan fitur tingkat sistem yang unik. Membangun LLM melibatkan perakitan komponen kunci, termasuk tokenizer, lapisan penyematan (embedding), dan blok Transformer.

Inti dari setiap LLM adalah tokenizer, yang memecah teks masukan menjadi urutan token. Metode tokenisasi umum melibatkan pembagian teks menjadi unit subkata, seperti dalam Byte-Pair Encoding (BPE). Setelah masukan ditokenisasi, ia melewati lapisan penyematan, di mana setiap token diberi representasi dimensi tinggi. Secara matematis, diberikan token $t$, vektor penyematannya $e_t$ adalah sebuah titik dalam ruang $d$-dimensi.

Blok Transformer terdiri dari beberapa lapisan, terutama mekanisme multi-head self-attention dan jaringan saraf feed-forward. Dalam mekanisme self-attention, model memperhatikan berbagai bagian dari urutan masukan dengan menghitung skor atensi. Kemampuan pemrosesan paralel Rust dapat dimanfaatkan untuk pelatihan model dan inferensi, memungkinkan penggunaan sumber daya perangkat keras secara efisien. Crate `Rayon` memungkinkan paralelisme data, di mana batch data dapat diproses secara bersamaan.

Kode berikut mendemonstrasikan penggunaan Python dan pustaka `sentence-transformers` untuk menghasilkan embedding kalimat dan menghitung kesamaan (similarity) antar kalimat.

### Contoh Python: Kesamaan Kalimat Menggunakan BERT

```python
from sentence_transformers import SentenceTransformer, util
import torch

def main():
    # Load model pra-terlatih (MiniLM efisien dan cepat)
    model = SentenceTransformer('all-MiniLM-L6-v2')

    sentences = [
        "The cat sits outside",
        "A man is playing guitar",
        "I love pasta",
        "The new movie is awesome",
        "The cat plays in the garden",
        "A woman watches TV",
        "The new movie is so great",
        "Do you like pizza?",
    ]

    # Menghasilkan embedding
    embeddings = model.encode(sentences, convert_to_tensor=True)

    # Menghitung kesamaan kosinus antar semua pasangan
    cosine_scores = util.cos_sim(embeddings, embeddings)

    # Mencari pasangan dengan kesamaan tertinggi (selain dirinya sendiri)
    pairs = []
    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            pairs.append({'index': [i, j], 'score': cosine_scores[i][j].item()})

    # Urutkan berdasarkan skor tertinggi
    pairs = sorted(pairs, key=lambda x: x['score'], reverse=True)

    print("Top 5 pasangan kalimat yang paling mirip:")
    for pair in pairs[:5]:
        i, j = pair['index']
        print(f"Score: {pair['score']:.2f} | '{sentences[i]}' VS '{sentences[j]}'")

if __name__ == "__main__":
    main()
```

Kode ini memuat model BERT dari repositori Hugging Face, memproses sekumpulan kalimat masukan, dan menghasilkan embedding. Embedding ini kemudian dibandingkan menggunakan kesamaan kosinus untuk menentukan seberapa mirip kalimat satu sama lain.

Tren terbaru dalam penerapan LLM juga menekankan pentingnya inferensi yang efisien di lingkungan produksi. Karena LLM menjadi lebih besar, kebutuhan akan pipeline inferensi yang dioptimalkan menjadi kritis. Performa Rust, dikombinasikan dengan alat seperti ONNX (Open Neural Network Exchange), memungkinkan model untuk diekspor dan dioptimalkan untuk inferensi tanpa mengorbankan jaminan keamanan memori.

## 1.4. Tantangan dan Pertimbangan Etis

Seiring dengan terus berkembangnya LLM, pengembangan dan penerapannya menghadirkan tantangan teknis dan pertimbangan etis. Mengoptimalkan operasi tensor dan pemrosesan data di lingkungan dengan memori terbatas merupakan tantangan utama. Kompleksitas komputasi atensi meningkat secara kuadratik dengan panjang urutan.

![Figure 8: Kompleksitas pertimbangan etis LLM.](https://lmvr.rantai.dev//images/L6A504iArDGBvUq8WSeO-Ke80Y1AtmMlCIrwByvrt-v1.png)
**Gambar 8:** Kompleksitas pertimbangan etis LLM.

Selain tantangan teknis, implikasi etis dari penerapan LLM telah menjadi area diskusi kritis, terutama mengenai bias, keadilan, dan transparansi. LLM dilatih pada dataset luas yang dapat mengandung bias yang melekat—sosial, budaya, atau linguistik. Studi formal dalam keadilan pembelajaran mesin telah menyoroti bagaimana LLM dapat melanggengkan bahkan memperkuat bias ketika digunakan dalam aplikasi dunia nyata seperti perekrutan atau peradilan pidana.

Kekhawatiran utama dalam penerapan LLM adalah memastikan privasi dan keamanan data. Salah satu strateginya adalah privasi diferensial (differential privacy), kerangka kerja matematis yang dirancang untuk mencegah model menghafal atau membocorkan informasi sensitif dari data pelatihan. Optimalisasi kinerja tetap menjadi pertimbangan krusial, menggunakan teknik seperti pemangkasan model (pruning), kuantisasi, dan aproksimasi peringkat rendah. Secara matematis, pemangkasan melibatkan penghilangan parameter yang kurang kritis dari matriks bobot model, yang mengarah pada representasi matriks jarang yang mengurangi penggunaan memori dan komputasi.

Masa depan pengembangan LLM berkembang pesat. Dengan meningkatnya fokus pada AI hemat energi, kemampuan Rust untuk menghasilkan kode yang sangat optimal dengan overhead runtime minimal akan menjadi penting dalam mengembangkan LLM yang tidak hanya berperforma tinggi tetapi juga berkelanjutan.

## 1.5. Kesimpulan

Bab 1 telah meletakkan landasan yang kuat untuk memahami persimpangan Model Bahasa Besar dan pemrograman Rust. Dengan mengintegrasikan wawasan teoretis dengan implementasi praktis, bab ini membekali pembaca dengan pengetahuan dan alat yang dibutuhkan untuk membangun LLM yang efisien dan etis.

### 1.5.1. Pembelajaran Lebih Lanjut dengan GenAI

- Rinci kemajuan arsitektur dan teoretis yang telah mentransisikan NLP dari model statistik ke LLM seperti GPT dan BERT.
- Analisis cara kerja arsitektur Transformer, dengan fokus pada mekanisme self-attention.
- Diskusikan berbagai teknik tokenisasi yang digunakan dalam LLM, seperti BPE dan WordPiece.
- Jelajahi proses pembuatan dan pengoptimalan lapisan penyematan dalam LLM.
- Jelaskan fase pra-pelatihan dan penyetelan halus LLM secara mendalam.
- Selami tantangan teknis dalam menangani dataset skala besar di Rust.
- Analisis bagaimana kemampuan konkurensi Rust dapat dimanfaatkan untuk mempercepat pelatihan LLM.
- Jelajahi implikasi model kepemilikan dan peminjaman Rust pada pengembangan LLM.
- Berikan analisis mendalam tentang bagaimana library tensor dapat digunakan untuk operasi yang efisien.
- Diskusikan teknik lanjutan untuk mengoptimalkan kinerja LLM (pruning, kuantisasi).
- Pandu pengembangan aplikasi berbasis LLM yang sederhana namun komprehensif.
- Jelajahi implikasi etis dari penerapan LLM (bias, keadilan, transparansi).
- Berikan analisis mendalam tentang sumber bias dalam LLM dan dampaknya.
- Diskusikan pentingnya interpretabilitas dan eksplanabilitas model.
- Bandingkan proses penerapan LLM di Rust dengan bahasa pemrograman lain (Python, C++).
- Berikan panduan komprehensif untuk menyebarkan LLM berbasis Rust di lingkungan produksi.
- Jelajahi bagaimana model konkurensi Rust dapat digunakan untuk inferensi waktu nyata yang skalabel.
- Diskusikan potensi mengintegrasikan Rust dengan bahasa seperti Python untuk pengembangan LLM.
- Analisis tren masa depan dalam Model Bahasa Besar (multimodal, zero-shot learning).
- Lakukan analisis mendalam terhadap studi kasus dunia nyata yang melibatkan penerapan LLM.

### 1.5.2. Latihan Praktis

#### Latihan Mandiri 1.1: Tokenisasi Lanjut dan Optimalisasi Penyematan
**Tujuan:** Merancang dan mengimplementasikan pipeline tokenisasi dan penyematan khusus yang dapat memproses data teks multibahasa yang kompleks secara efisien.

#### Latihan Mandiri 1.2: Implementasi Transformer Khusus
**Tujuan:** Merancang dan melatih model Transformer miniatur dari awal menggunakan prinsip dasar, tanpa mengandalkan pustaka tingkat tinggi yang kaku.

#### Latihan Mandiri 1.3: Pengembangan Pipeline Pelatihan Terdistribusi
**Tujuan:** Membangun pipeline pelatihan terdistribusi untuk LLM, dengan fokus pada paralelisasi dan skalabilitas.

#### Latihan Mandiri 1.4: Mitigasi Bias dan Audit Keadilan
**Tujuan:** Mengimplementasikan dan mengevaluasi teknik tingkat lanjut untuk mitigasi bias dalam LLM, dengan fokus pada keadilan.

#### Latihan Mandiri 1.5: Penyebaran Model yang Aman
**Tujuan:** Merancang dan mengimplementasikan strategi penyebaran yang aman untuk LLM, dengan fokus pada perlindungan terhadap serangan adversarial dan privasi data.