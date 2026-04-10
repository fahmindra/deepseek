---
slug: lmvr-9
title: lmvr-9
---
# Bab 9: Membangun LLM Sederhana dari Nol Menggunakan Rust

> **"Membangun model pembelajaran mesin dari nol menawarkan wawasan yang tak ternilai ke dalam kerja fundamental AI, dan fokus Rust pada keamanan serta performa menjadikannya alat yang ampuh untuk mengembangkan sistem yang andal dan efisien." — Andrew Ng**

Bab 9 dari LMVR memberikan panduan komprehensif untuk membangun model bahasa sederhana (LLM) dari nol menggunakan Rust. Bab ini dimulai dengan pengenalan model bahasa, menyoroti evolusi dari metode statistik tradisional ke pendekatan deep learning modern. Bab ini kemudian membahas pengaturan lingkungan Rust, menekankan fitur keamanan dan performa bahasa tersebut, diikuti oleh bagian terperinci tentang pemrosesan awal data, tokenisasi, dan arsitektur model. Bab ini memandu proses pelatihan, termasuk teknik optimasi dan penyetelan hyperparameter, serta mengeksplorasi pentingnya evaluasi dan penyetelan halus untuk tugas-tugas tertentu. Akhirnya, bab ini membahas penyebaran LLM, menangani tantangan seperti skalabilitas dan latensi, dan diakhiri dengan pandangan ke arah masa depan serta tantangan dalam membangun LLM dengan Rust.

## 9.1. Pengantar Model Bahasa

Model bahasa adalah kerangka kerja probabilistik yang dirancang untuk menetapkan probabilitas pada urutan kata dalam suatu bahasa. Mereka memainkan peran fundamental dalam tugas pemrosesan bahasa alami (NLP) seperti terjemahan mesin, pengenalan ucapan, dan pembuatan teks. Dengan memodelkan kemungkinan urutan kata, model bahasa memungkinkan mesin untuk memahami dan menghasilkan bahasa manusia secara efektif.

Tujuan utama dari model bahasa adalah untuk menghitung probabilitas gabungan dari urutan kata $w_1, w_2, \dots, w_n$. Ini dapat dinyatakan menggunakan aturan rantai probabilitas:

$$ P(w_1, w_2, \dots, w_n) = \prod_{t=1}^{n} P(w_t | w_1, w_2, \dots, w_{t-1}). $$

Formulasi ini menguraikan probabilitas gabungan menjadi produk dari probabilitas bersyarat, di mana setiap istilah mewakili probabilitas sebuah kata berdasarkan semua kata sebelumnya dalam urutan tersebut.

Sifat Markov adalah konsep dasar dalam proses stokastik, yang menyatakan bahwa keadaan masa depan dari suatu proses hanya bergantung pada keadaan sekarang dan independen dari keadaan masa lalu. Secara matematis, untuk proses stokastik $\{X_t\}$, sifat Markov dinyatakan sebagai:

$$ P(X_{t+1} | X_t, X_{t-1}, \dots, X_1) = P(X_{t+1} | X_t). $$

Dalam konteks pemodelan bahasa, menerapkan sifat Markov secara langsung akan menyiratkan bahwa probabilitas kata berikutnya hanya bergantung pada kata saat ini, mengabaikan semua kata sebelumnya. Namun, bahasa alami menunjukkan ketergantungan yang seringkali membentang di beberapa kata. Untuk menangkap lebih banyak konteks, asumsi Markov tingkat tinggi dibuat, di mana probabilitas kata berikutnya bergantung pada sejumlah kata sebelumnya yang tetap, bukan hanya kata yang segera mendahuluinya.

Dalam pemrosesan bahasa alami (NLP), sifat Markov adalah gagasan bahwa kemungkinan kata di masa depan hanya bergantung pada kata saat ini, bukan pada kata-kata yang muncul sebelumnya. Bayangkan Anda memprediksi cuaca berdasarkan pola sederhana: "Berawan," "Hujan," dan "Cerah." Jika ramalan hanya bergantung pada cuaca hari ini, kita katakan itu memiliki sifat Markov. Misalnya, jika hari ini "Berawan," peluang besok menjadi "Hujan" mungkin tinggi, sementara peluang "Cerah" lebih rendah. Dalam model Markov urutan ke-1, cuaca besok hanya akan bergantung pada hari ini, jadi jika hari ini "Berawan," kita tidak akan mempertimbangkan cuaca kemarin lusa, bahkan jika hari itu juga "Berawan." Penyederhanaan ini membuat prediksi lebih mudah karena mengurangi konteks yang diperlukan hanya pada "keadaan" terbaru—dalam hal ini, cuaca hari ini. Demikian pula, dalam NLP, model Markov akan memprediksi kata berikutnya dalam sebuah kalimat hanya berdasarkan kata tepat sebelum itu, mengabaikan konteks masa lalu yang lebih jauh.

![Gambar 1](https://lmvr.rantai.dev//images/L6A504iArDGBvUq8WSeO-TJTAnihZzki42o8jtNIz-v1.webp)
**Gambar 1:** Ilustrasi sederhana model Markov dalam prediksi cuaca (Kredit: GeeksforGeeks).

Asumsi Markov urutan $(n-1)$ mengarah pada model n-gram, yang mendekati probabilitas bersyarat sebagai:

$$ P(w_t | w_1, w_2, \dots, w_{t-1}) \approx P(w_t | w_{t-(n-1)}, \dots, w_{t-1}). $$

Penyederhanaan ini mengurangi kompleksitas komputasi dengan membatasi riwayat pada $n-1$ kata sebelumnya. Misalnya, dalam model bigram (di mana $n = 2$), probabilitas sebuah kata hanya bergantung pada kata yang segera mendahuluinya:

$$ P(w_t | w_1, w_2, \dots, w_{t-1}) \approx P(w_t | w_{t-1}), $$

dan probabilitas urutannya menjadi:

$$ P(w_1, w_2, \dots, w_n) \approx \prod_{t=1}^{n} P(w_t | w_{t-1}). $$

Dalam model trigram (di mana $n = 3$), probabilitas bergantung pada dua kata sebelumnya:

$$ P(w_t | w_1, w_2, \dots, w_{t-1}) \approx P(w_t | w_{t-2}, w_{t-1}), $$

menghasilkan:

$$ P(w_1, w_2, \dots, w_n) \approx \prod_{t=1}^{n} P(w_t | w_{t-2}, w_{t-1}). $$

Probabilitas bersyarat dalam model n-gram biasanya diperkirakan dari korpus besar menggunakan Maximum Likelihood Estimation (MLE). Rumus estimasinya adalah:

$$ P(w_t | w_{t-(n-1)}, \dots, w_{t-1}) = \frac{\text{Count}(w_{t-(n-1)}, \dots, w_{t-1}, w_t)}{\text{Count}(w_{t-(n-1)}, \dots, w_{t-1})}, $$

di mana $\text{Count}(w_{t-(n-1)}, \dots, w_t)$ adalah berapa kali urutan tersebut muncul dalam korpus. Namun, pendekatan ini menghadapi tantangan seperti kelangkaan data, karena banyak n-gram yang mungkin tidak muncul dalam data pelatihan, dan dimensionalitas tinggi, karena jumlah parameter tumbuh secara eksponensial seiring dengan $n$ dan ukuran kosakata.

Untuk mengatasi kelangkaan data, teknik penghalusan (smoothing) menyesuaikan probabilitas yang diperkirakan untuk menetapkan sejumlah massa probabilitas ke n-gram yang tidak terlihat. Metode penghalusan umum termasuk penghalusan Add-One (Laplace), pendiskonan Good-Turing, dan penghalusan Kneser-Ney. Misalnya, penghalusan Add-One memodifikasi rumus MLE menjadi:

$$ P_{\text{Laplace}}(w_t | w_{t-(n-1)}, \dots, w_{t-1}) = \frac{\text{Count}(w_{t-(n-1)}, \dots, w_t) + 1}{\text{Count}(w_{t-(n-1)}, \dots, w_{t-1}) + V}, $$

di mana $V$ adalah ukuran kosakata.

Terlepas dari teknik-teknik ini, model n-gram memiliki keterbatasan yang signifikan. Mereka tidak dapat menangkap ketergantungan di luar $n-1$ kata, mengabaikan ketergantungan sintaksis dan semantik jarak jauh. Selain itu, model n-gram tingkat tinggi membutuhkan data dalam jumlah besar untuk memperkirakan probabilitas secara andal, dan menyimpan semua hitungan n-gram yang mungkin menjadi tidak praktis untuk $n$ dan kosakata yang besar. Keterbatasan ini telah memotivasi pengembangan model bahasa yang lebih canggih yang dapat menangkap ketergantungan yang lebih panjang.

Neural Network Language Models (NNLM) telah muncul untuk mengatasi keterbatasan model n-gram. NNLM menggunakan jaringan saraf untuk memodelkan probabilitas bersyarat dan dapat menangkap ketergantungan jarak jauh dengan menggunakan representasi kata yang berkelanjutan (embeddings) dan arsitektur yang mampu memproses urutan. Recurrent Neural Networks (RNN), misalnya, mempertahankan keadaan tersembunyi yang menangkap informasi dari semua langkah waktu sebelumnya. Formulasi matematisnya melibatkan pembaruan keadaan tersembunyi $\mathbf{h}_t$ menggunakan:

$$ \mathbf{h}_t = f(\mathbf{W}_h \mathbf{h}_{t-1} + \mathbf{W}_x \mathbf{x}_t + \mathbf{b}_h), $$

di mana $\mathbf{x}_t$ adalah embedding input untuk kata $w_t$, dan menghasilkan distribusi probabilitas output menggunakan:

$$ P(w_t | w_1, \dots, w_{t-1}) = \text{Softmax}(\mathbf{W}_o \mathbf{h}_t + \mathbf{b}_o). $$

![Gambar 2](https://lmvr.rantai.dev//images/L6A504iArDGBvUq8WSeO-PHGpZcFye1BypzX89VqY-v1.webp)
**Gambar 2:** Ilustrasi model bahasa sederhana menggunakan RNN.

Dalam model bahasa jaringan saraf berulang (RNN) sederhana, kita memproses setiap kata dalam urutan langkah demi langkah. Untuk setiap kata di posisi $t$, yang diwakili oleh $x^{(t)}$, kita pertama-tama memetakannya ke penyematan kata (word embedding) $e^{(t)} = E \cdot x^{(t)}$, di mana $E$ adalah matriks penyematan yang mengubah $x^{(t)}$ menjadi vektor padat yang menangkap makna kata. Penyematan ini dimasukkan ke dalam RNN, yang mempertahankan keadaan tersembunyi $h_t$ yang menangkap informasi kontekstual dari kata-kata sebelumnya dalam urutan tersebut. Keadaan tersembunyi $h^{(t)}$ dihitung berdasarkan penyematan kata saat ini $e^{(t)}$ dan keadaan tersembunyi sebelumnya $h^{(t-1)}$, menggunakan serangkaian bobot $W$ yang diterapkan berulang kali pada setiap langkah. Akhirnya, model mengeluarkan distribusi $y_t$ atas kemungkinan kata-kata berikutnya, memprediksi kelanjutan yang mungkin seperti "books" atau "laptop" setelah urutan input "para siswa membuka... mereka". Ide kunci dari penerapan bobot $W$ yang sama berulang kali di setiap langkah waktu adalah pusat dari RNN; ini memungkinkan model untuk menggeneralisasi pola yang dipelajari dari satu bagian urutan ke bagian lainnya, memberikan konsistensi dan memungkinkan model menangani panjang urutan yang bervariasi. Penerapan bobot berulang ini juga memungkinkan jaringan untuk menangkap ketergantungan sekuensial secara efektif, karena setiap $h^{(t)}$ mencerminkan konteks kumulatif hingga titik tersebut dalam urutan.

Arsitektur tingkat lanjut seperti Long Short-Term Memory (LSTM) dan Gated Recurrent Units (GRU) mengatasi masalah seperti gradien menghilang, memungkinkan pembelajaran ketergantungan jarak jauh yang lebih baik. Selain itu, pengenalan arsitektur Transformer, yang mengandalkan mekanisme self-attention daripada rekurensi, telah memajukan pemodelan bahasa secara signifikan. Mekanisme self-attention memungkinkan model untuk menimbang pengaruh posisi yang berbeda dalam urutan input, dengan atensi yang dihitung sebagai:

$$ \text{Attention}(Q, K, V) = \text{Softmax}\left( \frac{QK^\top}{\sqrt{d_k}} \right) V, $$

di mana $Q, K, V$ adalah matriks query, key, dan value yang diturunkan dari penyematan input, dan $d_k$ adalah dimensionalitas vektor key.

Teknik pretraining model bahasa, seperti Masked Language Modeling (MLM) yang digunakan dalam BERT dan pemodelan autoregresif yang digunakan dalam GPT, telah lebih lanjut meningkatkan kemampuan model bahasa. Model-model ini dilatih pada korpus besar untuk memprediksi kata-kata yang hilang atau kata berikutnya dalam suatu urutan, memungkinkan mereka untuk menangkap pola dan ketergantungan yang kompleks dalam bahasa.

![Gambar 3](https://lmvr.rantai.dev//images/L6A504iArDGBvUq8WSeO-cq4L0EFMhfN2hc3LPWge-v1.jpeg)
**Gambar 3:** Ilustrasi model BERT untuk model bahasa (Kredit: GeeeksforGeeks).

Evaluasi model bahasa sering kali melibatkan metrik seperti perplexity, yang mengukur seberapa baik model bahasa memprediksi sebuah sampel. Perplexity didefinisikan sebagai:

$$ \text{Perplexity}(P) = \exp\left( -\frac{1}{N} \sum_{t=1}^{N} \log P(w_t | w_1, \dots, w_{t-1}) \right), $$

dengan perplexity yang lebih rendah menunjukkan kinerja prediktif yang lebih baik. Entropi adalah metrik lain yang mengukur ketidakpastian dalam memprediksi kata berikutnya, didefinisikan sebagai:

$$ H(P) = -\sum_{w} P(w) \log P(w). $$

Memahami sifat Markov dan asumsi Markov sangat penting untuk menyederhanakan pemodelan distribusi probabilitas yang kompleks dalam bahasa. Dengan membatasi ketergantungan pada jendela konteks yang tetap, model n-gram membuat perhitungan probabilitas urutan menjadi mudah ditangani. Namun, penyederhanaan ini harus dibayar dengan mengabaikan ketergantungan jarak jauh yang melekat dalam bahasa alami. Pengembangan model bahasa berbasis jaringan saraf mengatasi keterbatasan ini dengan menangkap struktur semantik dan sintaksis atas konteks yang lebih panjang, yang mengarah pada kinerja unggul dalam berbagai tugas NLP.

Kemajuan ini memiliki implikasi signifikan baik untuk pemahaman teoretis maupun aplikasi praktis. Mereka meningkatkan kualitas terjemahan mesin, pengenalan ucapan, agen percakapan, input teks prediktif, peringkasan, dan sistem tanya jawab. Dengan memanfaatkan representasi berkelanjutan dan arsitektur yang mampu menangkap ketergantungan jangka panjang, model bahasa modern berperan penting dalam pengembangan teknologi NLP yang sedang berlangsung.

Sebagai kesimpulan, sifat Markov dan asumsi Markov tingkat tinggi telah menjadi esensial dalam evolusi pemodelan bahasa, memberikan dasar bagi pendekatan awal untuk prediksi urutan. Keterbatasan model n-gram karena ketergantungan mereka pada asumsi ini telah memacu pengembangan model canggih seperti jaringan saraf dan Transformer, yang menangkap kompleksitas bahasa manusia dengan lebih baik. Formulasi matematis dan konsep yang dibahas sangat mendasar dalam memahami dan meningkatkan model bahasa, berkontribusi pada kemajuan dalam NLP dan meningkatkan kemampuan mesin untuk memproses bahasa alami.

Dalam Python, kita dapat mengilustrasikan model bigram (asumsi Markov urutan ke-1) dengan kode yang menghitung pasangan kata dan menghitung probabilitas transisi hanya berdasarkan kata yang segera mendahuluinya.

### Contoh Python: Model Bigram Sederhana

```python
from collections import defaultdict

def create_bigram_counts(text):
    words = text.split()
    bigram_counts = defaultdict(int)
    for i in range(len(words) - 1):
        bigram = (words[i], words[i+1])
        bigram_counts[bigram] += 1
    return bigram_counts

def calculate_bigram_probabilities(bigram_counts):
    probabilities = {}
    word_counts = defaultdict(int)
    
    for (w1, w2), count in bigram_counts.items():
        word_counts[w1] += count
        
    for (w1, w2), count in bigram_counts.items():
        probabilities[(w1, w2)] = count / word_counts[w1]
    return probabilities

def main():
    text = "the quick brown fox jumps over the lazy dog"
    counts = create_bigram_counts(text)
    probs = calculate_bigram_probabilities(counts)
    
    for (w1, w2), prob in probs.items():
        print(f"P({w2} | {w1}) = {prob:.4f}")

if __name__ == "__main__":
    main()
```

Meskipun Rust bukan bahasa yang paling umum digunakan untuk membangun model deep learning, ia menyediakan fondasi performa tinggi untuk eksperimen dengan jaringan saraf. Kode Python berikut (menggunakan PyTorch) membuat korpus teks sintetis untuk melatih model bahasa berdasarkan LSTM, yang setara dengan implementasi Rust menggunakan `tch-rs`.

### Contoh Python: Model Bahasa Berbasis LSTM (PyTorch)

```python
import torch
import torch.nn as nn
import torch.optim as optim
import math

def generate_synthetic_corpus():
    return [
        "the cat sat on the mat",
        "the dog lay on the log",
        "a man ate an apple",
        "a woman read a book",
        "the child ran to the park",
    ]

def tokenize_corpus(corpus):
    vocab = {}
    idx = 0
    tokenized = []
    for sentence in corpus:
        sentence_tokens = []
        for word in sentence.split():
            if word not in vocab:
                vocab[word] = idx
                idx += 1
            sentence_tokens.append(vocab[word])
        tokenized.append(sentence_tokens)
    return tokenized, vocab, idx

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        # x: (seq_len) -> add batch dim (1, seq_len)
        embeds = self.embedding(x).unsqueeze(0)
        lstm_out, _ = self.lstm(embeds)
        logits = self.linear(lstm_out)
        return logits.squeeze(0) # (seq_len, vocab_size)

def main():
    corpus = generate_synthetic_corpus()
    tokenized, vocab, vocab_size = tokenize_corpus(corpus)
    
    model = LSTMModel(vocab_size, 50, 100)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Pelatihan sederhana
    for epoch in range(10):
        total_loss = 0
        for seq in tokenized:
            inputs = torch.tensor(seq[:-1], dtype=torch.long)
            targets = torch.tensor(seq[1:], dtype=torch.long)
            
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch: {epoch+1}, Loss: {total_loss/len(tokenized):.4f}")

    # Hitung Perplexity
    total_loss = 0
    with torch.no_grad():
        for seq in tokenized:
            inputs = torch.tensor(seq[:-1], dtype=torch.long)
            targets = torch.tensor(seq[1:], dtype=torch.long)
            logits = model(inputs)
            total_loss += criterion(logits, targets).item()
    
    avg_loss = total_loss / len(tokenized)
    print(f"Perplexity: {math.exp(avg_loss):.4f}")

if __name__ == "__main__":
    main()
```

## 9.2. Menyiapkan Lingkungan Kerja

Di sini kita mempelajari teknis penyiapan lingkungan untuk mengembangkan model bahasa. Penekanan pada performa dan keamanan membuat pilihan bahasa pemrograman menjadi krusial. Dalam konteks deep learning, manipulasi tensor, komputasi pada GPU, dan operasi matriks berkinerja tinggi adalah hal yang esensial.

Manajemen memori Rust bersifat deterministik dan aman, ditegakkan melalui model kepemilikannya (ownership). Dengan menjamin bahwa objek dibebaskan segera setelah keluar dari cakupan, Rust menghilangkan overhead memori dan risiko kebocoran memori. Model deterministik ini sangat berharga dalam pembelajaran mesin, di mana model sering kali melibatkan tensor besar dan struktur data rumit yang haus memori.

Model konkurensi Rust menyediakan pemrograman konkuren yang aman melalui abstraksi tanpa biaya (zero-cost abstractions) dan keamanan thread. Ini memungkinkan komputasi paralel yang kritis untuk melatih model skala besar.

Berikut adalah dasar-dasar operasi tensor menggunakan PyTorch di Python.

### Contoh Python: Operasi Tensor Dasar

```python
import torch

def main():
    # Membuat tensor pada CPU
    tensor_a = torch.tensor([1.0, 2.0, 3.0])
    tensor_b = torch.tensor([4.0, 5.0, 6.0])
    result = tensor_a + tensor_b
    
    print(f"Hasil penjumlahan tensor: {result}")

if __name__ == "__main__":
    main()
```

Untuk membangun jaringan saraf, kita perlu mengimplementasikan perkalian matriks, aktivasi nonlinear, dan fungsi kerugian. Lapisan linier (fully connected) melakukan perkalian matriks dan penambahan bias. Secara matematis, untuk input $x$, matriks bobot $W$, dan bias $b$, output $y$ dihitung sebagai:

$$y = W \cdot x + b$$

### Contoh Python: Lapisan Linier Sederhana

```python
import torch
import torch.nn as nn

def main():
    # Inisialisasi lapisan linier (input=3, output=2)
    linear = nn.Linear(3, 2)

    # Membuat tensor input dummy (batch=1, features=3)
    x = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
    y = linear(x)

    print(f"Output dari lapisan linier: {y}")

if __name__ == "__main__":
    main()
```

Mari kita susun MLP sederhana dengan satu lapisan tersembunyi untuk mengilustrasikan penggunaan lapisan linier, fungsi aktivasi, dan perhitungan loss. Pass maju (forward pass) dapat dinyatakan sebagai:

$$y = \sigma(W_2 \cdot (\sigma(W_1 \cdot x + b_1)) + b_2)$$

di mana $W_1$ dan $W_2$ adalah matriks bobot, $b_1$ dan $b_2$ adalah bias, dan $\sigma$ adalah fungsi aktivasi nonlinear seperti ReLU.

### Contoh Python: Pelatihan MLP Sederhana

```python
import torch
import torch.nn as nn
import torch.optim as optim

def main():
    # Mendefinisikan arsitektur MLP
    net = nn.Sequential(
        nn.Linear(3, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 1) # Lapisan output
    )

    # Konfigurasi optimizer Adam
    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # Input dan target dummy
    x = torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32)
    target = torch.tensor([[0.4]], dtype=torch.float32)

    # Loop pelatihan
    for epoch in range(1001):
        optimizer.zero_grad()
        output = net(x)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch: {epoch}, Loss: {loss.item():.6f}")

if __name__ == "__main__":
    main()
```

Dalam NLP, pemrosesan awal teks mengubah teks mentah menjadi data terstruktur. Berikut cara mengimplementasikan tokenizer sederhana di Python, mengubah teks menjadi token huruf kecil dan menghapus tanda baca.

### Contoh Python: Tokenizer Sederhana

```python
import re

def tokenize(text):
    text = text.lower()
    # Menghapus karakter non-alfanumerik
    text = re.sub(r'[^\w\s]', '', text)
    return text.split()

def main():
    text = "The quick brown fox jumps over the lazy dog!"
    tokens = tokenize(text)
    print(f"Tokens: {tokens}")

if __name__ == "__main__":
    main()
```

## 9.3. Pemrosesan Awal Data dan Tokenisasi

Bagian ini fokus pada proses pemrosesan awal dan tokenisasi yang kuat. Tokenisasi memecah teks menjadi unit yang lebih kecil (token). Metode tokenisasi bervariasi dari pendekatan berbasis kata, berbasis karakter, hingga subkata seperti BPE (Byte-Pair Encoding).

Tujuan tokenisasi adalah mengubah teks $x$ menjadi urutan token $T = \{t_1, t_2, \dots, t_n\}$. Kosakata $V$ mencakup semua token unik dalam korpus.

### Contoh Python: Tokenisasi Berbasis Kata (Regex)

```python
import re

def word_tokenize(text):
    # Mencari kata-kata dan mengubah ke huruf kecil
    return re.findall(r'\b\w+\b', text.lower())

def main():
    text = "Hello, Rust world! How are you today?"
    tokens = word_tokenize(text)
    print(f"Word tokens: {tokens}")

if __name__ == "__main__":
    main()
```

Metode tokenisasi subkata seperti BPE sangat umum karena ukuran kosakatanya yang efisien. BPE secara iteratif menggabungkan pasangan karakter yang paling sering muncul.

### Contoh Python: Implementasi BPE Sederhana

```python
from collections import Counter, defaultdict

def get_stats(vocab):
    pairs = defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[symbols[i], symbols[i+1]] += freq
    return pairs

def merge_vocab(pair, v_in):
    v_out = {}
    bigram = ' '.join(pair)
    replacement = ''.join(pair)
    for word in v_in:
        w_out = word.replace(bigram, replacement)
        v_out[w_out] = v_in[word]
    return v_out

def main():
    # Kosakata awal (dengan spasi antar karakter dan penanda akhir kata </w>)
    vocab = {'l o w </w>': 5, 'l o w e r </w>': 2, 'n e w e s t </w>': 6}
    
    num_merges = 10
    for i in range(num_merges):
        pairs = get_stats(vocab)
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        vocab = merge_vocab(best, vocab)
        print(f"Merge: {best}")
    
    print(f"Tokens setelah BPE: {list(vocab.keys())}")

if __name__ == "__main__":
    main()
```

### Contoh Python: Tokenisasi Berbasis Karakter (Unicode)

```python
def char_tokenize(text):
    return list(text)

def main():
    text = "你好，世界!"
    tokens = char_tokenize(text)
    print(f"Character tokens: {tokens}")

if __name__ == "__main__":
    main()
```

### Contoh Python: Membangun Kosakata (Vocab Builder)

```python
def build_vocabulary(tokens_list):
    vocab = {}
    idx = 0
    for token in tokens_list:
        if token not in vocab:
            vocab[token] = idx
            idx += 1
    return vocab

def main():
    tokens = ["hello", "world", "hello", "python"]
    vocab = build_vocabulary(tokens)
    print(f"Vocabulary: {vocab}")

if __name__ == "__main__":
    main()
```

## 9.4. Membangun Arsitektur Model

Arsitektur mendefinisikan bagaimana informasi mengalir melalui jaringan. Pemodelan bahasa telah berevolusi dari jaringan feedforward dasar ke RNN dan Transformer.

### Contoh Python: Jaringan Feedforward (PyTorch)

```python
import torch
import torch.nn as nn

class SimpleFFN(nn.Module):
    def __init__(self):
        super(SimpleFFN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10)
        )

    def forward(self, x):
        return self.net(x)

def main():
    model = SimpleFFN()
    x = torch.randn(1, 128)
    output = model(x)
    print(f"Feedforward Output: {output.shape}")

if __name__ == "__main__":
    main()
```

### Contoh Python: Lapisan LSTM (PyTorch)

```python
import torch
import torch.nn as nn

def main():
    # input_size=128, hidden_size=64
    lstm = nn.LSTM(128, 64, batch_first=True)

    # Input: (batch=1, seq_len=10, features=128)
    input_data = torch.randn(1, 10, 128)
    output, (hn, cn) = lstm(input_data)

    print(f"LSTM Output shape: {output.shape}")

if __name__ == "__main__":
    main()
```

Arsitektur Transformer ditentukan terutama oleh mekanisme self-attention. Berikut adalah implementasi fungsi scaled dot-product attention di Python.

### Contoh Python: Mekanisme Self-Attention

```python
import torch
import torch.nn.functional as F

def scaled_dot_product_attention(q, k, v):
    d_k = q.size(-1)
    # Skor: (Q * K^T) / sqrt(d_k)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    weights = F.softmax(scores, dim=-1)
    return torch.matmul(weights, v)

def main():
    q = torch.randn(1, 64)
    k = torch.randn(1, 64)
    v = torch.randn(1, 64)
    
    # Tambahkan dimensi batch dan sequence
    q, k, v = q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)
    
    output = scaled_dot_product_attention(q, k, v)
    print(f"Self-Attention Output shape: {output.shape}")

if __name__ == "__main__":
    import math
    main()
```

## 9.5. Melatih Model Bahasa

Proses pelatihan mencakup propagasi maju, perhitungan loss, dan propagasi mundur (backpropagation). Fungsi loss yang paling umum digunakan adalah cross-entropy:

$$L = -\sum_{i=1}^N y_i \log(\hat{y}_i)$$

Optimization algorithms seperti Adam sangat efisien untuk jaringan dalam karena kemampuannya menangani laju pembelajaran adaptif. Regularisasi seperti dropout membantu memitigasi overfitting dengan menonaktifkan neuron secara acak selama pelatihan.

### Contoh Python: Training Loop Dasar (PyTorch)

```python
import torch
import torch.nn as nn
import torch.optim as optim

def main():
    model = nn.Sequential(
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    for epoch in range(100):
        inputs = torch.randn(10, 128)
        targets = torch.randn(10, 10)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch: {epoch}, Loss: {loss.item():.4f}")

if __name__ == "__main__":
    main()
```

Untuk mencegah gradien meledak (*exploding gradients*), kita sering menggunakan teknik *gradient clipping*.

### Contoh Python: Pelatihan dengan Gradient Clipping

```python
import torch
import torch.nn as nn
import torch.optim as optim

def main():
    model = nn.Linear(128, 10)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    clip_norm = 0.5

    for epoch in range(100):
        inputs = torch.randn(10, 128)
        targets = torch.randn(10, 10)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = torch.mean((outputs - targets)**2)
        loss.backward()
        
        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
        
        optimizer.step()

if __name__ == "__main__":
    main()
```

## 9.6. Mengevaluasi dan Menyetel Halus Model

Evaluasi diperlukan untuk mengukur kemampuan generalisasi model. Perplexity ($PP$) didefinisikan sebagai:

$$PP = \exp \left( -\frac{1}{N} \sum_{i=1}^{N} \log P(w_i | w_{<i}) \right)$$

Metrik lain seperti BLEU (Bilingual Evaluation Understudy) sangat relevan untuk tugas terjemahan. Penyetelan halus (fine-tuning) menyesuaikan bobot model $\theta$ dengan meminimalkan fungsi loss tugas spesifik $L_{\text{task}}$ pada dataset domain tertentu.

### Contoh Python: Menghitung Perplexity

```python
import torch
import torch.nn.functional as F

def calculate_perplexity(model, data_loader):
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            outputs = model(inputs)
            # CrossEntropyLoss dengan reduction='sum'
            loss = F.cross_entropy(outputs, targets, reduction='sum')
            total_loss += loss.item()
            total_tokens += targets.numel()
            
    return torch.exp(torch.tensor(total_loss / total_tokens)).item()

# Contoh simulasi
# model = ...
# data_loader = ...
# print(f"Perplexity: {calculate_perplexity(model, data_loader)}")
```

### Contoh Python: Fine-Tuning untuk Klasifikasi Teks

```python
import torch
import torch.nn as nn
import torch.optim as optim

def main():
    # Asumsikan model sudah dilatih sebelumnya
    model = nn.Sequential(
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 1) # Lapisan output baru untuk klasifikasi biner
    )

    optimizer = optim.Adam(model.parameters(), lr=1e-5) # LR lebih kecil untuk fine-tuning

    for epoch in range(10):
        inputs = torch.randn(10, 128)
        labels = torch.randint(0, 2, (10, 1)).float()

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = nn.functional.binary_cross_entropy_with_logits(outputs, labels)
        loss.backward()
        optimizer.step()
        
        print(f"Fine-tuning Epoch: {epoch}, Loss: {loss.item():.4f}")

if __name__ == "__main__":
    main()
```

## 9.7. Menyebarkan Model Bahasa

Penyebaran (deployment) melibatkan transformasi model menjadi layanan yang siap produksi. Teknik seperti kuantisasi membantu mengurangi jejak memori dengan mengubah nilai floating-point $x$ menjadi representasi integer $x_q$:

$$x_q = \text{round}\left(\frac{x}{s}\right)$$

Berikut cara menyajikan prediksi model melalui API REST sederhana menggunakan Flask di Python.

### Contoh Python: Menyajikan Model via REST API (Flask)

```python
from flask import Flask, request, jsonify
import torch
import torch.nn as nn

app = Flask(__name__)

# Inisialisasi model sederhana
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)
    def forward(self, x):
        return self.fc(x)

model = SimpleModel()
model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_text = data.get("text", "")
    
    # Konversi panjang teks sebagai fitur dummy
    input_tensor = torch.tensor([[float(len(input_text))]*10])
    
    with torch.no_grad():
        output = model(input_tensor)
    
    return jsonify({"prediction": output.item()})

if __name__ == "__main__":
    # app.run(port=5000)
    pass
```

## 9.8. Tantangan dan Arah Masa Depan

Membangun dan menyebarkan LLM membutuhkan sumber daya komputasi yang besar. Kelangkaan data dan bias dalam data pelatihan juga menjadi tantangan utama.

![Gambar 4](https://lmvr.rantai.dev//images/L6A504iArDGBvUq8WSeO-XeTIEUvLdvFdN3NwgpxX-v1.png)
**Gambar 4:** Peran Rust dalam pengembangan dan penyebaran LLM.

Tren saat ini mengarah pada integrasi kemampuan multimodal (teks, gambar, audio) dan peningkatan interpretabilitas model menggunakan metode seperti LRP (Layer-wise Relevance Propagation). Rust, dengan FFI (*Foreign Function Interface*), dapat berintegrasi mulus dengan Python untuk menggabungkan kemudahan eksperimen dengan kecepatan eksekusi.

## 9.9. Kesimpulan

Bab 9 membekali pembaca dengan pengetahuan dan keterampilan praktis untuk membangun LLM sederhana dari nol menggunakan Rust, mulai dari penyiapan awal hingga penyebaran.

### 9.9.1. Pembelajaran Lebih Lanjut dengan GenAI

- Jelaskan perbedaan mendasar antara model n-gram tradisional dan model bahasa berbasis jaringan saraf modern.
- Deskripsikan proses penyiapan lingkungan pengembangan Rust untuk machine learning.
- Diskusikan peran pemrosesan awal data dalam tugas NLP (tokenisasi, normalisasi).
- Bandingkan efektivitas RNN, LSTM, dan Transformer dalam menangkap ketergantungan jarak jauh.
- Jelaskan arsitektur jaringan saraf feedforward dasar untuk model bahasa.
- Apa pentingnya mekanisme atensi dalam model bahasa modern?
- Deskripsikan proses pelatihan model (forward propagation, loss calculation, backpropagation).
- Analisis tantangan overfitting dan teknik mitigasinya (dropout, regularisasi).
- Jelajahi peran penyetelan hyperparameter dalam mengoptimalkan model.
- Diskusikan signifikansi metrik evaluasi seperti perplexity dan BLEU score.

### 9.9.2. Latihan Praktis

#### Latihan Mandiri 9.1: Mengimplementasikan Berbagai Teknik Tokenisasi
Tujuan: Memahami dampak metode tokenisasi berbasis kata, subkata (BPE), dan karakter terhadap performa model.

#### Latihan Mandiri 9.2: Membangun Transformer Sederhana dari Nol
Tujuan: Mendapatkan pengalaman langsung dalam menyusun mekanisme self-attention dan positional encoding.

#### Latihan Mandiri 9.3: Penyetelan Hyperparameter
Tujuan: Bereksperimen secara sistematis dengan learning rate, ukuran batch, dan jumlah epoch.

#### Latihan Mandiri 9.4: Implementasi Kuantisasi Model
Tujuan: Mengoptimalkan model yang sudah dilatih untuk penyebaran dengan mengurangi presisi bobot dan mengukur kecepatan inferensinya.

#### Latihan Mandiri 9.5: Mengatasi Overfitting
Tujuan: Mengintegrasikan teknik dropout dan L2 ke dalam loop pelatihan dan memantau kurva loss validasi.