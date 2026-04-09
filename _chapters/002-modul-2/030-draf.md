---
slug: modul-2-3
title: modul-2-3
---
### 2.3 Tokenisasi dan Embedding: Pra-Proses Fundamental LLM
---

Kita telah menyelesaikan pembahasan kita tentang model n-gram (2.2). Kita melihat bagaimana model statistik klasik mencoba mengatasi masalah sparsitas data (Gambar 3) melalui asumsi (Markov) dan teknik smoothing (seperti Kneser-Ney). Namun, model n-gram memiliki dua keterbatasan fundamental yang tidak dapat diselesaikan oleh smoothing secanggih apa pun:

1.  **Konteks Terbatas:** Model trigram tidak akan pernah bisa menghubungkan kata di awal kalimat dengan kata di akhir kalimat.
2.  **Generalisasi Semantik Nihil:** Model n-gram tidak "tahu" bahwa "kucing" dan "meong" mirip secara makna. Jika ia dilatih pada "kucing makan ikan" tetapi tidak pernah melihat "meong makan ikan", ia tidak bisa menggeneralisasi.

Untuk mengatasi ini, kita harus beralih ke era model neural. Namun, sebelum kita dapat membangun model neural yang canggih (seperti RNN atau Transformer di 2.4 dan 2.5), kita harus menjawab pertanyaan yang lebih mendasar: Bagaimana kita merepresentasikan teks agar dapat diproses oleh neural network? Komputer tidak memahami kata "kucing"; ia hanya memahami angka.

Subtopik berikutnya adalah fondasi absolut untuk semua LLM modern. Kita akan membahas dua langkah pra-proses yang krusial, yaitu: Tokenisasi (proses mengubah teks mentah menjadi serangkaian ID numerik) dan Embedding (proses mengubah ID tersebut menjadi vektor yang padat makna, yang memungkinkan model untuk belajar bahwa 'kucing' dan 'meong' itu mirip).

---

#### 2.3.1 Tokenisasi: Mengubah Teks Menjadi Angka

Langkah pertama dalam pipeline NLP modern adalah Tokenisasi. Apa itu Tokenisasi?

> **Definisi Tokenisasi**
> Tokenisasi adalah sebuah proses memecah sekuens teks mentah (string) menjadi unit-unit yang lebih kecil yang disebut token. Token-token ini kemudian dipetakan ke ID numerik (angka integer) berdasarkan sebuah vokabulari yang telah ditetapkan.

Mengapa kita tidak bisa memecah kalimat berdasarkan spasi saja? ("word-based tokenization"). Pendekatan itu gagal total ketika menghadapi kompleksitas bahasa nyata:

*   **Tanda Baca:** Bagaimana "Jangan!" dipecah? Menjadi "Jangan" dan "!"? Atau menjadi "Jangan!"?
*   **Kontraksi/Singkatan:** "Don't" seharusnya satu token atau dua ("do" dan "n't")? Bagaimana dengan "Prof."?
*   **Kata Majemuk:** "Penanggung jawab" itu satu makna, tapi dua kata.
*   **Kata-kata Langka (Out-of-Vocabulary / OOV):** Apa yang terjadi jika model bertemu kata yang tidak ada di vokabularinya, misalnya "Xylophone" atau salah ketik "makann"?

Untuk mengatasi ini, LLM modern menggunakan Tokenisasi Subword (*Subword Tokenization*). Ada beberapa algoritma populer untuk menerapkan strategi subword ini (seperti WordPiece, Unigram, atau BPE). Dari semua itu, metode yang paling fundamental dan berpengaruh, yang awalnya dikembangkan untuk kompresi data dan kemudian diadopsi oleh NLP, adalah Byte-Pair Encoding (BPE).

**Byte-Pair Encoding (BPE)**

Algoritma tokenisasi subword yang paling umum adalah BPE. Bayangkan BPE sebagai seorang "ahli bahasa kompresi". la melihat seluruh korpus teks dan bekerja dari unit terkecil (karakter) ke atas.

**Cara Kerja BPE (Konseptual):**

1.  **Inisialisasi:** BPE memulai dengan vokabulari yang hanya berisi semua karakter dasar (misal: a, b, c, .., Z).
2.  **Iterasi (Merge):** BPE secara berulang mencari pasangan (*pair*) byte/karakter yang paling sering muncul berdampingan di korpus, dan menggabungkannya (*merge*) menjadi satu token baru di vokabulari.
3.  **Berhenti:** Proses ini diulang sampai vokabularinya mencapai ukuran yang diinginkan (misal, 50.000 token).

**Contoh Konkret (Cara Kerja BPE):**

Misalkan korpus kita memiliki banyak kata: low (5x), lowest (2x), newer (6x), wider (3x)
*   **Vokabulari awal:** l, o, w, e, s, t, n, r, d, i
*   **Iterasi 1:** Pasangan **e** dan **r (er)** adalah yang paling sering muncul (6+3=9x). Vokabulari baru: l, o, w, e, s, t, n, r, d, i, **er**. Teks menjadi: low, lowest, new**er**, wid**er**.
*   **Iterasi 2:** Pasangan **l** dan **o (lo)** sering muncul (5+2=7x). Vokabulari baru:.... er, **lo**. Teks menjadi: **lo**w, **lo**west, newer, wider.
*   **Iterasi 3:** Pasangan **lo** dan **w (low)** sering muncul (7x). Vokabulari baru: ... er, lo, **low**. Teks menjadi: **low**, **low**est, newer, wider.

...dan seterusnya.

**Implementasi Kode dari BPE:**

```python
import re
import collections

# 1. Definisikan korpus kita (kata dipisah spasi, </w> menandai akhir kata)
# dan frekuensinya (sesuai contoh).
corpus = {
    'l o w </w>': 5,
    'l o w e s t </w>': 2,
    'n e w e r </w>': 6,
    'w i d e r </w>': 3
}

def get_stats(corpus):
    """Hitung frekuensi semua pasangan (pair) yang berdampingan."""
    pairs = collections.defaultdict(int)
    for word, freq in corpus.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[symbols[i], symbols[i+1]] += freq
    return pairs

def merge_corpus(pair, corpus_in):
    """Gabungkan pasangan 'pair' di semua kata dalam korpus."""
    corpus_out = {}
    # 're.escape' penting agar karakter khusus (jika ada) tidak error
    pattern = re.compile(r'(?<!\S)' + re.escape(' '.join(pair)) + r'(?!\S)')
    for word_in, freq in corpus_in.items():
        # Ganti pasangan (misal: "e r") dengan token gabungan ("er")
        word_out = pattern.sub(''.join(pair), word_in)
        corpus_out[word_out] = freq
    return corpus_out

# -- Simulasi Pelatihan ---
num_merges = 3 # Kita hanya simulasikan 3 iterasi
vokab = ['l', 'o', 'w', 'e', 's', 't', 'n', 'r', 'd', 'i'] # Vokab awal

print("--- Memulai Pelatihan BPE (Simulasi) ---")
print(f"Vokabulari Awal: {vokab}\n")

for i in range(num_merges):
    pairs = get_stats(corpus)
    if not pairs:
        break
    # 1. Temukan pasangan terbaik (paling sering)
    best_pair = max(pairs, key=pairs.get)
    best_freq = pairs[best_pair]

    print(f"--- Iterasi {i+1} ---")
    print(f"Pasangan terbaik: {best_pair} (Frekuensi: {best_freq}x)")

    # 2. Gabungkan pasangan itu di korpus
    corpus = merge_corpus(best_pair, corpus)

    # 3. Tambahkan token baru ke vokabulari
    new_token = ''.join(best_pair)
    vokab.append(new_token)

    print(f"Vokabulari Baru (parsial): {vokab}")
    print(f"Teks Menjadi (Internal): {list(corpus.keys())}\n")
```

**Output dari kode di atas adalah sebagai berikut:**

```text
--- Memulai Pelatihan BPE (Simulasi) ---
Vokabulari Awal: ['l', 'o', 'w', 'e', 's', 't', 'n', 'r', 'd', 'i']

--- Iterasi 1 ---
Pasangan terbaik: ('e', 'r') (Frekuensi: 9x)
Vokabulari Baru (parsial): ['l', 'o', 'w', 'e', 's', 't', 'n', 'r', 'd', 'i', 'er']
Teks Menjadi (Internal): ['l o w </w>', 'l o w e s t </w>', 'n e w er </w>', 'w i d er </w>']

--- Iterasi 2 ---
Pasangan terbaik: ('er', '</w>') (Frekuensi: 9x)
Vokabulari Baru (parsial): ['l', 'o', 'w', 'e', 's', 't', 'n', 'r', 'd', 'i', 'er', 'er</w>']
Teks Menjadi (Internal): ['l o w </w>', 'l o w e s t </w>', 'n e w er</w>', 'w i d er</w>']

--- Iterasi 3 ---
Pasangan terbaik: ('l', 'o') (Frekuensi: 7x)
Vokabulari Baru (parsial): ['l', 'o', 'w', 'e', 's', 't', 'n', 'r', 'd', 'i', 'er', 'er</w>', 'lo']
Teks Menjadi (Internal): ['lo w </w>', 'lo w e s t </w>', 'n e w er</w>', 'w i d er</w>']
```

**Analisis Output:**

*   **Iterasi 1:** Algoritma memindai semua pasangan. la menemukan bahwa **('e', 'r')** adalah yang paling sering muncul (total 9x: 6x dari "newer" + 3x dari "wider"). Ini mengalahkan pasangan lain seperti **('w', 'e') (8x)** atau **('l', 'o') (7x)**. Pasangan **('e', 'r')** digabung menjadi token baru **er**.
*   **Iterasi 2 (Poin Kunci):** Algoritma memindai ulang korpus yang sudah diperbarui. Saat menghitung frekuensi baru, pasangan baru **('er', '</w>')** (dari "newer </w>" dan "wider </w>") sekarang memiliki frekuensi total **9x**. Karena 9 > 7, algoritma secara logis memilih **('er', '</w>')** sebagai pasangan terbaik berikutnya, menggabungkannya menjadi token **er</w>**.
*   **Iterasi 3:** Setelah dua pasangan 9x selesai, algoritma mencari frekuensi tertinggi berikutnya, yaitu **('l', 'o') (7x)**, dan menggabungkannya menjadi token **lo**.

Output ini adalah sebuah demonstrasi dari cara kerja BPE. Algoritma ini murni bersifat statistik. Ia tidak peduli bahwa **</w>** adalah **"akhir kata"**; ia hanya melihat bahwa kombinasi **('er', '</w>')** sangat sering muncul. Dengan menggabungkannya, ia secara efisien menemukan unit yang paling sering berulang dalam data, yang dalam hal ini adalah imbuhan akhir **"er"**.

**Hasil Akhir BPE:**
*   Kata-kata umum (seperti "makan") akan menjadi satu token tunggal.
*   Kata-kata langka (seperti "transformer") akan dipecah menjadi subword yang bermakna **(trans + former)**.
*   Kata-kata unseen atau salah ketik (seperti "makannn") dapat direpresentasikan sebagai kombinasi subword **(makan + n + n)** sehingga model tidak pernah menghadapi token yang benar-benar tidak dikenal (OOV).

**Visualisasi Tokenisasi**

![Gambar 5. Alur Tokenisasi](https://lms.sdmdigital.id/pluginfile.php/635363/mod_page/content/11/image.png)
**Gambar 5. Alur Tokenisasi**

Gambar 5 mengilustrasikan alur kerja lengkap dari proses tokenisasi. Proses dimulai dengan Teks Mentah (String), dalam contoh ini adalah **"Arsitektur Keren"**. Teks mentah ini kemudian dimasukkan ke dalam Tokenizer (BPE). Tokenizer BPE memecah string tersebut menjadi daftar Token Subword berdasarkan vokabulari yang telah dipelajarinya, menghasilkan **["Arsi", "tektur", "Keren"]**. Perhatikan bagaimana "Arsitektur" dipecah menjadi unit yang bermakna. Setiap token subword ini kemudian dicocokkan dengan Vokabulari Lookup (Tabel Pencarian ID). Hasil akhirnya adalah sekuens Token ID (Integer), misalnya **[4001, 890, 5011]**. Angka-angka inilah yang siap diterima sebagai input oleh model neural network.

Penjelasan dan diagram di atas telah mencakup alur untuk tokenisasi teks konten. Namun, pipeline tokenisasi belum selesai. Agar model neural dapat beroperasi, ia juga memerlukan **token fungsional** atau **token instruksional** khusus. Token-token ini ditambahkan secara eksplisit ke vokabulari untuk memberi tahu model struktur sekuens.

**Special Tokens**

Tokenizer juga menambahkan token-token khusus yang krusial untuk model, seperti yang diilustrasikan Tabel 4.

**Tabel 4. Token Khusus (Special Tokens) dalam Tokenisasi**

| Token | Nama | Fungsi Utama |
| :--- | :--- | :--- |
| **[CLS]** | Classification | Token penanda awal sekuens, sering dipakai untuk tugas klasifikasi. |
| **[SEP]** | Separator | Token pemisah antar kalimat (misal, dalam tugas Q&A). |
| **[UNK]** | Unknown | Token untuk kata yang benar-benar tidak dikenal (meski BPE mengurangi ini). |
| **[PAD]** | Padding | Token untuk "mengisi" sekuens yang lebih pendek agar seragam dalam satu batch. |

Tabel 4 merangkum token-token fungsional yang paling umum. Token-token inilah yang memungkinkan model untuk memahami struktur dari input dan untuk menangani batching data secara efisien.

Pada titik ini, kita telah berhasil mengubah teks mentah (seperti **"Arsitektur Keren"**) menjadi sekuens ID numerik yang bersih dan terstruktur (seperti **[4001, 890, 5011]**). Namun, angka-angka ini pada dasarnya arbitrer. Bagi komputer, angka **890** ("tektur") tidak memiliki hubungan matematis apapun dengan angka **891** (misal: "struktur"). Langkah selanjutnya adalah mengubah angka-angka ID yang arbitrer ini menjadi representasi yang kaya akan makna.

---

#### 2.3.2 Embedding: Mengubah Angka Menjadi Vektor Bermakna

Tokenisasi telah memberi kita output berupa ID numerik: **"Arsitektur Keren" → [4001, 890, 5011]**. Masalahnya, model tidak memiliki pemahaman bawaan bahwa ID **890** ("tektur") mungkin secara semantik lebih dekat ke ID **891** ("struktur") daripada ke ID **5011** ("Keren"). Di sinilah Embedding berperan.

> **Apa itu Embedding?**
> Embedding adalah sebuah representasi vektor yang padat (*dense*), berdimensi relatif rendah (misal, 768 dimensi), dan dapat dilatih (*trainable*) untuk setiap token dalam vokabulari.

Alih-alih menggunakan ID **890**, model akan "mencari" (*lookup*) vektor untuk ID **890** dari sebuah tabel besar (Embedding Layer). Secara teknis, Embedding Layer ini adalah sebuah *weight matrix* (matriks bobot). Karena "dapat dilatih", nilai-nilai (bobot) di dalam vektor-vektor ini akan terus-menerus diperbarui melalui proses backpropagation. Saat model membuat kesalahan prediksi, gradient (sinyal error) akan mengalir kembali hingga ke Embedding Layer, sedikit mengubah nilai vektor agar lebih sesuai. Inilah proses di mana model "belajar" makna semantik.

**Analogi: Embedding sebagai Peta Makna**

Bayangkan Embedding Layer sebagai sebuah "peta makna" raksasa dalam 768 dimensi. Tokenisasi memberi tahu Anda "alamat" (ID **890**), dan Embedding memberi tahu Anda koordinat (vektor) dari alamat tersebut di dalam peta. Dalam peta ini, token yang memiliki makna serupa akan secara alami memiliki koordinat yang berdekatan:
*   Vektor untuk "raja" akan dekat dengan "ratu" dan "kaisar".
*   Vektor untuk "kucing" akan dekat dengan "meong" dan "anjing".
*   Vektor untuk "raja" akan sangat jauh dari "apel" atau "logaritma".

**Bagaimana ini Memecahkan Masalah Generalisasi?**

Inilah keajaiban model neural. Vektor-vektor ini dapat dilatih (*trainable*). Ketika model dilatih pada kalimat "kucing makan ikan", ia akan memperbarui vektor untuk "kucing", "makan", dan "ikan". Karena vektor "kucing" dan "meong" berdekatan di peta makna, model secara otomatis menggeneralisasi pengetahuannya. Lain kali ia melihat "meong makan ikan", ia sudah "tahu" bagaimana memprosesnya karena ia memprosesnya menggunakan vektor yang hampir identik dengan "kucing".

**Masalah Terakhir: Bagaimana dengan Urutan?**

Jika kita hanya mengubah setiap token menjadi vektor, kita punya "sekantong vektor" (*bag of vectors*). Model tidak tahu urutan **["Arsi", "tektur", " Keren"]** vs **["Keren", "tektur", "Arsi"]**. Solusinya adalah Positional Encoding (Encoding Posisi).

> **Apa itu Positional Encoding?**
> Positional Encoding adalah sebuah vektor (dengan dimensi yang sama dengan embedding) yang secara unik mengidentifikasi posisi sebuah token dalam sekuens (apakah ia token ke-1, ke-2, ke-3, dst.).

Vektor posisi ini kemudian ditambahkan ke vektor token embedding:
```text
Input_Model (Pos 1) = Token_Embedding("Arsi") + Positional_Encoding(posisi_1) 
Input_Model (Pos 2) = Token_Embedding("tektur") + Positional_Encoding(posisi_2) 
Input_Model (Pos 3) = Token_Embedding(" Keren") + Positional_Encoding(posisi_3)
```

**Visualisasi Kombinasi Token Embedding dan Positional Encoding**

![Gambar 6. Kombinasi Token Embedding dan Positional Encoding](https://lms.sdmdigital.id/pluginfile.php/635363/mod_page/content/11/image%20%281%29.png)
**Gambar 6. Kombinasi Token Embedding dan Positional Encoding**

Gambar 6 mengilustrasikan proses bagaimana input akhir untuk model LLM dibentuk. Token Embedding (makna kata) dan Positional Encoding (lokasi kata) digabungkan melalui **Penjumlahan Vektor**. Vektor hasil penjumlahan inilah yang menjadi **Input Vektor Akhir** yang siap diproses oleh lapisan-lapisan neural network (seperti Transformer).

---

#### 2.3.3 Toolkit Praktis: Library transformers dan tokenizers

Hugging Face telah menyediakan library yang sangat kuat dan terstandardisasi untuk menangani pipeline ini:

1.  **tokenizers:** Library backend (Rust) yang berisi implementasi murni algoritma seperti BPE, WordPiece, dll.
2.  **transformers:** Library frontend (Python) yang menyediakan pembungkus (*wrapper*) level atas untuk mengunduh dan menggunakan model pre-trained.

Cuplikan kode Python menggunakan library transformers:

```python
# Mengimpor kelas 'AutoTokenizer' dari library transformers
from transformers import AutoTokenizer 

model_name = "bert-base-uncased"

# 2. Memuat tokenizer yang sudah dilatih (pre-trained)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 3. Teks mentah kita (konsisten dengan contoh 2.3.1)
teks_mentah = "Arsitektur Keren"

# 4. Menjalankan proses tokenisasi lengkap.
output = tokenizer(teks_mentah)

# 5. Menganalisis output
# .tokenize() adalah helper untuk melihat string token
print(f"Token Subword: {tokenizer.tokenize(teks_mentah)}")

# output['input_ids'] adalah hasil akhirnya
print(f"Token ID (Input IDs): {output['input_ids']}")
```

**Output yang dihasilkan:**
```text
Token Subword: ['arsitek', '##tur', 'keren']
Token ID (Input IDs): [101, 22223, 11467, 24209, 102]
```

**Analisis Output:**
1.  **Tokenisasi Subword:** Tokenizer BERT memecah "Arsitektur" menjadi token yang lebih umum: **"arsitek"** dan **"##tur"** (## berarti lanjutan dari kata sebelumnya).
2.  **Special Tokens:** Terdapat ID **101 ([CLS])** di awal dan **102 ([SEP])** di akhir sekuens.

**Bagaimana dengan Embedding?**
Untuk mengubah Token ID menjadi Vektor Embedding, kita perlu memuat model sesungguhnya menggunakan library deep learning (seperti PyTorch).

```python
from transformers import AutoModel
import torch

# --- (Melanjutkan kode sebelumnya) ---
input_ids = output['input_ids']

# 1. Memuat model penuh
model = AutoModel.from_pretrained(model_name)

# 2. Konversi List ID menjadi Tensor (batch size 1)
input_ids_tensor = torch.tensor(input_ids).unsqueeze(0)

# 3. Mendapatkan Embeddings (Gambar 6)
with torch.no_grad(): 
    embedding_output = model.embeddings(input_ids=input_ids_tensor)

# 4. Menganalisis hasil embedding
print(f"\n--- Output Embedding (Gambar 6) ---")
print(f"Tipe data output: {type(embedding_output)}")
print(f"Dimensi (Shape) Vektor: {embedding_output.shape}")
```

**Output yang dihasilkan:**
```text
Tipe data output: <class 'torch.Tensor'>
Dimensi (Shape) Vektor: torch.Size([1, 5, 768])
```

**Bedah `torch.size([1, 5, 768])`:**
*   **1 (Batch Size):** Memproses 1 kalimat.
*   **5 (Sequence Length):** Jumlah total token (CLS + arsitek + ##tur + keren + SEP).
*   **768 (Embedding Dimension):** Dimensi vektor untuk model `bert-base-uncased`.

Artinya, `embedding_output` berisi 5 vektor, di mana setiap vektor memiliki 768 angka. Vektor-vektor inilah yang merepresentasikan Input Vektor Akhir (gabungan makna dan lokasi) yang siap masuk ke LLM.

---

### Mini Quiz

*Silakan akses Mini Quiz melalui platform LMS:*
[Multiple Choice Quiz - Subtopik 2.3](https://lms.sdmdigital.id/h5p/embed.php?url=https%3A%2F%2Flms.sdmdigital.id%2Fpluginfile.php%2F635363%2Fmod_page%2Fcontent%2F11%2Fmultiple-choice-203.h5p)

---

> **Refleksi Singkat**
> Kita telah membahas bahwa model n-gram (2.2) gagal menggeneralisasi makna (misalnya, ia tidak tahu "kucing" dan "meong" itu mirip secara semantik). Bagaimana arsitektur neural yang menggunakan Embedding (2.3.2) secara fundamental menyelesaikan masalah generalisasi semantik ini? 
> *Petunjuk: Pikirkan tentang apa yang terjadi pada vektor "kucing" dan "meong" di "peta makna" selama proses pelatihan.*