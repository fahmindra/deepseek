---
slug: lmvr-5
title: lmvr-5
---
# Bab 5: Model Bidirectional: BERT dan Variannya

> **"Atensi adalah semua yang Anda butuhkan. Arsitektur Transformer telah secara fundamental mengubah cara kita berpikir tentang pemodelan urutan, memungkinkan kita untuk menangani tugas-tugas bahasa yang kompleks dengan efisiensi dan akurasi yang belum pernah terjadi sebelumnya." — Ashish Vaswani**

> **Catatan:** Bab 4 dari LMVR memberikan eksplorasi mendalam tentang arsitektur Transformer, model terobosan yang merevolusi pemrosesan bahasa alami dengan memungkinkan pemrosesan paralel yang efisien dan menangkap ketergantungan jarak jauh dalam teks. Bab ini mencakup komponen-komponen penting seperti self-attention, multi-head attention, dan positional encoding, menjelaskan bagaimana elemen-elemen ini bekerja bersama untuk meningkatkan kemampuan model dalam memahami dan menghasilkan bahasa. Bab ini juga menggali struktur encoder-decoder, normalisasi lapisan, dan koneksi residual, menyoroti peran mereka dalam menstabilkan dan mengoptimalkan model. Aspek praktis dari implementasi dan fine-tuning model Transformer di Rust juga dibahas, menawarkan alat bagi pembaca untuk menerapkan konsep-konsep ini pada tugas-tugas NLP di dunia nyata secara efektif.

## 5.1. Pengantar Model Bidirectional

Dalam pemrosesan bahasa alami (NLP), model bidirectional seperti BERT (Bidirectional Encoder Representations from Transformers) telah merevolusi cara mesin memahami bahasa dengan menangkap konteks dari kedua arah dalam sebuah kalimat. Secara tradisional, banyak model bahasa, seperti GPT (Generative Pretrained Transformer), bersifat unidirectional (searah), yang berarti mereka menghasilkan atau memprediksi kata-kata hanya berdasarkan konteks sebelumnya. Keterbatasan ini membuat model unidirectional sulit untuk sepenuhnya memahami makna kata-kata yang bergantung pada konteks masa depan dalam kalimat tersebut. Sebagai contoh, dalam kalimat "Bank tersebut menyetujui pinjaman itu," kata "bank" dapat merujuk pada lembaga keuangan atau tepi sungai (dalam bahasa Inggris "bank"), dan memahami konteks "menyetujui pinjaman" (konteks masa depan) sangat penting untuk menginterpretasikan kata "bank" dengan benar.

Model bidirectional seperti BERT mengatasi masalah ini dengan mempertimbangkan konteks kiri-ke-kanan dan kanan-ke-kiri secara bersamaan. Di BERT, misalnya, model belajar memprediksi kata-kata yang disamarkan (masked) dengan menggunakan informasi dari kedua sisi kata target. Secara formal, untuk kalimat yang diberikan $S = [x_1, x_2, ..., x_n]$, model bidirectional menghitung representasi untuk setiap kata $x_i$ dengan memperhatikan seluruh kalimat $S$, sehingga mempertimbangkan informasi dari $x_1$ hingga $x_{i-1}$ (konteks sebelumnya) dan $x_{i+1}$ hingga $x_n$ (konteks sesudahnya). Desain ini memungkinkan BERT untuk menangkap ketergantungan dua arah dalam bahasa, yang sangat penting untuk banyak tugas hilir seperti menjawab pertanyaan dan analisis sentimen, di mana memahami konteks penuh sangat penting untuk interpretasi yang akurat.

![Gambar 1: Prosedur pra-pelatihan dan penyetelan halus secara keseluruhan untuk BERT (Ref: https://arxiv.org/pdf/1810.04805).](https://lmvr.rantai.dev//images/L6A504iArDGBvUq8WSeO-7pzMEIdiYe415Ay4hTs1-v1.png)
**Gambar 1:** Prosedur pra-pelatihan dan penyetelan halus secara keseluruhan untuk BERT.

BERT dapat disetel halus (fine-tuned) untuk tugas-tugas seperti MNLI (Multi-Genre Natural Language Inference), NER (Named Entity Recognition), dan SQuAD (Stanford Question Answering Dataset) dengan memanfaatkan sifat bidirectional-nya. Untuk MNLI, BERT disetel halus untuk memahami hubungan antara pasangan kalimat dengan memperhatikan kedua kalimat secara bersamaan, membantunya menentukan apakah satu kalimat mengandung konsekuensi, kontradiksi, atau netral terhadap kalimat lainnya. Dalam NER, BERT menggunakan konteks dua arah untuk melabeli entitas bernama secara akurat dalam sebuah kalimat (misalnya nama, lokasi), memastikan bahwa model mempertimbangkan kata itu sendiri dan kata-kata di sekitarnya. Untuk SQuAD, BERT disetel halus untuk menjawab pertanyaan dengan memprediksi token awal dan akhir dalam sebuah bagian teks yang paling cocok dengan kueri yang diberikan. Pendekatan bidirectional ini memungkinkan BERT unggul dalam tugas-tugas yang memerlukan interpretasi tepat dari ketergantungan bahasa.

Secara matematis, proses ini sering dicapai menggunakan *masked language modeling* (MLM), di mana sebagian dari token masukan disamarkan, dan model dilatih untuk memprediksi token yang disamarkan tersebut menggunakan konteks sekitarnya. Untuk token yang disamarkan $x_m$, tujuannya adalah untuk memaksimalkan probabilitas $P(x_m | x_1, ..., x_{m-1}, x_{m+1}, ..., x_n)$, dengan memanfaatkan informasi dari kedua arah. Hal ini kontras dengan model unidirectional, yang biasanya mengoptimalkan tujuan autoregresif yang memprediksi kata berikutnya berdasarkan kata-kata sebelumnya, yaitu $P(x_m | x_1, ..., x_{m-1})$.

Konteks dalam Pemahaman Bahasa Alami adalah hal yang fundamental bagi sebagian besar tugas NLP. Pertimbangkan tugas-tugas seperti menjawab pertanyaan, di mana model harus memahami konteks pertanyaan dan mengekstrak informasi yang relevan dari sebuah paragraf. Model bidirectional memungkinkan pemahaman yang lebih kaya dengan memperhatikan seluruh input. Demikian pula, dalam analisis sentimen, sentimen suatu kata dapat bergantung pada seluruh konteks kalimat. Misalnya, frasa "tidak buruk" mengekspresikan sentimen positif, meskipun kata "buruk" saja memiliki konotasi negatif. Model bidirectional dapat menangkap nuansa ini secara efektif.

Pengembangan BERT dan pendekatan dua arahnya dimotivasi oleh keterbatasan model unidirectional, terutama dalam menangkap ketergantungan jarak jauh dalam teks. Model unidirectional, seperti GPT, mengandalkan autoregresi, memprediksi token secara berurutan dari kiri ke kanan. Meskipun ini berfungsi dengan baik untuk tugas generatif seperti pembuatan teks, ia tidak sepenuhnya menangkap hubungan antar kata yang memerlukan pemahaman seluruh urutan sekaligus. Model bidirectional memecahkan masalah ini dengan memproses kalimat secara holistik.

Sebelum membahas implementasi detail model BERT, mari kita mulai dengan lapisan input. Lapisan input BERT terdiri dari tiga penyematan (embedding) kunci: *token embedding*, *positional embedding*, dan *segment (atau sentence) embedding*. *Token embedding* bertanggung jawab untuk mengubah setiap token menjadi representasi vektor padat menggunakan tokenizer WordPiece. *Positional embedding* mengkodekan posisi setiap token dalam urutan. Terakhir, *segment embedding* membantu BERT membedakan antara token dari kalimat yang berbeda dalam tugas yang melibatkan pasangan kalimat.

![Gambar 2: Ilustrasi lapisan input model BERT.](https://lmvr.rantai.dev//images/L6A504iArDGBvUq8WSeO-8Tv8PqU5Q2zFhtK8hIds-v1.png)
**Gambar 2:** Ilustrasi lapisan input model BERT.

Lapisan klasifikasi mengambil representasi gabungan akhir dan mengubahnya menjadi logit untuk klasifikasi biner. Model dilatih menggunakan kerugian *cross-entropy* dan parameternya diperbarui menggunakan optimizer Adam. Dalam praktiknya, dataset yang telah diproses sebelumnya seperti IMDB (untuk analisis sentimen) akan digunakan.

Berikut adalah implementasi Python menggunakan library `transformers` dari Hugging Face untuk melakukan analisis sentimen.

### Contoh Python: Analisis Sentimen dengan Pipeline BERT

```python
from transformers import pipeline

def main():
    # Menyiapkan classifier sentiment menggunakan model default (DistilBERT/BERT)
    sentiment_classifier = pipeline("sentiment-analysis")

    # Mendefinisikan input
    inputs = [
        "Probably my all-time favorite movie, a story of selflessness, sacrifice and dedication to a noble cause, but it's not preachy or boring.",
        "This film tried to be too many things all at once: stinging political satire, Hollywood blockbuster, sappy romantic comedy, family values promo...",
        "If you like original gut wrenching laughter you will like this movie. If you are young or old then you will love this movie, hell even my mom liked it.",
    ]

    # Menjalankan model
    outputs = sentiment_classifier(inputs)
    
    for i, result in enumerate(outputs):
        print(f"Input {i+1}: {inputs[i][:50]}...")
        print(f"Hasil: {result}\n")

if __name__ == "__main__":
    main()
```

Berikut adalah contoh lain untuk analisis sentimen menggunakan konfigurasi model yang lebih spesifik (seperti FNet).

### Contoh Python: Analisis Sentimen dengan Model Spesifik (FNet/SST2)

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

def main():
    # Menggunakan model FNet yang telah dilatih pada dataset SST2
    model_name = "google/fnet-base" # Sebagai contoh representasi model sejenis
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    inputs = [
        "Probably my all-time favorite movie...",
        "This film tried to be too many things all at once...",
        "If you like original gut wrenching laughter..."
    ]

    # Proses tokenisasi
    encoded_input = tokenizer(inputs, padding=True, truncation=True, return_tensors="pt")

    # Prediksi
    with torch.no_grad():
        outputs = model(**encoded_input)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

    for i, prob in enumerate(predictions):
        label = "POSITIVE" if prob[1] > prob[0] else "NEGATIVE"
        print(f"Teks: {inputs[i][:30]}... -> Label: {label} (Skor: {prob.max().item():.4f})")

if __name__ == "__main__":
    main()
```

Sebagai kesimpulan, model bidirectional mewakili kemajuan besar dalam NLP dengan menangkap konteks dari kedua arah, yang secara signifikan meningkatkan kinerja pada tugas-tugas bahasa yang kompleks. Meskipun mereka membutuhkan lebih banyak sumber daya komputasi daripada model unidirectional, kemampuan mereka untuk memahami hubungan bernuansa dalam teks membuat mereka sangat efektif untuk pemahaman bahasa alami tingkat lanjut.

## 5.2. Arsitektur BERT

Arsitektur BERT (Bidirectional Encoder Representations from Transformers) adalah model pelopor dalam NLP yang dirancang untuk memahami konteks bahasa dengan memproses urutan input secara dua arah. Berbeda dengan model Transformer yang dirancang untuk pembuatan teks (seperti GPT yang menggunakan pendekatan autoregresif), BERT adalah model *encoder-only* yang sepenuhnya didasarkan pada tumpukan encoder Transformer. Struktur *encoder-only* ini adalah kunci kemampuan BERT untuk menangkap konteks yang kaya dari kiri dan kanan kata target.

![Gambar 3: Arsitektur BERT terdiri dari Transformer Encoder.](https://lmvr.rantai.dev//images/L6A504iArDGBvUq8WSeO-2zDddUej72FLKJBEC0mK-v1.png)
**Gambar 3:** Arsitektur BERT terdiri dari Transformer Encoder.

Pada intinya, arsitektur BERT terdiri dari beberapa lapisan *self-attention* dan jaringan saraf *feedforward*. Urutan input ditokenisasi dan diubah menjadi *word embeddings* sebelum dilewatkan melalui model. Representasi ini kemudian diproses oleh beberapa lapisan *multi-head attention*, di mana setiap kata dapat memperhatikan setiap kata lainnya dalam urutan tersebut. Secara formal, untuk urutan $S = [x_1, x_2, ..., x_n]$, BERT menghitung representasi kontekstual $h_i$ untuk setiap token $x_i$:

$$ h_i = \text{TransformerEncoder}(x_i | x_1, ..., x_{i-1}, x_{i+1}, ..., x_n) $$

Pendekatan pelatihan unik BERT berpusat pada dua tujuan utama: *masked language modeling* (MLM) dan *next sentence prediction* (NSP).

1.  **Masked Language Modeling (MLM):** BERT menyamarkan sekitar 15% token secara acak dan melatih model untuk memprediksi token tersebut. Hal ini memaksa model untuk mengandalkan konteks kiri dan kanan.
2.  **Next Sentence Prediction (NSP):** Model dilatih untuk memprediksi apakah kalimat B secara logis mengikuti kalimat A dalam dokumen asli. Ini sangat berguna untuk tugas-tugas seperti menjawab pertanyaan yang melibatkan hubungan antar kalimat.

Proses pra-pelatihan pada korpus besar memungkinkan model mempelajari representasi bahasa umum yang kemudian dapat disetel halus pada tugas-tugas tertentu. Berikut adalah simulasi proses penyetelan halus (fine-tuning) BERT untuk klasifikasi urutan menggunakan PyTorch.

### Contoh Python: Fine-Tuning BERT (Simulasi)

```python
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW

def main():
    # Inisialisasi model dan tokenizer
    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # Siapkan data dummy
    texts = ["I love this product!", "This is a bad experience."]
    labels = torch.tensor([1, 0]) # 1: Positive, 0: Negative

    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=1e-5)

    # Loop pelatihan sederhana
    model.train()
    for epoch in range(3):
        optimizer.zero_grad()
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    print("Fine-tuning simulasi selesai!")

if __name__ == "__main__":
    main()
```

Selain klasifikasi, BERT sering digunakan untuk menghasilkan *sentence embeddings*. Berikut adalah cara melakukannya menggunakan model yang dioptimalkan untuk kesamaan kalimat.

### Contoh Python: Menghasilkan Sentence Embeddings dengan BERT

```python
from transformers import AutoTokenizer, AutoModel
import torch

# Fungsi untuk normalisasi L2
def normalize_l2(v):
    norm = torch.norm(v, p=2, dim=1, keepdim=True)
    return v / norm

def main():
    model_id = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)

    sentences = [
        "The cat sits outside",
        "A man is playing guitar",
        "I love pasta",
        "The new movie is awesome"
    ]

    # Tokenisasi
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")

    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)
        # Mengambil rata-rata (mean pooling) dari token embeddings
        embeddings = outputs.last_hidden_state.mean(dim=1)
        embeddings = normalize_l2(embeddings)

    print("Bentuk Tensor Embeddings:", embeddings.shape)
    
    # Menghitung kesamaan antara kalimat pertama dan kedua
    similarity = torch.mm(embeddings[0:1], embeddings[1:2].T)
    print(f"Kesamaan antara '{sentences[0]}' dan '{sentences[1]}': {similarity.item():.4f}")

if __name__ == "__main__":
    main()
```

## 5.3. Varian dan Ekstensi BERT

Beberapa varian BERT telah dikembangkan untuk meningkatkan efisiensi dan performa. Yang paling menonjol adalah RoBERTa, ALBERT, dan DistilBERT.

**RoBERTa (Robustly Optimized BERT Pretraining Approach)** memodifikasi prosedur pelatihan asli BERT dengan menghapus tugas NSP, menggunakan ukuran batch yang lebih besar, dan melatih model lebih lama pada dataset yang lebih luas.

**ALBERT (A Lite BERT)** berfokus pada pengurangan ukuran model dengan memperkenalkan pembagian parameter (*parameter sharing*) antar lapisan dan faktorisasi matriks penyematan. Secara formal, ALBERT mendekomposisi matriks penyematan $E$ menjadi dua matriks yang lebih kecil: $E = W \times V$.

**DistilBERT** menggunakan konsep distilasi model untuk menciptakan versi BERT yang lebih kecil dan cepat. Model "murid" (student) dilatih untuk meniru distribusi output dari model "guru" (teacher) yang sudah dipra-latih. DistilBERT memiliki lapisan 50% lebih sedikit tetapi mempertahankan sekitar 97% kemampuan pemahaman bahasa BERT.

### Contoh Python: Menggunakan DistilBERT untuk Inferensi

```python
from transformers import AutoTokenizer, AutoModel
import torch

def main():
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    text = "DistilBERT is a small, fast, cheap and light Transformer model."
    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
    
    # Hidden states untuk setiap token
    print("Bentuk Hidden States:", outputs.last_hidden_state.shape)

if __name__ == "__main__":
    main()
```

Berikut adalah perbandingan antara penggunaan RoBERTa dan ALBERT dalam satu skrip Python.

### Contoh Python: Perbandingan Logit RoBERTa dan ALBERT

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

def run_inference(model_name):
    print(f"Memuat model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    inputs = tokenizer("Hello, this is a test sentence.", return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.logits

def main():
    roberta_logits = run_inference("roberta-base")
    print("RoBERTa Logits:", roberta_logits)

    albert_logits = run_inference("albert-base-v2")
    print("ALBERT Logits:", albert_logits)

if __name__ == "__main__":
    main()
```

## 5.4. Fine-Tuning BERT untuk Tugas NLP Spesifik

Fine-tuning melibatkan adaptasi representasi BERT yang sudah dipra-latih ke aplikasi tertentu dengan melatihnya pada dataset khusus tugas tersebut. Selama proses ini, lapisan khusus tugas (seperti kepala klasifikasi) ditambahkan di atas model BERT.

Secara formal, untuk klasifikasi, output dari token `[CLS]` dilewatkan melalui lapisan linier dengan aktivasi softmax:

$$ \hat{y} = \text{softmax}(W h_{\text{[CLS]}} + b) $$

Salah satu tantangan utama dalam penyetelan halus adalah risiko *overfitting* pada dataset kecil. Teknik regularisasi seperti *dropout*, *early stopping*, dan *weight decay* sangat penting. Selain itu, pemilihan *learning rate* harus dilakukan dengan hati-hati; biasanya membutuhkan angka yang jauh lebih kecil (misal $2 \times 10^{-5}$) dibandingkan pelatihan dari nol.

### Contoh Python: Siklus Fine-Tuning Lengkap (PyTorch)

```python
import torch
import torch.nn as nn
from transformers import BertForSequenceClassification, BertTokenizer, AdamW

def train_bert():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).to(device)
    optimizer = AdamW(model.parameters(), lr=1e-5)

    # Dummy data
    input_ids = torch.randint(0, 30522, (32, 64)).to(device)
    labels = torch.randint(0, 2, (32,)).to(device)

    for epoch in range(3):
        model.train()
        optimizer.zero_grad()
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

if __name__ == "__main__":
    train_bert()
```

## 5.5. Aplikasi BERT dan Variannya

Keberhasilan BERT dalam berbagai aplikasi didorong oleh kemampuannya untuk memanfaatkan *transfer learning*.

**Klasifikasi Teks:** BERT sangat efektif karena mekanisme atensi dua arahnya menangkap konteks secara penuh.
**Terjemahan Mesin:** Meskipun BERT adalah encoder-only, varian seperti mBERT (Multilingual BERT) dan BART telah diadaptasi untuk tugas urutan-ke-urutan.
**Menjawab Pertanyaan (QA):** Di SQuAD, BERT memprediksi rentang teks yang mengandung jawaban dengan menghitung probabilitas token awal dan akhir:
$$ \hat{y}_{\text{start}} = \arg\max P_{\text{start}}(i), \quad \hat{y}_{\text{end}} = \arg\max P_{\text{end}}(i) $$

Berikut adalah implementasi Python untuk tugas menjawab pertanyaan menggunakan model yang dioptimalkan untuk konteks panjang (Longformer).

### Contoh Python: Menjawab Pertanyaan dengan Longformer

```python
from transformers import pipeline

def main():
    # Menyiapkan pipeline QA dengan model Longformer yang dilatih pada SQuAD
    qa_pipeline = pipeline("question-answering", model="valhalla/longformer-base-4096-finetuned-squadv1")

    context = "Jaisy lives in Jakarta. While Jaisy lives in Jakarta, Evan is in The Hague."
    
    questions = [
        "Where does Jaisy live?",
        "Where does Evan live?"
    ]

    for question in questions:
        result = qa_pipeline(question=question, context=context)
        print(f"Pertanyaan: {question}")
        print(f"Jawaban: {result['answer']} (Skor: {result['score']:.4f})\n")

if __name__ == "__main__":
    main()
```

Berikut adalah contoh penggunaan model T5 untuk peringkasan teks.

### Contoh Python: Peringkasan Teks dengan T5

```python
from transformers import pipeline

def main():
    summarizer = pipeline("summarization", model="t5-small")
    
    text = """In findings published Tuesday in Cornell University's arXiv by a team of scientists 
    from the University of Montreal and a separate report published Wednesday in Nature Astronomy..."""

    summary = summarizer(text, max_length=50, min_length=10, do_sample=False)
    print("Ringkasan T5:", summary[0]['summary_text'])

if __name__ == "__main__":
    main()
```

## 5.6. Interpretabilitas dan Eksplanabilitas Model

Memahami bagaimana model BERT sampai pada prediksinya sangat penting dalam domain berisiko tinggi seperti kesehatan atau hukum. Salah satu teknik utama adalah visualisasi atensi.

Bobot atensi menentukan seberapa besar pengaruh satu token terhadap token lainnya. Meskipun peta atensi menunjukkan fokus model, mereka tidak selalu memberikan penjelasan intuitif yang selaras dengan pemahaman manusia. Teknik lain seperti SHAP (*Shapley Additive Explanations*) memberikan kontribusi fitur pada tingkat token.

Berikut adalah kode Python untuk mengekstrak dan memvisualisasikan bobot atensi.

### Contoh Python: Visualisasi Atensi BERT

```python
import torch
from transformers import BertModel, BertTokenizer
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased', output_attentions=True)

    sentence = "The cat sat on the mat"
    inputs = tokenizer(sentence, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Ambil atensi dari lapisan pertama, head pertama
    # outputs.attentions adalah tuple (n_layers) berisi tensor [batch, heads, seq, seq]
    attention = outputs.attentions[0][0, 0].cpu().numpy()
    
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

    plt.figure(figsize=(8, 6))
    sns.heatmap(attention, xticklabels=tokens, yticklabels=tokens, annot=True, cmap='Blues')
    plt.title("Visualisasi Atensi BERT (Layer 1, Head 1)")
    plt.show()

if __name__ == "__main__":
    main()
```

## 5.7. Kesimpulan

Bab ini membekali pembaca dengan pemahaman komprehensif tentang arsitektur model bidirectional, terutama BERT. Melalui mekanisme inovatif seperti *Masked Language Modeling* dan *Next Sentence Prediction*, model-model ini telah menetapkan standar baru dalam pemahaman bahasa alami.

### 5.7.1. Pembelajaran Lebih Lanjut dengan GenAI

*   Jelaskan perbedaan mendasar antara RNN tradisional dan arsitektur Transformer.
*   Deskripsikan mekanisme *self-attention* secara rinci, termasuk formulasi Q, K, dan V.
*   Diskusikan peran *multi-head attention* dalam memungkinkan model fokus pada berbagai bagian input.
*   Apa itu *positional encoding* dan mengapa ia penting?
*   Jelajahi interaksi antara encoder dan decoder melalui *cross-attention*.
*   Analisis pentingnya *layer normalization* dan *residual connections*.
*   Diskusikan tantangan melatih model Transformer besar (biaya komputasi dan memori).
*   Jelaskan konsep *learning rate warm-up*.
*   Apa perbedaan antara model *encoder-only*, *decoder-only*, dan *full encoder-decoder*?
*   Diskusikan keuntungan skalabilitas arsitektur Transformer dibandingkan RNN.

### 5.7.2. Latihan Praktis

#### Latihan Mandiri 5.1: Mengimplementasikan Mekanisme Self-Attention
Tujuan: Memahami operasi matematika di balik atensi dengan mengimplementasikannya secara manual di Python/PyTorch.

#### Latihan Mandiri 5.2: Eksplorasi Multi-Head Attention
Tujuan: Bereksperimen dengan jumlah head yang berbeda dan menganalisis dampaknya pada akurasi model.

#### Latihan Mandiri 5.3: Implementasi Positional Encoding
Tujuan: Memvisualisasikan fungsi sinus dan kosinus yang digunakan untuk positional encoding.

#### Latihan Mandiri 5.4: Fine-Tuning BERT pada Dataset Kustom
Tujuan: Melatih model BERT pada dataset analisis sentimen lokal dan membandingkan hasilnya dengan model baseline.

#### Latihan Mandiri 5.5: Optimasi dengan Mixed Precision
Tujuan: Menggunakan teknik FP16 untuk mempercepat waktu pelatihan model Transformer.