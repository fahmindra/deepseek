---
slug: lmvr-5
title: lmvr-5
---

# Bab 5: Model Bidireksional: BERT dan Varian-variannya

> 💡 **"Atensi adalah semua yang Anda butuhkan (*Attention is all you need*). Arsitektur Transformer telah secara mendasar mengubah cara kita berpikir tentang pemodelan urutan, memungkinkan kita untuk menangani tugas bahasa yang kompleks dengan efisiensi dan akurasi yang belum pernah terjadi sebelumnya." — Ashish Vaswani**

> 📘 **Ringkasan Bab:**
> Bab 4 dari LMVR memberikan eksplorasi mendalam tentang arsitektur Transformer, sebuah model terobosan yang merevolusi pemrosesan bahasa alami dengan memungkinkan pemrosesan paralel yang efisien dan menangkap ketergantungan jarak jauh dalam teks. Bab ini mencakup komponen-komponen penting seperti self-attention, multi-head attention, dan positional encoding, menjelaskan bagaimana elemen-elemen ini bekerja sama untuk meningkatkan kemampuan model dalam memahami dan menghasilkan bahasa. Bab ini juga menggali struktur encoder-decoder, normalisasi lapisan, dan koneksi residual, menyoroti perannya dalam menstabilkan dan mengoptimalkan model. Aspek praktis dalam mengimplementasikan dan menyempurnakan (*fine-tuning*) model Transformer dalam Rust juga dibahas, menawarkan alat bagi pembaca untuk menerapkan konsep-konsep ini pada tugas-tugas NLP di dunia nyata secara efektif.

---

## 5.1. Pengantar Model Bidireksional

Dalam pemrosesan bahasa alami (NLP), model bidireksional seperti BERT (*Bidirectional Encoder Representations from Transformers*) telah merevolusi cara mesin memahami bahasa dengan menangkap konteks dari kedua arah dalam sebuah kalimat. Secara tradisional, banyak model bahasa, seperti GPT (*Generative Pretrained Transformer*), bersifat unidireksional (satu arah), yang berarti mereka menghasilkan atau memprediksi kata-kata hanya berdasarkan konteks sebelumnya. Keterbatasan ini membuat model unidireksional sulit untuk sepenuhnya memahami arti kata-kata yang bergantung pada konteks masa depan dalam kalimat tersebut. Sebagai contoh, dalam kalimat *"The bank approved the loan,"* kata *"bank"* bisa merujuk pada lembaga keuangan atau tepi sungai, dan memahami konteks dari *"approved the loan"* (konteks masa depan) sangat penting untuk menafsirkan *"bank"* dengan benar.

Model bidireksional seperti BERT mengatasi masalah ini dengan mempertimbangkan konteks kiri-ke-kanan dan kanan-ke-kiri secara bersamaan. Dalam BERT, misalnya, model belajar memprediksi kata-kata yang disamarkan (*masked words*) dengan menggunakan informasi dari kedua sisi kata target. Secara formal, untuk kalimat yang diberikan $S = [x_1, x_2, ..., x_n]$, model bidireksional menghitung representasi untuk setiap kata $x_i$ dengan memperhatikan seluruh kalimat $S$, sehingga mempertimbangkan informasi dari $x_1$ hingga $x_{i-1}$ (konteks sebelumnya) dan $x_{i+1}$ hingga $x_n$ (konteks sesudahnya). Desain ini memungkinkan BERT untuk menangkap ketergantungan bidireksional dalam bahasa, yang sangat penting untuk banyak tugas hilir seperti menjawab pertanyaan (*question answering*) dan analisis sentimen, di mana pemahaman konteks penuh sangat penting untuk interpretasi yang akurat.

![Figure 1: Prosedur keseluruhan pre-training dan fine-tuning untuk BERT (Ref: https://arxiv.org/pdf/1810.04805).](https://lmvr.rantai.dev/images/L6A504iArDGBvUq8WSeO-7pzMEIdiYe415Ay4hTs1-v1.png)

BERT dapat disempurnakan (*fine-tuned*) untuk tugas-tugas seperti MNLI (*Multi-Genre Natural Language Inference*), NER (*Named Entity Recognition*), dan SQuAD (*Stanford Question Answering Dataset*) dengan memanfaatkan sifat bidireksionalnya. Untuk MNLI, BERT dilatih untuk memahami hubungan antara pasangan kalimat dengan memperhatikan kedua kalimat secara bersamaan. Dalam NER, BERT menggunakan konteks bidireksional untuk melabeli entitas bernama secara akurat dalam sebuah kalimat (misalnya nama, lokasi). Untuk SQuAD, BERT disempurnakan untuk menjawab pertanyaan dengan memprediksi token awal dan akhir dalam sebuah teks yang paling cocok dengan kueri yang diberikan. Pendekatan bidireksional ini memungkinkan BERT unggul dalam tugas-tugas ini dengan memahami konteks tingkat kalimat secara penuh.

Secara matematis, proses ini sering dicapai menggunakan *masked language modeling* (MLM), di mana sebagian dari token masukan disamarkan, dan model dilatih untuk memprediksi token yang disamarkan tersebut menggunakan konteks sekitarnya. Untuk token yang disamarkan $x_m$, tujuannya adalah untuk memaksimalkan probabilitas $P(x_m | x_1, ..., x_{m-1}, x_{m+1}, ..., x_n)$, dengan memanfaatkan informasi dari kedua arah. Ini kontras dengan model unidireksional, yang biasanya mengoptimalkan tujuan autoregresif yang memprediksi kata berikutnya berdasarkan kata-kata sebelumnya, yaitu $P(x_m | x_1, ..., x_{m-1})$.

Konteks dalam Pemahaman Bahasa Alami sangat mendasar bagi sebagian besar tugas NLP. Model bidireksional, seperti BERT, memungkinkan pemahaman yang lebih kaya dengan memperhatikan seluruh masukan, menjadikannya lebih unggul dalam tugas-tugas yang mengandalkan pemahaman kalimat lengkap. Pengembangan BERT dan pendekatan bidireksionalnya dimotivasi oleh keterbatasan model unidireksional, terutama dalam menangkap ketergantungan jarak jauh dalam teks. Model unidireksional, seperti GPT, mengandalkan autoregresi, memprediksi token secara berurutan dari kiri ke kanan. Meskipun ini berfungsi dengan baik untuk tugas generatif seperti pembuatan teks, ia tidak sepenuhnya menangkap hubungan antar kata yang memerlukan pemahaman seluruh urutan sekaligus.

Aliran informasi dalam model bidireksional seperti BERT kontras dengan model autoregresif seperti GPT. Dalam GPT, setiap token dalam urutan hanya bergantung pada token sebelumnya. Model bidireksional, di sisi lain, menggunakan mekanisme *self-attention* yang memungkinkan setiap token untuk memperhatikan setiap token lain dalam urutan, sehingga mempelajari hubungan dan ketergantungan di seluruh kalimat.

Sebelum membahas detail implementasi model BERT, mari kita mulai dengan lapisan masukan (*input layer*). Lapisan masukan BERT terdiri dari tiga embedding utama: *token embedding*, *positional embedding*, dan *segment* (atau kalimat) *embedding*. *Token embedding* bertanggung jawab untuk mengubah setiap token dalam sebuah kalimat menjadi representasi vektor padat. BERT menggunakan tokenizer WordPiece yang memecah kata menjadi subkata. *Positional embedding* mengkodekan posisi setiap token dalam urutan, memastikan model menyadari urutan token karena BERT memproses token secara paralel. Terakhir, *segment embedding* membantu BERT membedakan antara token dari kalimat yang berbeda dalam tugas yang melibatkan pasangan kalimat. Untuk setiap token, embedding yang sesuai dari ketiga lapisan tersebut dijumlahkan untuk membentuk representasi masukan lengkap.

![Figure 2: Ilustrasi lapisan masukan model BERT.](https://lmvr.rantai.dev/images/L6A504iArDGBvUq8WSeO-8Tv8PqU5Q2zFhtK8hIds-v1.png)

Lapisan klasifikasi mengambil representasi *pooled* final dan mengubahnya menjadi logit untuk klasifikasi biner. Model dilatih menggunakan kerugian *cross-entropy* dan parameternya diperbarui menggunakan optimizer Adam. Dalam praktiknya, dataset yang telah diproses sebelumnya seperti IMDB (untuk analisis sentimen) akan digunakan setelah ditokenisasi.

Membandingkan model bidireksional seperti BERT dengan model unidireksional seperti GPT menyoroti kekuatan bidireksionalitas dalam tugas yang memerlukan pemahaman konteks yang komprehensif. Namun, melatih model bidireksional menghadirkan tantangan komputasi karena memproses seluruh urutan sekaligus membutuhkan lebih banyak memori. Selain itu, tujuan MLM BERT memperlambat pelatihan dibandingkan dengan model autoregresif yang memprediksi satu token pada satu waktu.

Berikut adalah contoh penggunaan Python (sebagai pengganti `rust-bert`) untuk melakukan analisis sentimen menggunakan model BERT:

```python
# Contoh Python menggunakan library transformers
from transformers import pipeline

def main():
    # Menyiapkan classifier sentimen
    # Pipeline ini secara otomatis mengunduh model BERT default untuk sentiment-analysis
    sentiment_classifier = pipeline("sentiment-analysis")

    # Mendefinisikan input
    inputs = [
        "Probably my all-time favorite movie, a story of selflessness, sacrifice and dedication to a noble cause, but it's not preachy or boring.",
        "This film tried to be too many things all at once: stinging political satire, Hollywood blockbuster, sappy romantic comedy, family values promo...",
        "If you like original gut wrenching laughter you will like this movie. If you are young or old then you will love this movie, hell even my mom liked it.",
    ]

    # Menjalankan model
    outputs = sentiment_classifier(inputs)
    for text, sentiment in zip(inputs, outputs):
        print(f"Text: {text[:50]}...")
        print(f"Sentiment: {sentiment}\n")

if __name__ == "__main__":
    main()
```

Mari kita ambil contoh lain untuk analisis sentimen menggunakan model FNet (model alternatif yang lebih efisien daripada Transformer standar):

```python
# Contoh Python menggunakan model FNet untuk analisis sentimen
from transformers import pipeline

def main():
    # Menggunakan model FNet yang sudah dilatih pada SST2
    model_name = "google/fnet-base" # Catatan: Pastikan model checkpoint tersedia untuk tugas ini
    
    # Untuk kemudahan demonstrasi, kita gunakan pipeline standar dengan spesifikasi model
    try:
        classifier = pipeline("sentiment-analysis", model="gchhablani/fnet-base-sst2")
        
        inputs = [
            "Probably my all-time favorite movie, a story of selflessness, sacrifice and dedication to a noble cause.",
            "This film tried to be too many things all at once and failed.",
            "If you like original gut wrenching laughter you will like this movie."
        ]

        outputs = classifier(inputs)
        for sentiment in outputs:
            print(sentiment)
    except Exception as e:
        print(f"Error: {e}. Model mungkin memerlukan konfigurasi manual lebih lanjut.")

if __name__ == "__main__":
    main()
```

Sebagai kesimpulan, model bidireksional mewakili kemajuan besar dalam NLP dengan menangkap konteks dari kedua arah. Meskipun membutuhkan lebih banyak sumber daya komputasi, kemampuannya untuk memahami hubungan nuansa dalam teks membuatnya sangat efektif untuk tugas-tugas seperti menjawab pertanyaan dan analisis sentimen.

---

## 5.2. Arsitektur BERT

Arsitektur BERT adalah model pelopor dalam NLP, dirancang untuk memahami konteks bahasa dengan memproses urutan masukan secara bidireksional. Berbeda dengan model Transformer yang dirancang untuk pembuatan teks (seperti GPT yang menggunakan pendekatan autoregresif), BERT adalah model *encoder-only* yang sepenuhnya didasarkan pada tumpukan encoder Transformer. Struktur *encoder-only* ini adalah kunci kemampuan BERT untuk menangkap konteks kaya dari sisi kiri dan kanan kata target.

![Figure 3: Arsitektur BERT terdiri dari Transformer Encoder.](https://lmvr.rantai.dev/images/L6A504iArDGBvUq8WSeO-2zDddUej72FLKJBEC0mK-v1.png)

Secara formal, untuk urutan $S = [x_1, x_2, ..., x_n]$, BERT menghitung representasi kontekstual $h_i$ untuk setiap token $x_i$:

$$ h_i = \text{TransformerEncoder}(x_i | x_1, ..., x_{i-1}, x_{i+1}, ..., x_n) $$

Pendekatan pelatihan unik BERT berpusat pada dua tujuan utama: *masked language modeling* (MLM) dan *next sentence prediction* (NSP).
- **MLM:** BERT menyamarkan sekitar 15% token dan melatih model untuk memprediksinya. Secara matematis, model memaksimalkan $P(x_m | x_1, ..., x_{m-1}, x_{m+1}, ..., x_n)$.
- **NSP:** Model mempelajari hubungan antar kalimat dengan memprediksi apakah kalimat B mengikuti kalimat A dalam dokumen asli.

Keberhasilan BERT sebagian besar dikaitkan dengan konteks bidireksional dan MLM-nya. Berikut adalah contoh Python untuk melakukan *fine-tuning* sederhana pada model BERT untuk klasifikasi urutan (sebagai pengganti contoh Rust `BertForSequenceClassification`):

```python
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertForSequenceClassification, BertTokenizer, AdamW

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "bert-base-uncased"
    
    # Load model dan tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(device)

    # Dummy data
    input_ids = torch.randint(0, tokenizer.vocab_size, (8, 128)).to(device)
    attention_mask = torch.ones((8, 128)).to(device)
    labels = torch.randint(0, 2, (8,)).to(device)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=1e-5)

    # Training loop sederhana
    model.train()
    for epoch in range(3):
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        print(f"Epoch: {epoch + 1}, Loss: {loss.item():.4f}")

    print("Fine-tuning selesai!")

if __name__ == "__main__":
    main()
```

Berikut adalah contoh lain untuk menghasilkan embedding kalimat menggunakan model BERT (MiniLM) di Python:

```python
from sentence_transformers import SentenceTransformer, util

def main():
    # Memuat model MiniLM yang efisien
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
    
    print(f"Generated embeddings shape: {embeddings.shape}")

    # Menghitung kesamaan kosinus antar pasangan kalimat
    cosine_scores = util.cos_sim(embeddings, embeddings)

    # Mencari pasangan dengan skor tertinggi (selain diri sendiri)
    pairs = []
    for i in range(len(sentences)):
        for j in range(i + i, len(sentences)):
            pairs.append({'index': [i, j], 'score': cosine_scores[i][j]})

    # Urutkan berdasarkan skor
    pairs = sorted(pairs, key=lambda x: x['score'], reverse=True)

    print("\nTop 5 similar sentence pairs:")
    for pair in pairs[:5]:
        i, j = pair['index']
        print(f"Score: {pair['score']:.2f} | '{sentences[i]}' VS '{sentences[j]}'")

if __name__ == "__main__":
    main()
```

Dalam beberapa tahun terakhir, varian BERT seperti RoBERTa, DistilBERT, dan ALBERT telah diperkenalkan, masing-masing meningkatkan model BERT asli dengan cara yang berbeda. Sebagai contoh, RoBERTa memodifikasi prosedur pre-training dengan menghapus NSP, sementara DistilBERT mengurangi ukuran model agar lebih efisien.

---

## 5.3. Varian dan Ekstensi BERT

Beberapa varian BERT telah dikembangkan untuk meningkatkan efisiensi, proses pelatihan, dan performa keseluruhan.
- **RoBERTa:** Menghapus tugas NSP, menggunakan ukuran batch yang lebih besar, dan melatih lebih lama pada data yang lebih banyak.
- **ALBERT:** Fokus pada pengurangan parameter melalui pembagian parameter antar lapisan dan faktorisasi matriks embedding. $E = W \times V$ di mana $V$ jauh lebih kecil.
- **DistilBERT:** Menggunakan distilasi model di mana model "murid" yang lebih kecil dilatih untuk meniru model "guru" yang besar. Kerugian distilasinya adalah:
  $$ \text{Distillation loss} = \alpha \times L_{\text{soft}}(S, T) + (1 - \alpha) \times L_{\text{hard}}(S, y) $$

Berikut adalah contoh penggunaan Python untuk menjalankan model DistilBERT:

```python
from transformers import AutoTokenizer, AutoModel
import torch

def main():
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModel.from_pretrained("distilbert-base-uncased")

    inputs = tokenizer("Example prompt to encode", return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # hidden_state berisi embedding untuk setiap token
    print(f"DistilBERT hidden states shape: {outputs.last_hidden_state.shape}")

if __name__ == "__main__":
    main()
```

Berikut adalah contoh menjalankan klasifikasi urutan menggunakan RoBERTa dan ALBERT di Python:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

def run_model(model_name, text):
    print(f"\nLoading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    print(f"Logits: {logits}")

def main():
    sample_text = "Rust and Python are great for AI."
    run_model("roberta-base", sample_text)
    run_model("albert-base-v2", sample_text)

if __name__ == "__main__":
    main()
```

---

## 5.4. Fine-Tuning BERT untuk Tugas NLP Spesifik

*Fine-tuning* melibatkan penambahan lapisan khusus tugas (seperti kepala klasifikasi) di atas model BERT yang sudah dilatih. Untuk klasifikasi, output dari token `[CLS]` dilewatkan melalui lapisan linear dengan aktivasi softmax:

$$ \hat{y} = \text{softmax}(W h_{\text{[CLS]}} + b) $$

Selama *fine-tuning*, sangat penting untuk memilih tingkat pembelajaran (*learning rate*) yang kecil (misalnya 2e-5 atau 5e-5) untuk menjaga pengetahuan bahasa umum yang sudah dipelajari.

Berikut adalah contoh prapemrosesan manual untuk *fine-tuning* di Python:

```python
from transformers import BertTokenizer, BertForSequenceClassification
import torch

def main():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    
    text = "Example prompt to encode"
    inputs = tokenizer(text, padding='max_length', truncation=True, max_length=64, return_tensors="pt")
    
    # Forward pass
    outputs = model(**inputs)
    print(f"Logits: {outputs.logits}")

if __name__ == "__main__":
    main()
```

---

## 5.5. Aplikasi BERT dan Varian-variannya

BERT telah diterapkan pada berbagai tugas:
1.  **Klasifikasi Teks:** Menggunakan token `[CLS]` untuk label kategori.
2.  **Penerjemahan Mesin:** Meskipun BERT adalah encoder-only, varian seperti BART (bidireksional + autoregresif) digunakan untuk ini.
3.  **Menjawab Pertanyaan (QA):** Memprediksi probabilitas posisi awal dan akhir jawaban dalam teks.
    $$ \hat{y}_{\text{start}} = \arg\max P_{\text{start}}(i), \quad \hat{y}_{\text{end}} = \arg\max P_{\text{end}}(i) $$

Berikut adalah implementasi Question Answering (QA) menggunakan Python:

```python
from transformers import pipeline

def main():
    # Menyiapkan pipeline QA
    qa_model = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

    context_1 = "Jaisy lives in Jakarta"
    question_1 = "Where does Jaisy live?"
    
    context_2 = "While Jaisy lives in Jakarta, Evan is in The Hague."
    question_2 = "Where does Evan live?"

    res1 = qa_model(question=question_1, context=context_1)
    res2 = qa_model(question=question_2, context=context_2)

    print(f"Answer 1: {res1['answer']}")
    print(f"Answer 2: {res2['answer']}")

if __name__ == "__main__":
    main()
```

Berikut adalah contoh peringkasan menggunakan model T5 di Python:

```python
from transformers import pipeline

def main():
    summarizer = pipeline("summarization", model="t5-small")
    
    text = """In findings published Tuesday in Cornell University's arXiv by a team of scientists 
    from the University of Montreal, the presence of water vapour was confirmed in the atmosphere 
    of K2-18b, a planet circling a star in the constellation Leo..."""
    
    summary = summarizer(text, max_length=50, min_length=10, do_sample=False)
    print(f"Summary: {summary[0]['summary_text']}")

if __name__ == "__main__":
    main()
```

Berikut adalah contoh penerjemahan multibahasa menggunakan T5 di Python:

```python
from transformers import T5Attributes, T5ForConditionalGeneration, T5Tokenizer

def main():
    model_name = "t5-base"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    sentence = "translate English to French: This sentence will be translated."
    
    inputs = tokenizer(sentence, return_tensors="pt")
    outputs = model.generate(inputs.input_ids)
    
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()
```

---

## 5.6. Interpretabilitas dan Eksplanabilitas Model

Mengingat BERT sering digunakan dalam domain kritis, memahami "mengapa" model membuat prediksi tertentu sangatlah penting. Salah satu teknik utamanya adalah visualisasi atensi.

$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) $$

Visualisasi ini membantu praktisi memahami kata mana yang diprioritaskan oleh model. Selain itu, teknik seperti SHAP (*Shapley Additive Explanations*) dan LIME dapat digunakan untuk menganalisis kontribusi setiap token terhadap prediksi final.

Berikut adalah simulasi cara memplot matriks atensi sederhana menggunakan Python:

```python
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def main():
    # Simulasi bobot atensi (misal 7x7 untuk 7 token)
    tokens = ["[CLS]", "the", "cat", "sat", "on", "mat", "[SEP]"]
    attention_weights = np.random.rand(7, 7)
    
    # Normalisasi baris agar jumlahnya 1 (seperti softmax)
    attention_weights /= attention_weights.sum(axis=1, keepdims=True)

    plt.figure(figsize=(8, 6))
    sns.heatmap(attention_weights, xticklabels=tokens, yticklabels=tokens, cmap="Blues", annot=True)
    plt.title("Simulasi Visualisasi Atensi BERT")
    plt.xlabel("Key Tokens")
    plt.ylabel("Query Tokens")
    plt.show()

if __name__ == "__main__":
    main()
```

---

## 5.7. Kesimpulan

Bab 5 membekali pembaca dengan pemahaman komprehensif tentang arsitektur model bidireksional, menekankan mekanisme inovatifnya dalam memproses bahasa. Dengan menguasai konsep-konsep ini, pembaca akan siap untuk membangun dan mengoptimalkan model yang mendorong batas-batas pemrosesan bahasa alami.

### 5.7.1. Pembelajaran Lebih Lanjut dengan GenAI

- Jelaskan perbedaan mendasar antara RNN tradisional dan arsitektur Transformer (BERT).
- Deskripsikan mekanisme *self-attention* secara detail (matematika Q, K, V).
- Diskusikan peran *multi-head attention* dalam menangkap pola masukan yang beragam.
- Apa itu *positional encoding* dan mengapa itu esensial?
- Analisis pentingnya normalisasi lapisan dan koneksi residual.

### 5.7.2. Praktik Mandiri

- **Latihan 5.1:** Implementasikan mekanisme *self-attention* dari nol dan analisis kompleksitas komputasinya.
- **Latihan 5.2:** Eksplorasi dan optimalkan *multi-head attention* dengan bereksperimen pada jumlah kepala yang berbeda.
- **Latihan 5.3:** Implementasikan *positional encoding* sinusoidal dan bandingkan dengan metode *learned encoding*.
- **Latihan 5.4:** Lakukan *fine-tuning* pada model Transformer terlatih (BERT/DistilBERT) untuk tugas pengenalan entitas bernama (NER).
- **Latihan 5.5:** Optimalkan pelatihan Transformer dengan *mixed precision* dan *learning rate scheduling*.