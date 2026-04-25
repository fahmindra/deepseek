---
slug: lmvr-11
title: lmvr-11
---
# Bab 11: Retrieval-Augmented Generation (RAG)

> **"Masa depan AI terletak pada penggabungan pengambilan (retrieval) dan pembuatan (generation) untuk menciptakan sistem yang berpengetahuan luas sekaligus sadar konteks, meningkatkan kemampuan mereka dalam menghasilkan informasi yang akurat dan relevan secara waktu nyata." — Fei-Fei Li**

Bab 11 dari LMVR memberikan eksplorasi mendalam tentang Retrieval-Augmented Generation (RAG) dan implementasinya. Bab ini dimulai dengan memperkenalkan RAG, menjelaskan bagaimana ia menggabungkan model berbasis pencarian dan model generatif untuk meningkatkan relevansi serta akurasi teks yang dihasilkan. Bab ini membahas pengaturan lingkungan pengembangan untuk RAG, mengimplementasikan komponen retriever (pengambil) dan generator (pembangkit), serta mengintegrasikannya ke dalam sistem yang kohesif. Bab ini juga menggali penyetelan halus dan optimalisasi model RAG untuk tugas-tugas tertentu, menyebarkannya di berbagai lingkungan, dan mengatasi tantangan seperti skalabilitas dan latensi pengambilan. Terakhir, bab ini mengeksplorasi arah masa depan RAG, termasuk tren yang muncul dan pertimbangan etis, menawarkan kerangka kerja komprehensif untuk membangun sistem RAG yang tangguh.

## 11.1. Pengantar Retrieval-Augmented Generation (RAG)

Retrieval-Augmented Generation (RAG) adalah teknik NLP tingkat lanjut yang memperkuat akurasi faktual, relevansi kontekstual, dan adaptabilitas teks yang dihasilkan dengan mengintegrasikan pendekatan berbasis pengambilan dan generatif. Metodologi ini berbeda dari model generatif tradisional, yang hanya mengandalkan parameter pra-terlatih, dengan menggabungkan mekanisme pengambilan yang mengakses sumber pengetahuan eksternal non-parametrik. Dalam RAG, model secara dinamis mengambil informasi terkait dari korpus skala besar, basis data, atau basis pengetahuan dan menggunakan konteks ini untuk memandu serta menyempurnakan proses generatif. Secara formal, RAG memanfaatkan dua komponen utama: sebuah *retriever* $R(q, D)$ dan sebuah *generator* $G(y | x, c)$. Retriever memberi skor pada dokumen $d \in D$ berdasarkan relevansinya dengan kueri masukan $q$, memilih sekumpulan dokumen top-$k$ ($c$) yang memaksimalkan $P(d | q)$. Dokumen-dokumen ini digabungkan dengan input $x$ untuk membentuk konteks $c = \{d_1, d_2, \dots, d_k\}$ yang diteruskan ke generator $G(y | x, c)$ untuk menghasilkan output akhir $y$. Kerangka kerja ini memberi model RAG akses dinamis ke informasi eksternal, menjembatani celah dalam memori parametrik model dan meningkatkan kemampuannya untuk menghasilkan tanggapan yang didasarkan pada konten terbaru dan terspesialisasi.

![Gambar 1: Konsep kunci dalam metode RAG.](https://lmvr.rantai.dev//images/L6A504iArDGBvUq8WSeO-Y8tgbXYP2svtGvPIbDQI-v1.png)
**Gambar 1:** Konsep kunci dalam metode RAG.

RAG unggul dalam aplikasi di mana akurasi informasi, kebaruan, dan spesifisitas sangat penting, menjadikannya ideal untuk bidang-bidang seperti tanya jawab domain terbuka, dialog berbasis pengetahuan, verifikasi fakta, dan peringkasan teknis. Dalam tanya jawab domain terbuka, RAG mengungguli model tradisional dengan secara efisien mempersempit korpus yang luas untuk mengidentifikasi dokumen yang paling relevan dengan kueri. Generator kemudian memanfaatkan konteks yang ditargetkan ini untuk menyusun tanggapan yang spesifik dan akurat. Kombinasi ini membantu memitigasi halusinasi—di mana model generatif menghasilkan informasi yang salah namun terdengar meyakinkan—dengan mendasarkan tanggapan pada data faktual. Dalam verifikasi fakta dan bidang teknis seperti domain biomedis atau hukum, RAG memungkinkan model untuk merujuk pada sumber yang terverifikasi, memastikan bahwa tanggapan yang dihasilkan dapat diandalkan dan secara kontekstual diinformasikan oleh temuan atau peraturan terbaru.

Dalam dialog berbasis pengetahuan, RAG meningkatkan konsistensi dan koherensi percakapan dengan mengambil pertukaran dialog masa lalu yang relevan secara kontekstual atau basis pengetahuan terkait. Fitur ini sangat berharga dalam layanan pelanggan, di mana mempertahankan aliran dialog yang kohesif sangat penting. Untuk menyeimbangkan latensi respons dengan akurasi pengambilan, model RAG dapat menggunakan retriever yang efisien, seringkali berdasarkan representasi vektor padat (dense vector representations) untuk mengidentifikasi dokumen relevan dengan cepat. Namun, integrasi yang efektif antara retriever dan generator sangatlah penting, karena pengambilan yang tidak relevan dapat merusak kefasihan respons. Penyetelan halus dan pengujian ketat sering kali diperlukan agar informasi yang diambil selaras dengan teks yang dihasilkan.

Peningkatan terbaru dalam pendekatan RAG melibatkan kombinasinya dengan model bahasa besar (LLM), seperti ChatGPT atau LLaMA, di mana RAG memperkuat kemampuan generatif model-model ini dengan menyediakan konteks terbaru yang diambil dari basis data eksternal. Pendekatan hibrida ini memanfaatkan akurasi berbasis pengambilan RAG dan kefasihan bahasa alami LLM. Selain itu, sistem RAG dapat terus mengakses sumber eksternal tanpa pelatihan ulang, memungkinkan respons dinamis yang mencerminkan pengetahuan terbaru yang tersedia.

Berikut adalah kode Python yang mendemonstrasikan pipeline RAG yang disederhanakan menggunakan library `langchain`.

### Contoh Python: Pipeline RAG Sederhana

```python
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 1. Simulator Retriever: Mencari dokumen relevan secara sederhana
class SimpleRetriever:
    def __init__(self):
        self.knowledge_base = [
            "Rust adalah bahasa pemrograman sistem yang berfokus pada keamanan dan konkurensi.",
            "Penulis buku '20,000 Leagues Under the Sea' adalah Jules Verne.",
            "Langchain menyediakan alat untuk membangun aplikasi menggunakan model bahasa besar."
        ]

    def retrieve(self, query):
        # Simulasi pencarian berbasis kata kunci sederhana
        return [doc for doc in self.knowledge_base if query.lower() in doc.lower()]

def main():
    # Inisialisasi LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    retriever = SimpleRetriever()

    # Kueri pengguna
    query = "penulis 20,000 Leagues Under the Sea"
    retrieved_docs = retriever.retrieve("20,000 Leagues")
    context = " ".join(retrieved_docs)

    # Membuat Prompt yang diperkaya dengan konteks
    template = """Anda adalah asisten yang sangat berpengetahuan. 
    Gunakan informasi berikut untuk menjawab pertanyaan.
    
    Konteks: {context}
    Pertanyaan: {question}
    
    Jawaban:"""
    
    prompt = ChatPromptTemplate.from_template(template)

    # Rantai RAG
    rag_chain = (
        {"context": lambda x: context, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Menjalankan kueri
    result = rag_chain.invoke("Siapa penulis buku 20,000 Leagues Under the Sea?")
    print("Hasil RAG:", result)

if __name__ == "__main__":
    # main() # Memerlukan API key OpenAI
    pass
```

Kode di atas mendefinisikan `SimpleRetriever` dengan metode `retrieve` yang melakukan pencocokan kata kunci sederhana untuk mengembalikan dokumen yang relevan. `rag_chain` menggabungkan konteks yang diambil ke dalam prompt sebelum mengirimkannya ke LLM untuk mendapatkan jawaban yang akurat.

## 11.2. Menyiapkan Lingkungan untuk RAG

Membangun sistem RAG yang berkinerja tinggi memerlukan integrasi beberapa alat untuk pemrosesan teks, pengindeksan, dan inferensi model.

### 11.2.1. Library Tokenizer

Langkah pertama adalah prapemrosesan teks. Library `tokenizers` menyediakan implementasi untuk banyak tokenizer yang umum digunakan seperti Byte-Pair Encoding (BPE) dan WordPiece. Pipeline tokenisasi terdiri dari empat komponen utama: Normalizer (standarisasi teks), PreTokenizer (pemecahan teks awal), Model (pemetaan ke unit subkata), dan PostProcessor (penambahan token khusus seperti `[CLS]` atau `[SEP]`).

Berikut adalah versi Python menggunakan library `transformers` untuk mengunduh dan memuat tokenizer.

### Contoh Python: Memuat Tokenizer Pra-terlatih

```python
from transformers import AutoTokenizer

def main():
    # Mengunduh dan memuat tokenizer BERT
    model_name = "bert-base-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Melakukan encoding teks
    text = "Hey there! Rust and Python are cool."
    encoding = tokenizer.encode(text)
    tokens = tokenizer.convert_ids_to_tokens(encoding)

    print(f"Token: {tokens}")
    print(f"ID Token: {encoding}")

if __name__ == "__main__":
    main()
```

### 11.2.2. Mesin Pencari (Search Engine)

Untuk komponen pengambilan (retrieval), kita memerlukan mesin pencari yang efisien. Jika di Rust menggunakan `Tantivy`, di Python kita bisa menggunakan simulator sederhana atau library seperti `Whoosh` atau `rank_bm25` untuk mencari dokumen berdasarkan skor BM25.

### Contoh Python: Pengindeksan dan Pencarian BM25 Sederhana

```python
from rank_bm25 import BM25Okapi

def main():
    corpus = [
        "Rust for System Programming: Rust offers safety and concurrency.",
        "Advantages of Rust: Rust is known for its memory safety and efficiency.",
        "Introduction to Python: Python is great for data science and AI."
    ]
    
    tokenized_corpus = [doc.split(" ") for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)

    query = "Rust programming safety"
    tokenized_query = query.split(" ")
    
    # Mendapatkan dokumen terbaik
    top_n = bm25.get_top_n(tokenized_query, corpus, n=2)
    print("Hasil Pencarian Relevan:")
    for doc in top_n:
        print(f"- {doc}")

if __name__ == "__main__":
    main()
```

### 11.2.3. Inferensi Model Bahasa

Untuk langkah pembuatan (generation), kita menggunakan model transformer. Python sangat kaya dengan dukungan model melalui library `transformers` dan `torch`.

### Contoh Python: Inferensi Klasifikasi Gambar (SqueezeNet)

```python
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

def main():
    # Memuat model SqueezeNet pra-terlatih
    model = models.squeezenet1_1(pretrained=True)
    model.eval()

    # Preprocessing gambar
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Simulasi input gambar (Red image)
    img = Image.new('RGB', (224, 224), color='red')
    input_tensor = preprocess(img).unsqueeze(0)

    # Forward pass
    with torch.no_grad():
        output = model(input_tensor)
    
    # Softmax untuk mendapatkan probabilitas
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    print(f"Top Probabilitas: {torch.max(probabilities).item():.4f}")

if __name__ == "__main__":
    main()
```

## 11.3. Mengimplementasikan Komponen Retriever

Komponen retriever dalam sistem RAG sangat kritis karena ia memilih informasi yang paling relevan untuk diberikan kepada generator. Metode tradisional seperti BM25 menggunakan representasi jarang (*sparse*), sedangkan metode modern menggunakan pengambilan padat (*dense retrieval*) dengan neural embeddings.

BM25 menghitung skor relevansi berdasarkan frekuensi kata kunci (TF) dan kebalikan frekuensi dokumen (IDF). Formulanya adalah:

$$ \text{BM25}(q, d) = \sum_{t \in q} \frac{\text{IDF}(t) \cdot f(t, d) \cdot (k_1 + 1)}{f(t, d) + k_1 \cdot (1 - b + b \cdot \frac{|d|}{\text{avgdl}})} $$

## 11.4. Mengimplementasikan Komponen Generator

Generator bertanggung jawab mensintesis informasi yang diambil menjadi teks yang fasih. Model seperti GPT, BART, dan T5 sangat cocok untuk ini. Tantangannya adalah pengkondisian generator pada informasi yang diambil agar tetap koheren. Teknik umum yang digunakan adalah penggabungan (concatenation) kueri dan konteks: $G(c, r)$.

## 11.5. Mengintegrasikan Komponen Retriever dan Generator

Interaksi antara retriever dan generator adalah inti dari sistem RAG. Alirannya dimulai dari pemrosesan kueri pengguna, pengambilan dokumen, hingga pembuatan respons akhir.

![Gambar 2: Dari kueri pengguna ke respons kontekstual.](https://lmvr.rantai.dev//images/L6A504iArDGBvUq8WSeO-oihURZpS6SKfqRGrBAZJ-v1.png)
**Gambar 2:** Dari kueri pengguna ke respons kontekstual.

Strategi integrasi meliputi integrasi pipeline (berurutan) dan pelatihan ujung-ke-ujung (*end-to-end*). Integrasi pipeline bersifat modular dan lebih cepat diimplementasikan.

## 11.6. Penyetelan Halus dan Optimasi Sistem RAG

Penyetelan halus (fine-tuning) menyelaraskan retriever dengan kosakata domain target dan melatih generator untuk merespons sesuai batasan bahasa domain tersebut.

![Gambar 3: Dari inisialisasi hingga output RAG yang dioptimalkan.](https://lmvr.rantai.dev//images/L6A504iArDGBvUq8WSeO-UVVf4a2wCxM7YuHcd9Uq-v1.png)
**Gambar 3:** Dari inisialisasi hingga output RAG yang dioptimalkan.

Optimasi tambahan seperti kuantisasi (mengubah parameter ke presisi rendah) dan pemangkasan (*pruning*) sangat berharga untuk penyebaran di perangkat dengan daya rendah.

## 11.7. Menyebarkan Sistem RAG

Penyebaran sistem RAG harus mempertimbangkan skalabilitas dan latensi. Karena sistem ini bergantung pada pengambilan dan sintesis, setiap keterlambatan dalam pengambilan akan berdampak pada waktu respons keseluruhan.

![Gambar 4: Alur penyebaran RAG.](https://lmvr.rantai.dev//images/L6A504iArDGBvUq8WSeO-5t54iKgBsb4W3J5foMjB-v1.png)
**Gambar 4:** Alur penyebaran RAG.

Berikut adalah implementasi sederhana server RAG menggunakan `FastAPI` di Python.

### Contoh Python: Server RAG dengan FastAPI

```python
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

app = FastAPI()

class QueryRequest(BaseModel):
    question: str

@app.post("/rag")
async def rag_endpoint(request: QueryRequest):
    # Tahap 1: Pengambilan (Simulasi)
    context = "Informasi relevan yang diambil dari database..."
    
    # Tahap 2: Pembuatan (Simulasi)
    response_text = f"Berdasarkan konteks: '{context}', jawabannya untuk '{request.question}' adalah..."
    
    return {"response": response_text}

if __name__ == "__main__":
    # uvicorn.run(app, host="0.0.0.0", port=8000)
    pass
```

## 11.8. Tantangan dan Arah Masa Depan dalam RAG

Tantangan utama RAG meliputi skalabilitas dalam menangani volume data besar dan akurasi pengambilan. Jika presisi pengambilan rendah, generator akan menerima informasi yang tidak relevan, yang merusak kualitas jawaban.

![Gambar 5: Tantangan dan arah masa depan RAG.](https://lmvr.rantai.dev//images/L6A504iArDGBvUq8WSeO-qMQ4kmQzSbNCqDWRbUNj-v1.png)
**Gambar 5:** Tantangan dan arah masa depan RAG.

Tren masa depan meliputi RAG Multimodal (menggabungkan teks dengan gambar/audio) dan sistem RAG yang sepenuhnya dapat dideferensiasi (joint optimization). Etika juga menjadi pusat perhatian, terutama risiko penyebaran misinformasi jika data dasar pengambilan mengandung bias.

## 11.9. Kesimpulan

Bab 11 membekali pembaca dengan pengetahuan untuk membangun dan menyebarkan sistem RAG. Dengan menguasai teknik pengambilan dan generatif, pembaca dapat menciptakan sistem AI canggih yang memberikan output akurat dan relevan secara kontekstual di berbagai domain.

### 11.9.1. Pembelajaran Lebih Lanjut dengan GenAI

- Jelaskan prinsip inti RAG dan perbedaannya dengan model generatif tradisional.
- Diskusikan peran komponen retriever dalam sistem RAG.
- Bandingkan metode pengambilan jarang (sparse) dan padat (dense).
- Analisis dampak latensi pengambilan terhadap performa keseluruhan sistem.
- Jelajahi proses fine-tuning sistem RAG untuk domain hukum atau medis.
- Apa saja pertimbangan etis utama saat membangun sistem RAG?
- Diskusikan manfaat integrasi RAG dengan LLM modern seperti Llama 3.

### 11.9.2. Latihan Praktis

#### Latihan Mandiri 11.1: Membandingkan Pengambilan Jarang vs Padat
Tujuan: Mengimplementasikan BM25 dan Dense Retrieval (menggunakan embeddings) pada korpus yang sama dan membandingkan akurasinya.

#### Latihan Mandiri 11.2: Fine-Tuning RAG untuk Domain Spesifik
Tujuan: Menyesuaikan retriever dan generator untuk memahami dokumen teknis internal perusahaan.

#### Latihan Mandiri 11.3: Pengujian Latensi Real-Time
Tujuan: Mengukur waktu respons sistem RAG pada berbagai ukuran basis data pengetahuan.

#### Latihan Mandiri 11.4: Deteksi Data di Luar Distribusi (OOD)
Tujuan: Mengimplementasikan mekanisme yang memberi tahu pengguna jika kueri mereka tidak dapat dijawab oleh basis pengetahuan yang ada.

#### Latihan Mandiri 11.5: Implementasi RAG Hybrid
Tujuan: Menggabungkan skor dari pencarian kata kunci tradisional dan pencarian semantik untuk hasil pengambilan yang lebih kuat.