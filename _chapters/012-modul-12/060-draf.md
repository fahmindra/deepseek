---
slug: modul-12-6
title: modul-12-6
---
Berikut adalah isi konten dari subtopik 8.6 yang telah dirapikan ke dalam format Markdown sesuai hirarki tanpa mengurangi isinya:

---

# 8.6 Tools Evaluasi Aplikasi: Framework Otomatisasi

Di subtopik sebelumnya (8.4 & 8.5), kita belajar membuat "Juri AI" dari nol (*scratch*) menggunakan OpenAI API. Itu bagus untuk pemahaman dan pembelajaran dasar, namun untuk skala produksi terutama untuk sistem RAG yang kompleks yaitu menulis *prompt* evaluasi manual untuk setiap metrik (misal: *Faithfulness, Context Recall, Answer Relevance*) sangatlah melelahkan dan tidak standar. Di sinilah Framework Evaluasi masuk.

Dalam pengembangan *software* konvensional, kita memiliki *pytest* atau *JUnit*. Kita tidak menulis logika *testing framework* sendiri setiap kali coding.

Dalam era LLM, ekosistem serupa telah muncul. *Tools* seperti **RAGAS (Retrieval Augmented Generation Assessment)**, **DeepEval**, dan **TruLens** telah menjadi standar industri untuk "menguji" aplikasi LLM. Mereka membungkus logika *LLM-as-a-Judge* yang rumit menjadi fungsi sederhana yang bisa dipanggil, lengkap dengan metrik yang sudah terkalibrasi secara akademis.

Di subtopik ini, kita akan fokus pada **RAGAS**, framework paling populer untuk mengaudit sistem RAG, dan bagaimana menggunakannya untuk mendeteksi halusinasi secara otomatis.

## 8.6.1 Metrik RAG: Tritunggal Evaluasi (RAG Triad)

Framework seperti RAGAS tidak hanya memberi skor "Bagus/Jelek". Mereka memecah performa RAG menjadi tiga komponen terukur (dikenal sebagai **RAG Triad**):

**1. Faithfulness (Kesetiaan/Faktualitas):**
*   **Pertanyaan:** "Apakah jawaban model berasal **hanya** dari dokumen yang diambil (Context), atau model mengarang bebas (halusinasi)?"
*   **Logika Juri:** Bandingkan `Answer` vs `Retrieved Context`.

**2. Answer Relevance (Relevansi Jawaban):**
*   **Pertanyaan:** "Apakah jawaban model benar-benar **menjawab** pertanyaan pengguna?"
*   **Logika Juri:** Bandingkan `Answer` vs `User Question`.

**3. Context Precision/Recall (Kualitas Retriever):**
*   **Pertanyaan:** "Apakah dokumen yang diambil oleh sistem pencarian (Retriever) sebenarnya relevan dengan pertanyaan?"
*   **Logika Juri:** Bandingkan `Retrieved Context` vs `Ground Truth`.

![Metrik RAG](https://lms.sdmdigital.id/pluginfile.php/635508/mod_page/content/9/image36.png)
**Gambar 5. Metrik RAG**

Diagram di atas menunjukkan bahwa evaluasi RAG tidak bisa bersifat satu arah. Untuk mendiagnosis masalah, kita harus melihat hubungan antar komponen:

**1. Kegagalan Garis Bawah (Faithfulness Rendah):**
*   **Artinya:** Jawaban model mengandung klaim yang **tidak ditemukan** dalam dokumen sumber (Context).
*   **Gejala:** *Hallucination*.
*   **Contoh Kasus:** Dokumen berisi "Produk A harganya Rp50.000". Model menjawab: "Produk A harganya Rp100.000 dan sangat populer". Bagian "Rp100.000" dan "sangat populer" adalah halusinasi yang tidak berdasar.

**2. Kegagalan Garis Kiri (Context Precision Rendah):**
*   **Artinya:** Dokumen yang diambil oleh mesin pencari (Retriever) banyak yang **tidak relevan** dengan pertanyaan.
*   **Gejala:** *Bad Retrieval*.
*   **Contoh Kasus:** User bertanya "Cara reset password". Sistem malah mengambil dokumen tentang "Cara cuti pegawai". LLM akhirnya menjawab "Maaf, saya tidak tahu" bukan karena LLM bodoh, tapi karena ia diberi bahan bacaan yang salah. Di sini, yang harus diperbaiki adalah *Retriever*, bukan *Prompt*.

**3. Kegagalan Garis Kanan (Answer Relevance Rendah):**
*   **Artinya:** Jawaban model mungkin benar secara fakta, tapi **tidak menjawab** inti pertanyaan pengguna.
*   **Gejala:** *Off-topic* atau bertele-tele.
*   **Contoh Kasus:** User bertanya "Siapa nama CEO?". Model menjawab dengan biografi panjang sejarah perusahaan tanpa menyebut nama CEO secara eksplisit di awal. Jawaban ini setia pada dokumen, tapi buruk bagi user.

Memahami teori segitiga RAG ini sangat penting, tetapi menghitung korelasi antara ketiga titik tersebut secara manual pada ribuan percakapan adalah hal yang mustahil. Kita membutuhkan alat bantu perangkat lunak yang dapat mengotomatisasi peran "Juri" untuk setiap sisi segitiga tersebut. Mari kita lihat bagaimana *library* RAGAS mengimplementasikan konsep ini dalam kode.

## 8.6.2 Implementasi Kode: Evaluasi RAG dengan RAGAS

Berikut adalah langkah demi langkah menggunakan *library* `ragas`.

**Tahap 1: Instalasi Library**

```bash
!pip install ragas openai datasets
```

Kita menginstal `ragas` sebagai *framework* evaluasi utama. `datasets` (dari Hugging Face) diperlukan karena RAGAS mewajibkan format data input yang spesifik menggunakan objek `Dataset` dari *library* ini, bukan *list* atau *dictionary* biasa Python.

**Tahap 2: Setup Data Evaluasi**

```python
from datasets import Dataset
import os


# Ragas menggunakan OpenAI GPT-4/3.5 sebagai juri default
os.environ["OPENAI_API_KEY"] = "sk-..."


# 1. Siapkan Data (Biasanya hasil log dari aplikasi RAG Anda)
data = {
    # (A) User Query
    'question': [
        'Siapa penemu lampu pijar?',
        'Apa ibukota Indonesia?'
    ],
   
    # (B) Generated Answer (Output Model)
    'answer': [
        'Thomas Edison adalah penemu lampu pijar komersial.',
        'Bali adalah ibukota Indonesia.' # (Contoh Halusinasi)
    ],
   
    # (C) Retrieved Contexts (Dokumen pendukung)
    # Perhatikan: Ini harus berupa list of lists (karena satu query bisa retrieve banyak dokumen)
    'contexts': [
        ['Thomas Alva Edison mengembangkan lampu pijar praktis pertama...'],
        ['Jakarta adalah ibukota negara Indonesia, terletak di pulau Jawa...']
    ],
   
    # (D) Ground Truth (Opsional, tapi bagus untuk Context Recall)
    'ground_truth': [
        'Thomas Edison',
        'Jakarta'
    ]
}


# Konversi ke HuggingFace Dataset
dataset = Dataset.from_dict(data)
```

Sel ini mempersiapkan **Dataset Evaluasi**.
*   Variabel `data` harus memetakan komponen segitiga RAG: `question` (Titik Atas), `answer` (Titik Kanan Bawah), dan `contexts` (Titik Kiri Bawah).
*   **Penting:** `contexts` wajib berupa *list of lists* (contoh: `[['dokumen 1', 'dokumen 2'], ...]`) karena RAG biasanya mengambil lebih dari satu potongan dokumen (*chunks*) untuk satu pertanyaan.

**Tahap 3: Eksekusi Evaluasi**

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
)


# 2. Jalankan Evaluasi
# Ragas akan memanggil GPT-4 di belakang layar untuk menjadi juri
# bagi setiap baris data.
results = evaluate(
    dataset = dataset,
    metrics = [
        faithfulness,      # Mengecek sisi Bawah segitiga (Halusinasi)
        answer_relevancy,  # Mengecek sisi Kanan segitiga (Relevansi)
        context_precision  # Mengecek sisi Kiri segitiga (Kualitas Search)
    ],
)


# 3. Tampilkan Hasil
print(results)
df_results = results.to_pandas()
print(df_results[['question', 'answer', 'faithfulness', 'answer_relevancy']])
```

Fungsi `evaluate()` adalah orkestrator utama. Ia akan mengirimkan serangkaian *prompt* ke OpenAI (Juri) untuk setiap metrik yang diminta.

*   Untuk pertanyaan kedua ("Apa ibukota Indonesia?"), RAGAS akan memberikan skor *Faithfulness* rendah.
*   **Logika Juri:** Model menjawab "Bali", tetapi di dalam `contexts` tertulis "Jakarta". Juri mendeteksi kontradiksi fakta antara *Answer* dan *Context*, sehingga skornya dihukum.

Evaluasi menggunakan RAGAS seperti yang kita lakukan di atas biasanya dijalankan secara **Offline** (misalnya, sebelum kita merilis model baru, kita mengujinya pada 100 soal **Golden Set**). Ini mirip dengan ujian sekolah.

Namun, bagaimana setelah model *live* di aplikasi dan melayani ribuan *user*? Kita tidak bisa menjalankan RAGAS manual setiap saat. Kita membutuhkan sistem "CCTV" yang memantau percakapan secara *real-time*, melacak jalur data, dan memberi peringatan jika kualitas jawaban menurun. Inilah ranah **LLM Observability**.

## 8.6.3 Alat Observabilitas (Observability Tools)

Alat Observabilitas (*LLM Observability Tools*) seperti **LangSmith**, **Arize Phoenix**, atau **Weights & Biases (W&B)** berfungsi untuk merekam, melacak, dan menganalisis perilaku aplikasi LLM di lingkungan produksi.

Berbeda dengan evaluasi statis, *tools* ini berfokus pada **Traces** (Jejak Eksekusi).

**Komponen Utama Observabilitas:**

**1. Tracing (Jejak):**
*   Merekam perjalanan satu *request* dari awal hingga akhir.
*   **Contoh:** User Input -> Retriever (berapa lama? dokumen apa yang diambil?) -> LLM (apa prompt yang dikirim? berapa token?) -> Output Akhir.
*   Ini memungkinkan *engineer* untuk melihat *bottleneck* latensi atau kegagalan di tengah jalan.

**2. Feedback Loop (Human-in-the-Loop):**
*   Mengintegrasikan tombol "Thumbs Up / Thumbs Down" di antarmuka *chat* pengguna.
*   Data ini dikirim ke *dashboard* observabilitas untuk menandai percakapan mana yang buruk dan perlu ditinjau ulang untuk perbaikan *dataset* (*Fine-Tuning* selanjutnya).

**3. Cost & Latency Monitoring:**
*   Melacak berapa biaya API ($) yang dihabiskan per percakapan dan berapa lama waktu responsnya.

Dengan kombinasi **Evaluasi Offline** (menggunakan RAGAS/Golden Set untuk pengujian standar) dan **Observabilitas Online** (menggunakan Tracing untuk memantau dunia nyata), seorang *AI Engineer* memiliki kendali penuh atas kualitas sistemnya. Kita tidak lagi menebak-nebak apakah model kita pintar; kita memiliki data yang membuktikannya.

---

### Mini Quiz
*(Konten interaktif tersedia pada platform LMS)*

---

> **Refleksi Singkat**
>
> Anda telah melihat bagaimana RAGAS bisa mendeteksi halusinasi secara otomatis (contoh "Bali" vs "Jakarta").
> Jika Anda adalah **Lead Engineer**, metrik mana dari **RAG Triad** (Faithfulness, Answer Relevance, Context Precision) yang akan Anda prioritaskan untuk dipantau secara harian?
>
> *   *Jika skor **Faithfulness** rendah, apa yang salah? (Modelnya suka mengarang?)*
> *   *Jika skor **Context Precision** rendah, apa yang salah? (Mesin pencarinya bodoh?) Refleksikan bagaimana skor ini membantu Anda melakukan debugging sistem RAG tanpa harus membaca ribuan log manual.*