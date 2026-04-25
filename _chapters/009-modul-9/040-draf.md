---
slug: modul-9-4
title: modul-9-4
---
## 5.4 Evaluasi RAG

---

Sama seperti aplikasi pada umumnya, RAG juga perlu dievaluasi agar performa baik dari segi kinerja menyelesaikan masalah (akurasi) maupun dari segi operasionalnya (latency) dapat dimonitor untuk memastikan kualitas layanan dari aplikasi yang dibuat konsisten, ada metrik terukur yang bisa dijadikan basis untuk optimasi sistem, dan memperingatkan developer ketika diukur performanya semakin memburuk karena dinamika lainnya dari sistem yang berjalan di production seperti meningkatnya jumlah user atau jumlah request.

RAG terdiri dari beberapa komponen yang masing-masing memiliki peran dalam menghasilkan jawaban akhir, sehingga setiap komponen ini perlu dievaluasi juga secara terpisah, tidak hanya evaluasi jawaban akhir saja. Dalam RAG, ada dua proses kunci yang menjadi perhatian khusus untuk melakukan evaluasi:

1.  Proses pencarian chunk dokumen yang relevan
2.  Menggunakan chunk tersebut untuk menghasilkan jawaban

Pertama mari kita bahas metrik evaluasi yang digunakan untuk melakukan evaluasi retrieval.

### 5.4.1 Evaluasi Retrieval

Metrik retrieval berikut berfokus pada pertanyaan: “Apakah dokumen yang diberikan ke LLM benar-benar berisi fakta yang diperlukan untuk menjawab pertanyaan?”

Metode ini tidak menilai dokumen mana yang relevan, seperti Recall@K tradisional, tetapi menilai apakah fakta-fakta dalam ground truth ditemukan di dalam konteks hasil retrieval. Fokusnya adalah konteks, bukan dokumen. Evaluasi seperti ini umumnya memerlukan LLM.

#### 1. Context Recall@K

**Definisi:** Context Recall@K mengukur seberapa banyak poin/gagasan penting dari jawaban ground truth yang berhasil muncul di dalam K dokumen hasil retrieval. Yang dievaluasi bukan dokumennya, tetapi point informasi yang ada dalam hasil retrieval. Jika informasi tersebut ada di dokumen, berarti retriever berhasil menyediakan bukti yang dibutuhkan LLM.

**Cara menghitung:**
1.  Ambil jawaban ground-truth.
2.  Ekstrak fakta-fakta penting dari ground truth menggunakan LLM.
3.  Ambil K dokumen hasil retrieval.
4.  Gunakan LLM untuk memeriksa fakta mana saja yang didukung / ditemukan di dalam dokumen-dokumen tersebut.
5.  Context Recall@K = jumlah fakta yang ditemukan / jumlah total fakta ground truth.
6.  Hitung untuk semua test case, lalu ambil rata-rata untuk seluruh dataset.

**Kegunaan:**
*   Mengukur kemampuan retrieval menyediakan bukti faktual yang dibutuhkan untuk menjawab pertanyaan.
*   Sensitif terhadap kualitas chunking dan indexing: jika fakta terpotong atau tidak terambil, recall turun.
*   Biasanya digunakan untuk tuning nilai K, mencari nilai K terkecil yang bisa memberikan fakta yang dibutuhkan secara konsisten.

#### 2. Context Precision@K

**Definisi:** Context Precision@K mengukur proporsi informasi dalam K dokumen retrieval yang benar-benar relevan dengan fakta ground truth. Metrik ini menilai seberapa “bersih” konteks dari fakta-fakta yang tidak terkait.

**Cara menghitung:**
1.  Ekstrak fakta ground truth.
2.  Gunakan LLM untuk ekstraksi semua fakta yang muncul di K dokumen retrieval.
3.  Hitung berapa banyak dari fakta tersebut yang relevan dengan jawaban ground truth.
4.  Context Precision@K = fakta relevan / total fakta di dokumen.
5.  Hitung untuk semua test case, lalu ambil rata-rata untuk seluruh dataset.

**Kegunaan:**
*   Menilai apakah konteks yang diberikan terlalu banyak noise (informasi yang tidak relevan) yang berpotensi membuat LLM bingung / berhalusinasi.
*   Digunakan untuk tuning K, sebagai gambaran seberapa banyak noise untuk nilai K tertentu.
*   Mengukur efektivitas reranker untuk memberi ranking lebih tinggi dokumen yang benar-benar mengandung fakta penting.

#### 3. Mean Reciprocal Rank (MRR)

**Definisi:** MRR mengukur dalam K dokumen hasil retrieval, di mana letak dokumen yang benar-benar relevan pertama kali muncul, dihitung satu per satu mulai dari hasil retrieval dengan urutan tertinggi.

**Cara menghitung:**
1.  Urutkan hasil retrieval berdasarkan skor relevansi.
2.  Tentukan relevansi antara setiap dokumen dengan jawaban menggunakan LLM.
3.  Cari rank, yaitu indeks terkecil dari dokumen yang relevan. Indeksnya mulai dari 1 untuk urutan pertama.
4.  Hitung reciprocal rank atau dengan rumus 1/rank.
5.  Ambil rata-rata untuk semua test case.

**Kegunaan:**
*   Mengukur seberapa mampu model embedding yang digunakan untuk memetakan dokumen yang relevan dengan query secara berdekatan.
*   Mengukur kemampuan re-ranking untuk menempatkan dokumen yang sebenarnya relevan pada urutan yang lebih tinggi.
*   Bisa juga memberikan gambaran untuk tuning nilai K.

#### Pakai yang mana?

Idealnya semuanya dipakai, karena ketiganya sama-sama penting. Tapi untuk sistem RAG yang masih minimal, bisa bermula dari Recall@K dulu. Ketika mulai terdeteksi kalau jawaban RAG menurun kualitasnya padahal berhasil mendapatkan dokumen yang relevan, bisa ditambahkan Precision@K. MRR bisa ditambahkan untuk mengevaluasi lebih lanjut ketika ingin menggunakan re-ranking atau ingin mencari tahu seberapa cocok model embedding yang sekarang digunakan.

Setelah retrieval menghasilkan Top-K dokumen yang dianggap paling relevan dengan query, semua dokumen tersebut akan digunakan untuk menghasilkan jawaban. Efektivitas dari jawaban yang dihasilkan juga perlu diukur untuk memastikan:
*   Jawabannya benar-benar menggunakan informasi yang diberikan retrieval.
*   Apakah jawabannya faktual dan menjawab pertanyaanya dengan baik.

### 5.4.2 Evaluasi Jawaban Akhir

Untuk mengukur kualitas jawaban dari RAG, ada tiga metrik utama:

#### 1. Faithfulness

**Definisi:** Faithfulness mengukur apakah jawaban yang diberikan oleh LLM didasarkan pada hasil retrieval. Metrik ini berfokus kepada apakah jawaban dari LLM berisikan fakta yang didukung oleh hasil retrieval.

**Cara menghitung:**
Metric ini perlu dihitung menggunakan LLM-as-a-judge. Deepeval membagi implementasi evaluasi faithfulness menjadi beberapa tahapan:
1.  Dari jawaban, LLM diminta mengekstraksi “klaim” atau “pernyataan” independen.
2.  Dari konteks, LLM diminta mengekstraksi “fakta” independen.
3.  LLM diminta mengidentifikasi setiap “klaim” pada jawaban apakah ada yang berkontradiksi dengan “fakta” dalam konteks.
4.  Faithfulness dihitung sebagai persentasi klaim yang didukung oleh fakta dalam konteks yang berasal retrieval.

**Kegunaan:**
*   Memastikan jawaban LLM berasal dari dokumennya, bukan halusinasi maupun dari pengetahuan internalnya.

#### 2. Relevance

**Definisi:** Relevance mengukur apakah respon LLM menjawab pertanyaan / input dari user.

**Cara menghitung:**
Menggunakan LLM-as-a-judge untuk menentukan apakah respon LLM relevan untuk menjawab pertanyaan user. Contoh prompt:
> Diberikan suatu pertanyaan dan jawaban dari AI, tentukan apakah jawaban tersebut relevan untuk menjawab pertanyaannya.
>
> Pertanyaan: {{pertanyaan}}
> Jawaban AI: {{jawaban_ai}}
> Relevan (ya/tidak):

**Kegunaan:**
*   Mengukur kemampuan LLM mengerti pertanyaan user.
*   Secara tidak langsung mengukur kemampuan LLM membuat keputusan dalam menjawabnya (contoh: apakah memang perlu retrieval?).

#### 3. Correctness

**Definisi:** Memastikan validitas jawaban.

**Cara menghitung:**
Menggunakan LLM-as-a-judge untuk membandingkan jawaban akhir LLM dengan jawaban ground truth. Tidak terlalu banyak perbedaan dengan evaluasi jawaban akhir LLM pada umumnya.

**Kegunaan:**
*   Memastikan jawaban LLM benar dan sesuai ekspektasi.

#### Pakai yang mana?

Idealnya, semua metrik di atas baik perlu digunakan secara bersama:
*   **Faithfulness** memastikan jawabannya berdasarkan hasil retrieval. (inti dari RAG)
*   **Relevance** memastikan baik jawaban akhir serta retrieval yang dilakukan relevan / diperlukan.
*   **Correctness** memeriksa validitas jawaban dibandingkan dengan ground truth.

### 5.4.3 Metrik Operasional

Selain metrik fungsional, juga diukur metrik operasional, utamanya **latency**:
*   **Latency end-to-end**: mengukur waktu keseluruhan dari input sampai jawaban.
*   **Latency proses retrieval**: mengukur waktu untuk retrieval.
*   **Latency untuk proses individual**: menghitung embedding, vector search, re-ranking, dsb.

Metrik ini biasanya dihitung untuk mencari bottleneck performa sebuah sistem menggunakan library tracing dan monitoring.

### 5.4.4 Implementasi Evaluasi RAG

Implementasinya membutuhkan LLM lain untuk menilai jawaban (LLM-as-a-judge). Berikut pilihan framework evaluasi:

**Tabel 11. Pilihan Library untuk melakukan evaluasi RAG**

| Nama Library | Keterangan |
| :--- | :--- |
| **RAGAS** | Berisi fungsi-fungsi untuk menghitung metrik penting untuk aplikasi AI, termasuk RAG. |
| **DeepEval** | Berisi fungsi-fungsi untuk menghitung metrik penting untuk aplikasi AI, termasuk RAG. |
| **LangChain Eval** | Bagian dari LangChain dan langsung terintegrasi dengan ekosistem LangChain. |
| **Custom** | Digunakan ketika kebutuhan metrik cukup custom sehingga perlu dibuat sendiri. |

**Contoh menggunakan DeepEval untuk menghitung metrik:**

```python
import os
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    GEval
)
from deepeval.models import GPTModel, GeminiModel
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval import evaluate

# Inisialisasi model evaluasi
model = GeminiModel(
    "gemini-2.5-flash",
    api_key=os.getenv("API_KEY_LLM")
)

test_case = LLMTestCase(
    input="Apa penyebab utama kemacetan di Jakarta?",
    actual_output="Kemacetan di Jakarta disebabkan oleh semakin banyaknya kendaraan pribadi.",
    expected_output="Faktor utama kemacetan di Jakarta adalah pertumbuhan kendaraan pribadi yang tinggi dan rendahnya penggunaan transportasi umum.",
    retrieval_context=[
        "Studi pada tahun 2020 menunjukkan pertumbuhan kendaraan pribadi meningkat 8% per tahun di Jakarta.", # relevan
        "Transportasi umum masih belum menjadi pilihan mayoritas warga karena keterbatasan jangkauan.", # relevan
        "Sejarah Jakarta menunjukkan bahwa pada era kolonial, jalur kereta dan trem pernah digunakan.", # tidak relevan
        "Jakarta memiliki lebih dari 13 juta perjalanan harian dalam kota.", # sedikit relevan
        "Kemacetan juga dipengaruhi oleh pembangunan infrastruktur yang memakan satu lajur jalan." # agak relevan
    ]
)

# Definisi metrik
faithfulness = FaithfulnessMetric(model=model)
relevance = AnswerRelevancyMetric(model=model)
correctness = GEval(
    name="Correctness",
    model=model,
    criteria="Tentukan apakah jawaban tersebut secara faktual benar dibandingkan dengan jawaban referensi.",
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT]
)
precision = ContextualPrecisionMetric(model=model)
recall = ContextualRecallMetric(model=model)

# Jalankan evaluasi
evaluate(
    test_cases=[test_case],
    metrics=[faithfulness, relevance, correctness, precision, recall],
)
```

Bisa diperhatikan bahwa setiap test case ini tidak hanya terdiri dari input, output, dan ground truth, tapi juga ada nilai perantara yang didapatkan dari retrieval. Untuk mendapatkan nilai-nilai ini perlu yang namanya instrumentasi untuk “mencatat” hasil retrieval, query, dan hasil re-ranking.

---

### Mini Quiz
*(H5P Content Placeholder - Multiple Choice Quiz)*

> #### Refleksi Singkat
> Evaluasi RAG kini sangat mengandalkan LLM-as-a-judge untuk menilai recall, precision, faithfulness, relevance, dan correctness.
>
> Menurut Anda, apa tantangan terbesar ketika evaluasi otomatis seperti ini digunakan dalam sistem produksi? Apakah Anda melihat kebutuhan untuk tetap melakukan evaluasi manual oleh manusia, atau LLM-as-a-judge sudah cukup di sebagian besar kasus?