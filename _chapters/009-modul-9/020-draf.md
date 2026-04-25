---
slug: modul-9-2
title: modul-9-2
---

## 5.2 Retrieval

---

Proses retrieval persis seperti menggunakan search engine. Anda memberikan query yang ingin dicari, misalnya “cara merakit komputer”, lalu hasilnya adalah informasi yang berkaitan, misalnya serangkaian artikel yang berisi petunjuk merakit komputer. Dalam istilah retrieval, informasi yang dicari dan didapatkan dari retrieval disebut document (dokumen).

**Saat LLM melakukan retrieval, ada beberapa langkah yang dilakukan:**
*   LLM menyiapkan query. Bisa langsung menggunakan pertanyaan user, atau LLM menentukan sendiri querynya berdasarkan konteks situasi.
*   Query lalu diberikan ke sistem pencari untuk menemukan beberapa dokumen yang relevan. Hasil pencarian ini diurutkan berdasarkan skor relevansi.
*   Untuk meningkatkan kualitas pencarian, dokumen-dokumen ini kemudian bisa diurutkan lagi dengan kriteria yang lebih ketat. Istilahnya adalah re-ranking.
*   Setelah itu, dokumen yang sudah didapatkan diberikan ke LLM untuk menghasilkan jawaban.

![Alur proses retrieval](https://lms.sdmdigital.id/pluginfile.php/635468/mod_page/content/72/image6.png)
*Gambar 2. Alur proses retrieval*

Agar prosesnya efisien, retrieval biasanya dibatasi oleh parameter Top-K, yaitu hanya mengambil K dokumen paling relevan. Namun karena pemeringkatan bersifat relatif, threshold kedekatan juga digunakan untuk mencegah false positive (dokumen yang tidak relevan malah dianggap relevan). Threshold ini berfungsi mirip dengan KKM nilai ujian. Meskipun diambil K dokumen dengan skor paling tinggi, kalau semuanya tidak ada yang mencapai KKM, tetap dianggap tidak relevan.

### 5.2.1 Persiapan Query

Umumnya tahap ini tidak dilakukan manual, tapi otomatis ditangani oleh LLM. Ada dua cara, yaitu secara manual atau otomatis lewat deskripsi tool.

Misal kita punya query seperti ini:
> *"Saya ingin membeli handphone baru, apa ya yang bagus?"*

Kalau pertanyaannya seperti ini, sudah jelas apa keinginan user. Berbeda dengan kasus lainnya yang berbasis interaksi. Misalnya ada percakapan berikut antara User dengan chatbot assistant. Input dari user ditandai dengan >>.

> **>> Halo**
> Hai! Apa yang saya bisa bantu?
>
> **>> Saya ingin membeli sebuah handphone baru.. Ada apa saja ya?** ← perlu retrieval (1)
> Dari katalog kami, keluaran terbaru ada dari merk ………..
>
> **>> Oke, kalau merk yang populer dan banyak dibeli yang mana ya?** ← perlu retrieval (2)
> Dari data penjualan, top 3 hape yang terjual adalah hape XX, YY, dan ZZ

Pada retrieval pertama, konteksnya jelas, user ingin membeli handphone, artinya perlu dicarikan informasi handphone yang tersedia di etalase. Tapi pada interaksi selanjutnya, ada konteks percakapan yang perlu dimasukkan ke dalam query. Untuk kasus (2), perlu ada konteks bahwa “merek” yang dimaksud adalah merk untuk hape berdasarkan pembicaraan sebelumnya.

Berikut gambaran kira-kira bentuk query yang dihasilkan pada kedua retrieval:
> Query 1: “katalog merk handphone”
> Query 2: “data penjualan merek handphone”

### 5.2.2 Melakukan Pencarian

Setelah mendapatkan query, langkah selanjutnya adalah mengirimkan query ini untuk melakukan pencarian. Apa yang sebenarnya terjadi dalam sebuah proses pencarian?

Ketika melakukan pencarian, setiap entry informasi dalam database dicocokkan dengan sebuah query berdasarkan suatu kriteria kecocokan. Bagaimana kriteria ini menentukan mana informasi yang “cocok” dengan query pencarian?

Mayoritas informasi yang perlu disimpan dalam sebuah sistem RAG adalah informasi berbentuk teks. Untuk melakukan retrieval teks, ada dua pendekatan yang biasa digunakan, yaitu melakukan matching berdasarkan teks secara langsung, yang disebut **sparse retrieval**, dan melakukan matching melalui kedekatan geometris antara embedding dari setiap informasi dengan embedding yang dihitung dari query (**dense retrieval**).

**Sparse Retrieval: melakukan matching berdasarkan konten teks**
Metode ini melakukan pencarian dengan cara melakukan matching (pencocokan) keyword tertentu antara query dengan informasi yang tersimpan dalam database. Metode ini sangat baik digunakan untuk domain yang memiliki konvensi / kata kunci yang baku dan disepakati. Misalnya domain seperti hukum yang memiliki terminologi yang diatur maupun domain seperti kode yang memiliki syntax tertentu. Metode ini juga efektif apabila memang yang perlu dicari adalah sebuah kata kunci tertentu.

![Ilustrasi sparse retrieval](https://lms.sdmdigital.id/pluginfile.php/635468/mod_page/content/72/image27.png)
*Gambar 3. Ilustrasi sparse retrieval*

Metode yang paling banyak digunakan untuk sparse retrieval adalah BM25. BM25 menghitung skor relevansi dokumen D berdasarkan sebuah query Q dimana suatu query terdiri atas banyak term. Cara BM25 bekerja adalah dengan menghitung 2 skor:

*   **Inverse Document Frequency (IDF).** IDF menghitung skor yang menunjukkan seberapa “langka” suatu term pada seluruh dokumen di database. Logikanya, semakin langka suatu bagian query, informasinya semakin spesifik, sehingga lebih berguna untuk pencarian. Semakin langka suatu term, semakin tinggi skor IDF.
*   **Term Frequency (TF).** TF memberikan skor ke setiap dokumen dengan cara mengukur frekuensi kemunculan term dalam dokumen tersebut, dinormalisasi berdasarkan panjang dokumennya. TF memiliki saturasi, artinya skor TF tidak meningkat secara linear.

**Cara kerja BM25 cukup sederhana:**
1. BM25 membagi query menjadi bagian-bagian query yang disebut term.
2. Term tersebut dinilai skor “kelangkaan”-nya dengan cara menghitung IDF. Semakin langka, skor IDF semakin tinggi.
3. TF dihitung untuk setiap dokumen.
4. Perhitungan TF dan IDF dihitung seterusnya untuk semua term dalam query.
5. TF dan IDF setiap term dikalikan, lalu dijumlahkan menjadi skor akhir.

**Skor BM25 memiliki rumus berikut:**
> $\text{score}(D, Q) = \sum_{i=1}^{n} \text{IDF}(q_i) \cdot \text{TF}(q_i)$ (Eq. 1)

**Dimana IDF dan TF dihitung dengan rumusnya masing-masing:**
> $\text{IDF}(q_i) = \ln \left( \frac{N - \text{df}(q_i) + 0.5}{\text{df}(q_i) + 0.5} \right)$ (Eq. 2)

*   **N** adalah jumlah seluruh dokumen
*   **df(qi)** adalah jumlah dokumen yang mengandung term qi

> $\text{TF}(q_i) = \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$ (Eq. 3)

*   **f(qi, D)** adalah frekuensi term qi dalam sebuah dokumen D
*   **K1** adalah parameter saturasi term, biasanya antara 1.2 - 2.
*   **b** merupakan parameter normalisasi panjang.
*   **|D|** merupakan panjang dokumen.
*   **avgdl** merupakan rata-rata panjang dokumen di database.

![Saturasi skor TF](https://lms.sdmdigital.id/pluginfile.php/635468/mod_page/content/72/image1.png)
*Gambar 4. Saturasi skor TF berdasarkan frekuensi term*

**Kelebihan Sparse Retrieval:**
*   Cepat dan ringan, tidak membutuhkan pemrosesan ekstra.
*   Transparan, faktor yang membuat suatu teks dianggap match dengan query bisa langsung diamati.

**Kelemahan Sparse Retrieval:**
*   Karena berbasis exact matching, metode ini lemah terhadap kasus seperti parafrase maupun kasus multilingual dimana teksnya berbeda walaupun maknanya sama.

#### **Dense Retrieval (vector search): matching berbasis embedding**

Untuk kasus bahasa natural secara umum dimana sebuah makna yang sama bisa direpresentasikan menggunakan istilah yang berbeda-beda, diperlukan metode matching lain yang bisa menangkap kedekatan semantik.

Dense retrieval melakukan matching dengan cara menghitung kedekatan vektor antara query dan setiap informasi yang terdapat dalam database yang semuanya direpresentasikan dalam bentuk embedding.

![Ilustrasi kedekatan embedding](https://lms.sdmdigital.id/pluginfile.php/635468/mod_page/content/72/image26.png)
*Gambar 5. Ilustrasi menghitung kedekatan teks menggunakan embedding.*

**Kelebihan Dense Retrieval:**
*   Mampu menangani parafrase dan kasus multilingual.
*   Lebih baik dalam memproses query yang berupa bahasa natural.
*   Dilatih untuk secara eksplisit memetakan kedekatan makna antara query dan informasi.

**Kelemahan Dense Retrieval:**
*   Tidak selamanya kesamaan semantik = relevansi.
*   Interpretasi kedekatan query lebih sulit karena representasinya abstrak.
*   Sangat bergantung pada kemampuan model embedding.
*   Lebih mahal secara komputasi.

#### **Hybrid retrieval: Kombinasi sparse dan dense retrieval**

Pendekatan ini menggunakan keduanya sekaligus. Ada dua cara menggunakan sparse dan dense retrieval berbarengan:
1. Melakukan sparse retrieval dulu untuk menyaring dokumen, lalu digunakan untuk dense retrieval.
2. Melakukan keduanya secara terpisah, lalu hasil pencariannya digabungkan.

**Cara menggabungkan hasil pencarian:**

1.  **Reciprocal Rank Fusion (RRF)**
    > $\text{RRF}(d) = \sum_{r \in R} \frac{1}{k + \text{rank}_r(d)}$ (Eq. 4)
    *   **K** adalah konstanta (biasanya 60).
    *   **R** adalah seluruh metode ranking yang digunakan.
    *   **rank_r(d)** adalah ranking dokumen mulai dari 1.

2.  **Langsung menggunakan re-ranking.** Semua hasil digabungkan lalu diproses model re-ranking.

3.  **Menggabungkan skor (Weighted Sum)**
    > $\text{score}(d) = \alpha \cdot \text{norm}(s_{\text{BM25}}(d)) + (1 - \alpha) \cdot \text{norm}(s_{\text{vec}}(d))$ (Eq. 5)

**Mana yang sebaiknya digunakan?**

**Tabel 1. Rangkuman singkat mengenai berbagai pendekatan**

| Metode | Kelebihan | Kapan digunakan? | Contoh |
| :--- | :--- | :--- | :--- |
| **Dense** | Cocok untuk kedekatan semantik, menangani parafrase/bahasa. | Default untuk sebagian besar use case. | Artikel, berita, website dengan pertanyaan bebas. |
| **Sparse** | Cocok untuk keyword tertentu. | Ketika perlu matching exact (jarang berdiri sendiri). | Kode surat, nama kategori, prosedur baku. |
| **Hybrid** | Menggabungkan exact dan semantik. | Meningkatkan performa pencarian. | Keluhan pelanggan: butuh SOP (semantik) + nama produk (exact). |

### 5.2.3 Mengenal embedding dan memahami proses inti dense retrieval

Dalam retrieval, digunakan semantic embedding. Model embedding dilatih untuk memetakan input yang berdekatan secara makna agar memiliki kedekatan secara geometris pada ruang vektor.

![Embedding mapping](https://lms.sdmdigital.id/pluginfile.php/635468/mod_page/content/72/image23.png)
*Gambar 6. Embedding memetakan kata yang maknanya berdekatan menjadi vektor yang dekat secara geometris*

Agar embedding memberikan representasi informasi yang baik, umumnya sebuah dokumen dibagi menjadi bagian-bagian kecil (**chunking**) sebelum dihitung embedding-nya.

![Proses chunking](https://lms.sdmdigital.id/pluginfile.php/635468/mod_page/content/72/image8.png)
*Gambar 7. Ilustrasi proses chunking*

### 5.2.4 Bagaimana Kalau Dokumennya Tidak Hanya Berisi Teks?

Dokumen bisa berisi gambar, bagan, atau tabel. Salah satu caranya adalah merepresentasikan informasi tersebut sebagai teks (misal: deskripsi gambar lewat LLM lain) atau diproses dalam tahap ingestion.

### 5.2.5 Database Informasi

Penyimpanan yang umum digunakan adalah **vector database**. Selain menyimpan vektor embedding, database ini menyimpan:
*   **Bentuk teks** dari embedding tersebut.
*   **Metadata** (tanggal, kategori, topik) untuk filtering.

![Cara menghitung kedekatan vektor](https://lms.sdmdigital.id/pluginfile.php/635468/mod_page/content/72/image4.jpg)
*Gambar 8. Ilustrasi berbagai cara menghitung kedekatan vektor.*

#### **Retrieval yang efisien menggunakan metode approximate nearest neighbors (ANN)**

Untuk database skala besar, mencari tetangga terdekat secara akurat ke seluruh vektor akan sangat lambat. Maka digunakan strategi **indexing** melalui ANN.

![K-Nearest Neighbors](https://lms.sdmdigital.id/pluginfile.php/635468/mod_page/content/72/image25.png)
*Gambar 9. Ilustrasi K-Nearest Neighbors*

**Tabel 2. Bermacam index yang digunakan untuk melakukan ANN**

| Index | Cara Kerja | Digunakan di | Kapan Digunakan | Kapan Dihindari |
| :--- | :--- | :--- | :--- | :--- |
| **Flat** | Brute Force (Exact search) | FAISS, Qdrant | Dataset kecil (<50k) | Dataset sangat besar |
| **HNSW** | Graph multi-layer | Qdrant, Milvus | Latency rendah, akurasi tinggi | Sering update data |
| **IVF** | Clustering (k-means) | FAISS, Pinecone | Dataset jutaan+, latency minimal | Sering update data |
| **IVF + PQ** | IVF + Kompresi vektor | FAISS, Milvus | Memory terbatas | Butuh akurasi sangat tinggi |
| **DiskANN** | Graph berbasis disk | Azure Search | Dataset >1B vektor | RAM melimpah |

### 5.2.6 Library/Framework/Tooling Vector Database

**Tabel 3. Bermacam vector database**

| Nama | Fitur | Indeks | Jenis | Akses |
| :--- | :--- | :--- | :--- | :--- |
| **Pinecone** | Hybrid, serverless | HNSW | Full DB | Fully-managed |
| **Weaviate** | Hybrid, integrasi model | HNSW | Full DB | OS + Managed |
| **FAISS** | GPU acceleration | IVF, HNSW, PQ | Library | OS |
| **Qdrant** | Advanced search | HNSW | Full DB | OS + Managed |
| **Milvus** | High performance, skala miliaran | Banyak | Full DB | OS + Managed |

**Tabel 4. Fitur / ekstensi dari solusi penyimpanan umum**

| Ekstensi | Asal | Indeks | Fitur |
| :--- | :--- | :--- | :--- |
| **pgvector** | PostgreSQL | HNSW, IVFFlat | Vector search langsung di SQL |
| **Elasticsearch** | Elasticsearch | HNSW | Hybrid semantic + keyword |
| **RediSearch** | Redis Stack | HNSW, FLAT | Hybrid text + vector di Redis |

**Pakai yang mana?**
*   **Awal Pengembangan:** Weaviate, Qdrant, atau ChromaDB (Full DB).
*   **Menghindari Dependency Baru:** PGVector (jika sudah pakai Postgres).
*   **Skala Besar/Produksi:** Milvus atau Pinecone.

### 5.2.7 Hands on: Contoh Implementasi Retrieval Sederhana

**Langkah 1: Impor Library**
```python
from sentence_transformers import SentenceTransformer
import numpy as np
from openai import OpenAI
import json
```

**Langkah 2: Inisiasi Variabel**
```python
TOP_K = 3
SIMILARITY_THRESHOLD = 0.3
DOCS_PATH = "samples/articles/"
DOCUMENTS = list()
for doc in glob(os.path.join(DOCS_PATH, "*.txt")):
    with open(doc, encoding="utf8") as f:
        DOCUMENTS.append(f.read())
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
```

**Contoh isi dokumen (Rafael Amani):**
> *Rafael Amani was born on October 14, 1975, in the coastal city of Port Valora... At age 13, he was recruited into the academy of Valora Athlétique...*

**Langkah 3: Hitung Embedding & Backup**
```python
BACKUP_PATH = "dataset.npy"
if os.path.exists(BACKUP_PATH):
    DOC_EMBEDDINGS = np.load(BACKUP_PATH)
else:
    DOC_EMBEDDINGS = embedder.encode(DOCUMENTS, normalize_embeddings=True, batch_size=len(DOCUMENTS))
    np.save(BACKUP_PATH, DOC_EMBEDDINGS)
```

**Langkah 4: Fungsi Retrieval**
```python
def retrieve_docs(query: str, top_k: int = 3):
    q_emb = embedder.encode([query], normalize_embeddings=True)[0]
    sims = np.dot(DOC_EMBEDDINGS, q_emb)
    idx = sims.argsort()[::-1][:top_k]
    results = [{"document": DOCUMENTS[i], "score": float(sims[i])} for i in idx]
    results = [result for result in results if result["score"] >= SIMILARITY_THRESHOLD]
    return results
```

**Langkah 5: Mendefinisikan Tool & System Prompt**
```python
tools = [{
    "type": "function",
    "function": {
        "name": "retrieve_docs",
        "description": "Retrieve top-k documents relevant to a query.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    }
}]

SYSTEM_PROMPT = """\
# ROLE
Take the role of a football knowledge assistant...
# RULES
- Answer using Indonesian language.
- Answer about football players must be based only on the information given.
"""
```

**Langkah 6: Fungsi Chat**
```python
def chat(query):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]
    response = client.chat.completions.create(model=MODEL_NAME, messages=messages, tools=tools, tool_choice="auto")
    msg = response.choices[0].message

    if msg.tool_calls:
        for call in msg.tool_calls:
            if call.function.name == "retrieve_docs":
                args = json.loads(call.function.arguments)
                results = retrieve_docs(args["query"], TOP_K)
                messages.append(msg)
                messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(results)})
                final = client.chat.completions.create(model=MODEL_NAME, messages=messages)
                return final.choices[0].message.content
    return msg.content
```

**Uji Skenario:**
*   **Case 1:** "Di klub apa Rafael Amani pertama kali bermain?"
    *   *Hasil:* Rafael Amani pertama kali direkrut masuk ke akademi Valora Athlétique pada usia 13 tahun.
*   **Case 2:** "Berapa jumlah gol Zinedine Zidane?"
    *   *Hasil:* Maaf, saya tidak memiliki informasi spesifik mengenai jumlah gol Zinedine Zidane.
*   **Case 3:** "Siapa yang membuat nasi uduk?"
    *   *Hasil:* Nasi uduk adalah hidangan tradisional... Maaf, topik ini bukan keahlian saya.

### Mini Quiz
*(H5P Content Placeholder)*

> **Refleksi Singkat**
>
> Retrieval memungkinkan sebuah aplikasi secara mencari informasi yang tersimpan di luar aplikasi tersebut secara dinamis. Kemampuan seperti ini sangat berguna kalau kita ingin LLM dapat menjawab pertanyaan yang jawabannya perlu menggunakan referensi tertentu.
>
> Apakah ada use case selain tanya-jawab yang bisa menggunakan retrieval? Seperti apa use case tersebut?