---
slug: modul-9-3
title: modul-9-3
---
# Modul 5.3 Ingestion

---

Dokumen yang ingin digunakan dalam RAG bisa memiliki format yang berbeda-beda. Selain itu, dokumen-dokumen ini bisa memiliki struktur dan konten yang berbeda, bahkan konten selain dalam bentuk teks seperti tabel, gambar, dan visualisasi data.

Dokumen baru juga seiring waktu akan ditambahkan ke sistem. Diperlukan sistem terpisah untuk memproses dan mempersiapkan dokumen-dokumen ini, mulai dari bentuk aslinya hingga masuk ke database dalam format dan bentuk data yang bisa dicari. Untuk itu, kita perlu sistem ingestion.

**Ingestion merupakan proses untuk mempersiapkan suatu dokumen agar siap digunakan dalam RAG yang terdiri dari beberapa tahap:**

1.  **Melakukan pemrosesan dan normalisasi dokumen.** Berbagai bentuk dokumen seperti dokumen teks, presentasi, spreadsheet, PDF memiliki format yang berbeda yang perlu diproses dan dinormalisasi agar memiliki representasi teks yang konsisten. Selain itu aturan pemrosesan dan pembersihan data yang sama seperti menyiapkan data untuk training LLM juga perlu diterapkan, termasuk di antaranya membersihkan data dan melakukan deduplikasi.
2.  **Melakukan chunking,** yaitu memecah sebuah dokumen besar menjadi bagian-bagian kecil yang diharapkan mengandung kepingan-kepingan informasi yang lebih terfokus agar pencarian lebih efektif. Metode untuk melakukannya bermacam, mulai dari metode sederhana seperti langsung memotong-motong dokumen berdasarkan jumlah karakter atau kata, menggunakan struktur internal dari sebuah dokumen untuk melakukan pemisahan secara semantik, hingga menggunakan sebuah model untuk memisahkan dokumen menjadi bagian-bagian yang memiliki informasi semantik yang independen satu sama lain.
3.  **Menghitung embedding.** Tahap ini dilakukan untuk sistem yang menggunakan dense retrieval yang merupakan mayoritas dari use case RAG.
4.  **Menambahkan metadata.** Setiap kepingan informasi bisa disertai dengan informasi tambahan yang memudahkan identifikasi maupun membantu mengerucutkan ruang pencarian berdasarkan atribut yang terdapat di dalam metadata setiap dokumen.
5.  **Menyimpan setiap chunk dokumen, embedding, serta metadatanya ke dalam database.** Tahap ini tidak sesimpel hanya mengirimkan nilai-nilai tersebut ke database untuk disimpan. Untuk database yang menggunakan indexing, perlu dipikirkan strategi yang tepat untuk mengupdate indeks tersebut.

![Ilustrasi proses ingestion](https://lms.sdmdigital.id/pluginfile.php/635469/mod_page/content/47/image34.png)
*Gambar 10. Ilustrasi proses ingestion*

Selanjutnya akan dibahas setiap tahap ini.

### 5.3.1 Pemrosesan dan Normalisasi Dokumen

Dokumen yang menjadi sumber informasi untuk sebuah sistem RAG bisa memiliki format yang beragam dan berisi informasi dalam berbagai macam modalitas. Semua format ini perlu dinormalisasi menjadi bentuk yang seragam agar:

1.  Memiliki representasi yang konsisten agar lebih mudah di-chunking dan pencarian lebih efisien, dan
2.  Data yang tidak berbentuk teks juga bisa dimasukkan ke dalam database untuk dicari.

**Sebuah pipeline pemrosesan dan normalisasi dokumen umumnya memiliki alur pemrosesan sebagai berikut:**

1.  Melakukan ekstraksi konten teks dari sebuah dokumen.
2.  Menangani pemrosesan untuk konten selain teks.
3.  Memproses hasil ekstraksi teks dan konten lain menjadi sebuah representasi yang seragam.

#### Melakukan ekstraksi konten teks dari sebuah dokumen

Setiap jenis dokumen memiliki format spesifiknya masing-masing yang membutuhkan logika pemrosesan yang terspesialisasi untuk melakukan ekstraksi informasinya. Proses ekstraksi informasi dari suatu format data disebut sebagai parsing. Pada praktiknya, berbagai macam library menyediakan fungsionalitas parsing untuk format dokumen tertentu.

**Tabel 5. Bermacam library yang bisa membantu parsing format file tertentu**

| Format | Library untuk parsing | Catatan |
| :--- | :--- | :--- |
| PDF | `pdfplumber`, `PyMuPDF`, `pdfminer.six` | PDF merupakan format kompleks, perlu perhatian spesifik setelah parsing. |
| PDF hasil scan | `tesseract-ocr`, `AWS Textract`, `Azure Form Recognizer` | Digunakan jika halaman PDF berbentuk gambar. |
| DOCX | `python-docx`, `docx2txt` | - |
| HTML | `BeautifulSoup`, `trafilatura` | Informasi tidak relevan (iklan, navigasi) perlu dibersihkan. |
| Markdown | `markdown` | Format paling dasar untuk normalisasi dokumen. |
| PowerPoint (PPTX) | `python-pptx` | - |
| Excel | `openpyxl`, `pandas` | Umumnya hanya memberi akses nilai baris dan kolom. |
| JSON / YAML / XML | `PyYAML` | Struktur data sering sangat nested dan perlu normalisasi. |
| Emails (.eml) | `email` module | - |
| Gambar, tabel | Model khusus atau LLM multimodal | Untuk mengekstraksi teks di dalam gambar. |

Selain melakukan ekstraksi teks, berbagai library di atas juga bisa mengekstraksi data selain teks, misalnya gambar, tabel atau grafik. Nantinya data selain teks ini perlu diubah menjadi representasi teks, lalu dimasukkan kembali ke teks hasil pemrosesan.

#### Mengubah hasil ekstraksi teks menjadi format baku

Tahap ini dilakukan bersamaan bersamaan dengan melakukan ekstraksi teks. Agar dokumen yang tersimpan seragam, perlu ditentukan suatu format penyimpanan dengan beberapa atribut berikut:

1.  Struktur yang jelas dan konsisten.
2.  Ada pembagian hierarki semantik yang jelas.
3.  Teks yang ternormalisasi (tidak ada karakter spesial yang telalu niche, tidak ada whitespace berlebihan, dsb).

Format seperti markdown dan HTML/XML merupakan beberapa format yang umum digunakan sebagai representasi kanonikal / “baku”. Contohnya, misal kita merepresentasikan sebuah resep makanan menggunakan markdown:

```markdown
# Resep Nasi Goreng Sederhana

## Deskripsi
Nasi goreng sederhana yang mudah dibuat dengan bahan sehari-hari. Cocok untuk sarapan atau makan malam cepat.

## Bahan
* 2 piring nasi putih dingin
* 2 siung bawang putih cincang
* 3 siung bawang merah iris
* 1 butir telur
* 1 sdm kecap manis
* 1 sdt kecap asin
* 1 sdt garam
* 1 sdt kaldu bubuk
* Minyak untuk menumis
* Topping optional:
  * Bawang goreng
  * Kerupuk
  * Irisan mentimun

## Peralatan
* Wajan
* Spatula
* Kompor
```

Selain format, data juga mungkin menggunakan character set yang terlalu spesifik. Misalnya karakter **가** dalam aksara hangul punya dua representasi unicode:
*   Bentuk precomposed: **가** (U+AC00)
*   Dua karakter berurutan: **ㄱ** (U+1100) + **ㅏ** (U+1161)

Representasi seperti ini perlu dinormalisasi agar baku. Kenapa perlu normalisasi? LLM menggunakan tokenization (byte pair encoding). Karakter yang menggunakan unicode berbeda bisa memiliki token berbeda, sehingga penghitungan embedding dan pencarian menjadi tidak konsisten.

#### Menangani pemrosesan untuk konten selain teks

Saat melakukan parsing dokumen, konten seperti tabel, gambar, atau visualisasi data membutuhkan perlakuan khusus karena "makna" informasinya bergantung pada konteks di sekitarnya. Contohnya form berikut:

| Keterangan | Data |
| :--- | :--- |
| Nama | Agus |
| Jenis Kelamin | Laki-laki |
| Umur | 18 Tahun |
| Tempat, Tanggal Lahir | Mataram, 21 Maret 2007 |
| Golongan Darah | O |
| Pekerjaan | Mahasiswa |

Jika hanya diubah menjadi teks: *"Agus adalah seorang Mahasiswa berumur 18 tahun dengan jenis kelamin laki-laki dan golongan darah O. Dia lahir di Mataram pada tanggal 21 maret 2007."* Kita kehilangan konteks: Apakah Agus seorang pasien? Pelaku kriminal? Setelah melihat dokumen penuhnya:

> **Laporan orang hilang**
> Pada tanggal 21 Maret 2025, Seorang pendaki dilaporkan hilang... (disertai data diri di atas).

Paling sederhana adalah mengubah konten non-teks menjadi representasi teks lalu dimasukkan kembali ke dokumennya.

![Integrasi konten non-teks](https://lms.sdmdigital.id/pluginfile.php/635469/mod_page/content/47/image35%20%281%29.jpg)
*Gambar 11. Cara integrasi konten yang bentuknya bukan teks ke dokumen sebelum di-chunk dan dihitung embeddingnya*

**Tabel**
Tabel dapat diubah ke bentuk CSV atau Markdown. Contoh tabel:

**Tabel 6. Contoh tabel**

| Quarter | Penjualan (Juta Rupiah) | Operasional (Juta Rupiah) | Keuntungan Bersih (Juta Rupiah) |
| :--- | :--- | :--- | :--- |
| Q1 | 50.000 | 37.500 | 12.500 |
| Q2 | 65.000 | 40.400 | 24.500 |
| Q3 | 75.000 | 50.000 | 25.000 |
| Q4 | 70.000 | 60.000 | 10.000 |

Format CSV:
```csv
Quarter,Penjualan (Juta Rupiah),Operasional (Juta Rupiah),Keuntungan bersih (Juta Rupiah)
Q1,50.000,37.500,12.500
Q2,65.000,40.400,24.500
Q3,75.000,50.000,25.000
Q4,70.000,60.000,10.000
```

Atau representasi struktur lain:
```text
Data penjualan:
  - Q1:
      Penjualan (juta rupiah): 50.000
      Operasional (juta rupiah): 37.500
      Keuntungan Bersih (juta rupiah): 12.500
```

**Form**
Form bisa disimpan sebagai data terstruktur (JSON). Contoh:
```json
{
  "dokumen": "LAPORAN ORANG HILANG GUNUNG YY, ZZ",
  "tanggal_dokumen": "21-03-2025",
  "data": {
    "nama": "agus",
    "jenis_kelamin": "laki-laki",
    "umur": 18,
    "golongan_darah": "O"
  }
}
```

**Kode, Rumus, dan Informasi Bersyntax**
Kode dan rumus harus dijaga syntax-nya. Bisa diberikan caption oleh LLM dan dipisahkan dalam blok khusus:
> *Kode: Implementasi proses pembersihan data untuk dokumen hasil crawling dari internet.*

**Grafik / Visualisasi Data**
Gunakan model image captioning (VLM) untuk mendeskripsikan insight visual.

![Contoh grafik](https://lms.sdmdigital.id/pluginfile.php/635469/mod_page/content/47/image33.png)
*Gambar 12. Contoh data berbentuk grafik / visualisasi data*

Contoh caption: *"Figure: Sebuah bar chart menunjukkan tingkat partisipasi peserta Tech Conference 2024. Mayoritas berada di rentang 35-44..."*

**Gambar**
Gambar yang berisi teks diproses dengan OCR. Gambar visual diberi caption. Gambar dekoratif bisa di-filter. Untuk gambar dengan info semantik tinggi (foto produk/medis), hitung embeddingnya menggunakan model multimodal seperti CLIP atau SigLIP agar bisa dicari melalui query teks.

**Tabel 7. Pendekatan untuk memproses jenis konten selain teks di dalam dokumen**

| Tipe | Cara Memproses | Representasi Teks | Tools |
| :--- | :--- | :--- | :--- |
| Tabel | Ekstraksi/Parsing/OCR | CSV atau Bullet points | Camelot, Unstructured |
| Gambar | OCR atau Captioning | Teks OCR / Caption | Tesseract, VLM |
| Visualisasi | Captioning | Caption | VLM |
| Form | Parsing format dokumen | Key-value/JSON | Library parsing |

### 5.3.2 Chunking

Setelah menjadi teks, dokumen dipecah menjadi **chunk** agar embedding lebih efektif.
**Analogi:** Foto sebuah kota. Terlalu jauh hanya terlihat gambaran besar, terlalu dekat hanya terlihat satu rumah. Ukuran chunk harus tepat ("sweet spot").

**Keuntungan Chunking:**
*   Akurasi retrieval lebih tinggi.
*   Konteks lebih terfokus.
*   Mengurangi risiko halusinasi LLM.

**Metode Chunking:**
1.  **Fixed-length:** Membagi chunk sama panjang (token/karakter) dengan overlap untuk menjaga konteks di perbatasan.
2.  **Structure-aware:** Menggunakan heading/struktur dokumen sebagai batas.
3.  **Semantic chunking:** Mengelompokkan kalimat berdasarkan kedekatan makna menggunakan embedding.
4.  **Hierarchical / Multi-Level Chunking:** Membagi menjadi parent chunk (besar) dan child chunk (kecil).
    *   *Ekspansi konteks:* Saat retrieval child, ambil summary parent atau chunk "saudara" di bawah parent yang sama.
5.  **Chunking menggunakan LLM:** LLM diminta membagi dokumen menjadi N bagian berdasarkan topik.

**Tabel 8. Strategi memilih metode chunking**

| Metode | Kapan Digunakan | Cara Menggabungkan |
| :--- | :--- | :--- |
| Fixed-length | Awal/Default | Pemotongan lanjutan setelah metode lain. |
| Structure-aware | Dokumen terstruktur (Legal/Laporan) | Menjadi parent chunk. |
| Semantic | Fokus pada topik tunggal | Pemecahan lanjutan setelah structure-aware. |
| Hierarchical | Dokumen panjang butuh presisi | Parent (structure) -> Child (fixed). |
| LLM-based | Topik sangat tersebar | Membentuk struktur baru di awal. |

**Tabel 9. Bermacam framework dan library untuk chunking**

| Nama | Jenis | Penjelasan |
| :--- | :--- | :--- |
| LangChain | Framework LLM | Fungsi chunking terpisah. |
| LlamaIndex | Framework LLM | Terintegrasi dalam ingestion pipeline. |
| Unstructured | Library ingestion | Bagian dari pipeline pemrosesan. |
| Chonkie | Library chunking | Berbagai metode chunking & integrasi. |

### 5.3.3 Pentingnya Metadata Untuk Database RAG

Metadata adalah informasi ekstra (JSON/tabel) yang disimpan bersama chunk dan embedding.

**Kegunaan Metadata:**
1.  **Efektivitas Retrieval:** Membatasi pencarian berdasarkan topik/tanggal (filtering).
2.  **Konteks Tambahan:** Menyimpan summary parent atau lokasi hierarki.
3.  **Operasional & Audit:** Menyimpan sumber dokumen (sitasi), versi model embedding, dan informasi debugging.

### 5.3.4 Tahap Terakhir: Menghitung Embedding dan Memasukkan Data Ke Database

Setelah chunk siap, gunakan model embedding untuk mengubahnya menjadi vektor. Gunakan benchmark seperti **MTEB Leaderboard** untuk memilih model.

![Embedding Leaderboard MTEB](https://lms.sdmdigital.id/pluginfile.php/635469/mod_page/content/47/image17.png)
*Gambar 13. Embedding Leaderboard MTEB*

Setup vector database dan masukkan data. Tahap pembangunan index biasanya berjalan otomatis sesuai dokumentasi masing-masing database.

### 5.3.5 Hands on: Implementasi Ingestion Sederhana

**Langkah 1: Instalasi Library**
```bash
pip install python-docx gradio langchain langchain-openai langchain-google-genai langchain-huggingface langchain-text-splitters langchain-chroma jinja2
```

**Langkah 2: Ekstraksi Elemen DOCX (Paragraph & Table)**
```python
import os, uuid, base64
from docx import Document as DocxDocument
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.text.paragraph import Paragraph
from docx.table import Table

def extract_docx_elements(docx_path: str) -> list[dict]:
    doc = DocxDocument(docx_path)
    elements = []
    image_rels = {}
    
    for rel_id, rel in doc.part.rels.items():
        if "image" in rel.target_ref:
            try:
                image_data = rel.target_part.blob
                image_rels[rel_id] = {
                    "data": base64.b64encode(image_data).decode(),
                    "format": rel.target_ref.split(".")[-1],
                }
            except: pass

    image_counter = 0
    for elem_idx, element in enumerate(doc.element.body):
        if isinstance(element, CT_P):
            para = Paragraph(element, doc)
            para_images = []
            for run in para.runs:
                for drawing in run.element.findall(".//{http://schemas.openxmlformats.org/wordprocessingml/2006/main}drawing"):
                    for blip in drawing.findall(".//{http://schemas.openxmlformats.org/drawingml/2006/main}blip"):
                        embed_id = blip.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed")
                        if embed_id in image_rels:
                            para_images.append({
                                "image_data": image_rels[embed_id]["data"],
                                "format": image_rels[embed_id]["format"],
                                "index": image_counter
                            })
                            image_counter += 1
            # Handling Text & Images
            if para.text.strip() or para_images:
                elements.append({"type": "paragraph", "content": para.text, "style": para.style.name, "images": para_images, "position": elem_idx})
        elif isinstance(element, CT_Tbl):
            table = Table(element, doc)
            table_data = [[cell.text for cell in row.cells] for row in table.rows]
            elements.append({"type": "table", "content": table_data, "rows": len(table.rows), "cols": len(table.columns), "position": elem_idx})
    return elements
```

**Langkah 3: Memproses Tabel ke Teks**
```python
def process_table_to_text(table_data: list) -> str:
    if not table_data: return ""
    headers = table_data[0]
    table_texts = ["[Table]:"]
    for i, row in enumerate(table_data[1:]):
        table_texts.append(f"\t Row {i}:")
        for j, cell in enumerate(row):
            table_texts.append(f"\t\t - {headers[j]}: {str(cell).replace('\n', ' ')}")
    return "\n".join(table_texts)
```

**Tabel 10. Contoh hasil konversi tabel**
| Wilayah | Jumlah Toko | Total Penjualan (Rp) | Pertumbuhan |
| :--- | :--- | :--- | :--- |
| Jakarta | 15 | 3.850.000.000 | +15,2% |

*Hasil teks:*
```text
[Table]:
 Row 0:
  - Wilayah: Jakarta
  - Jumlah Toko: 15
  - Total Penjualan: 3.850.000.000
```

**Langkah 4: Definisi Prompt & Class RAG**
Sistem menggunakan `SYSTEM_PROMPT` untuk asisten, `GENERATE_SUMMARY_PROMPT` untuk ringkasan dokumen, dan `IMAGE_CAPTION_PROMPT` untuk VLM.

**Langkah 5: Inisiasi Agent & VectorDB (Chroma)**
Gunakan `GoogleGenerativeAIEmbeddings` dan `ChatGoogleGenerativeAI`. Implementasikan tool `retrieval` menggunakan Python *closure*.

**Langkah 6: Pipeline Ingestion Lengkap**
1. Ekstraksi elemen -> 2. Map ke Markdown (dengan Captioning Gambar) -> 3. Gabungkan Dokumen -> 4. Header-based Chunking -> 5. Add to ChromaDB -> 6. Siapkan Agent.

---

#### Hasil Uji Coba

![Output Aplikasi Siap](https://lms.sdmdigital.id/pluginfile.php/635469/mod_page/content/47/image15.png)
*Gambar 14. Output yang menandakan aplikasi sudah siap digunakan*

![Tampilan Aplikasi](https://lms.sdmdigital.id/pluginfile.php/635469/mod_page/content/47/image11.png)
*Gambar 15. Tampilan aplikasi yang dibuat*

![Proses Ingestion](https://lms.sdmdigital.id/pluginfile.php/635469/mod_page/content/47/image19.png)
*Gambar 16. Tampilan saat menunggu proses ingestion selesai.*

![Ingestion Selesai](https://lms.sdmdigital.id/pluginfile.php/635469/mod_page/content/47/image12%20%281%29.png)
*Gambar 17. Tampilan ketika ingestion selesai*

![Interaksi Asisten](https://lms.sdmdigital.id/pluginfile.php/635469/mod_page/content/47/image5%20%281%29.png)
*Gambar 18. Interaksi dengan asisten*

![Informasi Gambar](https://lms.sdmdigital.id/pluginfile.php/635469/mod_page/content/47/image28.png)
*Gambar 19. Informasi berbentuk gambar dalam dokumen*

![Jawaban Gambar](https://lms.sdmdigital.id/pluginfile.php/635469/mod_page/content/47/image13.png)
*Gambar 20. Jawaban asisten berdasarkan caption gambar*

![Tabel Dokumen](https://lms.sdmdigital.id/pluginfile.php/635469/mod_page/content/47/image9.png)
*Gambar 21. Contoh tabel dalam dokumen*

![Respon Tabel](https://lms.sdmdigital.id/pluginfile.php/635469/mod_page/content/47/image5%20%282%29.png)
*Gambar 22. Respon model ketika ditanya mengenai detail tabel di atas*

### Mini Quiz
*(H5P Content Placeholder)*

> **Refleksi Singkat**
>
> Evaluasi RAG kini sangat mengandalkan LLM-as-a-judge untuk menilai recall, precision, faithfulness, relevance, dan correctness.
>
> Menurut Anda, apa tantangan terbesar ketika evaluasi otomatis seperti ini digunakan dalam sistem produksi? Apakah Anda melihat kebutuhan untuk tetap melakukan evaluasi manual oleh manusia, atau LLM-as-a-judge sudah cukup di sebagian besar kasus?