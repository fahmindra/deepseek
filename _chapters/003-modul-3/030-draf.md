---
slug: modul-3-3
title: modul-3-3
---
### 3.3 Adaptasi untuk LLM Multimodal (Multi-modal Projectors)
---

Sejauh ini, kita telah berfokus secara eksklusif pada **teks.** Kita telah membedah arsitektur seperti Decoder-Only yang dilatih pada triliunan kata. Namun, dunia manusia tidak hanya berisi teks. Kita berkomunikasi dengan gambar, audio, dan video. Ini adalah tantangan besar: bagaimana kita bisa membuat model bahasa (LLM), yang "otaknya" dilatih untuk memahami vektor *embedding* dari kata, tiba-tiba "melihat" dan memahami piksel dari sebuah gambar?

![VLM, LLM Berbasis Gambar](https://lms.sdmdigital.id/pluginfile.php/635376/mod_page/content/24/image31.jpg?time=1767072257958)
**Gambar 6. VLM, LLM Berbasis Gambar**

Jika kita memberikan gambar (matriks piksel [R, G, B]) langsung ke LLM, model itu tidak akan mengerti apa-apa. Keduanya berbicara dengan bahasa yang jauh berbeda. Di subtopik ini, kita akan membedah arsitektur penghubung yang menjembatani kesenjangan modalitas ini yaitu sebuah istilah yang mengacu pada tipe atau jenis data, seperti teks, gambar, atau audio. Kita akan melihat arsitektur cerdas yang memungkinkan LLM untuk "melihat" gambar dan berbicara tentangnya.

#### 3.3.1 Konsep: Vision Encoders dan Projector "Penerjemah"

Masalah intinya adalah: piksel gambar dan *embedding* kata berada di "ruang vektor" yang sama sekali berbeda. Kita tidak bisa begitu saja menggabungkan keduanya. Solusinya adalah arsitektur hybrid yang terdiri dari tiga bagian, yang diilustrasikan pada Gambar 7.

![Arsitektur Vision Language Model (VLM) Sederhana](https://lms.sdmdigital.id/pluginfile.php/635376/mod_page/content/24/image30.jpg?time=1767072340613)
**Gambar 7. Arsitektur Vision Language Model (VLM) Sederhana**

Mari kita bedah alur datanya:

**Bagian 1: "Mata" (Vision Encoder)**
Kita tidak memberikan piksel mentah ke LLM. Pertama, kita memproses gambar melalui model terpisah yang khusus dilatih untuk "memahami" gambar. Model ini disebut Vision Encoder.
*   **Tugas:** Menerima gambar sebagai input dan mengubahnya menjadi serangkaian vektor *embedding* gambar (misal, 50 vektor yang mewakili 50 "potongan" gambar).
*   **Analogi:** Bertindak sebagai "mata". Ia memindai gambar dan mengubah apa yang dilihatnya menjadi sinyal-sinyal neural internal (vektor) yang mewakili "konsep" seperti "bulu", "telinga runcing", "langit biru", dll.

**Bagian 2: "Penerjemah" (Multi-modal Projector)**
Sekarang kita punya vektor gambar dari Vision Encoder dan vektor teks dari *embedding* LLM. Tapi keduanya masih dalam "bahasa" yang berbeda. Kita butuh penerjemah yaitu Projector.

> **Apa itu Projector?**
> Projector adalah sebuah lapisan neural network yang bertindak sebagai jembatan.
> **Tugasnya:** Menerima vektor gambar dan memproyeksikannya ke ruang vektor yang sama dengan vektor teks yang dipahami oleh LLM.

**Bagian 3: "Otak" (Language Model - LLM)**
Setelah vektor gambar "diterjemahkan" oleh Projector, arsitektur LLM Decoder-Only kini dapat memprosesnya.

**Alur Pemrosesan Gabungan:** Input untuk LLM sekarang adalah sebuah **sekuens gabungan** (*concatenated sequence*) yang terdiri dari:
1.  **Vektor Teks:** Misal, dari prompt "Apa yang ada di gambar ini?".
2.  **Vektor Gambar (yang sudah diproyeksikan):** Misal, 50 vektor yang mewakili "potongan" gambar.

LLM memperlakukan vektor gambar yang sudah diproyeksikan ini **seolah-olah mereka adalah token kata biasa**.

Arsitektur *hybrid* yang diuraikan pada **Gambar 7** (ViT + Projector + LLM) adalah cetak biru konseptual yang sangat kuat. Membangun dan melatih sistem tiga-bagian ini dari nol adalah tugas penelitian yang sangat kompleks. Untungnya, sebagai AI Engineer, kita tidak harus memulai dari nol. Sama seperti *tokenizer, library* transformers telah membungkus keseluruhan arsitektur VLM yang canggih ini ke dalam sebuah pipeline yang siap pakai. Kita dapat memanfaatkan model *pre-trained* yang sudah dilatih pada jutaan pasangan gambar-teks untuk menjalankan tugas "image-to-text" secara langsung.

#### 3.3.2 Implementasi Praktis: Pipeline image-to-text

Seperti yang telah kita bahas sebelumnya, kita tidak perlu membangun Vision Encoder atau Projector dari nol. *Library* transformers telah membungkus arsitektur kompleks ini ke dalam pipeline yang sangat sederhana.

Kita dapat menggunakan model pre-trained seperti **BLIP** dari Salesforce, yang menggunakan arsitektur *hybrid* mirip Gambar 7 untuk melakukan tugas Image Captioning yaitu memberi deskripsi pada gambar.

Berikut adalah cuplikan kode Python yang dapat dijalankan di Google Collab untuk mengimplementasikan VLM secara praktis.

```python
# Impor library yang dibutuhkan
from transformers import pipeline
from PIL import Image
import requests # Untuk mengunduh gambar contoh

# Memuat pipeline "image-to-text"
print("Memuat pipeline image-to-text...")
# Kita gunakan model BLIP yang kuat dari Salesforce
captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large")
print("Pipeline berhasil dimuat.")

# Menyiapkan gambar input
url = "https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png"
image = Image.open(requests.get(url, stream=True).raw)

# Anda bisa menampilkan gambar ini di notebook jika mau
# image.show()

# Menjalankan model
print("Memproses gambar untuk menghasilkan deskripsi...")
caption = captioner(image)

# Menampilkan hasil
print("\n--- Hasil Deskripsi Model ---")
# Outputnya mungkin sedikit bervariasi
print(caption[0]['generated_text'])
```

**Contoh Output yang Dihasilkan:**

```text
Memuat pipeline image-to-text...
Pipeline berhasil dimuat.
```
![Parrots Image](https://lms.sdmdigital.id/pluginfile.php/635376/mod_page/content/24/image40.png?time=1767072372253)
**Gambar 8. Contoh Gambar Input Pipeline**

```text
Memproses gambar untuk menghasilkan deskripsi...

--- Hasil Deskripsi Model ---
There are two parrots that are standing to each other
```

Analisis kode ini menunjukkan bahwa, sebagai AI Engineer, kita tidak perlu coding Projector-nya. Tugas kita adalah **memahami arsitektur** sehingga kita tahu pipeline apa yang harus digunakan dan model apa yang cocok untuk tugas tersebut.

#### Mini Quiz

*(Silakan akses kuis interaktif melalui platform LMS untuk menguji pemahaman Anda)*

> **Refleksi Singkat**
>
> Kita telah melihat bagaimana Projector (Gambar 7) bertindak sebagai Multi-modal yang menjembatani "bahasa gambar" dari Vision Encoder ke "bahasa teks" yang dapat dipahami LLM.
>
> Refleksikan: Menurut Anda, apa tantangan terbesar dalam melatih **Multi-modal Projector** ini? Mengapa sekadar memetakan vektor gambar 'kucing' ke vektor kata 'kucing' mungkin tidak cukup untuk pemahaman multimodal yang mendalam (misal, untuk memahami "kucing sedang melompat")?