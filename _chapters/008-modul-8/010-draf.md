---
slug: modul-8-1
title: modul-8-1
---

# 4.1 Mengakses dan Berinteraksi dengan LLM

Sama seperti model ML pada umumnya, LLM perlu diintegrasikan dengan sebuah sistem agar bisa digunakan untuk menyelesaikan sebuah masalah. Bentuk integrasinya bermacam, mulai dari menjalankan modelnya secara langsung hingga membungkus akses model melalui sebuah layanan API. Submodul ini akan membahas berbagai macam cara mengintegrasikan LLM agar bisa diakses oleh sebuah sistem.

---

## 4.1.1 Berbagai Cara Mengakses dan Menjalankan LLM

Sebelum lanjut ke topik yang lebih teknikal, mari coba kita pahami dulu apa yang sebenarnya dimaksud dengan mengakses dan menjalankan LLM. Bayangkan anda memiliki sebuah sistem yang ingin menggunakan LLM. Sistem tersebut perlu mengirimkan sebuah input ke LLM dan menerima output dari LLM untuk diproses lebih lanjut.

Kalau dibuat analogi, aplikasi yang dibangun adalah sebuah "kantor" yang menerima "surat" untuk diproses oleh LLM, lalu "mengembalikan" surat tersebut kepada anda sang pengirim, untuk diproses lebih lanjut. Tentunya perlu ada "jalur" yang menghubungkan antara LLM dengan sistem kita. "Jalur" ini adalah cara kita mengakses LLM, dan proses mengirim input dan output dari dan ke LLM adalah "menjalankan" LLM.

![Gambar 2. Aplikasi yang dibuat memerlukan akses ke LLM](https://lms.sdmdigital.id/pluginfile.php/635455/mod_page/content/25/image51.png)
**Gambar 2.** Aplikasi yang dibuat memerlukan "akses" ke LLM agar bisa "memanggil" LLM untuk melakukan suatu hal.

Kita mengenal istilah **abstraksi** yang umum ditemui dalam dunia *engineering*. Abstraksi adalah penyederhanaan sebuah konsep atau sistem yang umumnya dilakukan agar seorang *engineer* bisa lebih berfokus pada penggunaan dari sebuah sistem tersebut. Bagaimana konsep tersebut dapat "ditransfer" ke LLM?

---

## 4.1.2 LLM dari Sudut Pandang Pengembang Aplikasi

Dalam modul-modul sebelumnya, sudah dipelajari cara menginisialisasi LLM, yaitu melalui inisiasi arsitekturnya menggunakan sebuah kode lalu memuat parameter hasil training. Selain itu, kita juga sudah mempelajari kalau melakukan *inference* LLM melibatkan beberapa tahap:

1.  Proses *tokenization* teks input.
2.  Menerapkan sebuah *chat template* pada input tersebut.
3.  Melakukan proses *inference* LLM.
4.  Melakukan sampling untuk menentukan token output, lalu menggunakan output ini menjadi input untuk menentukan token selanjutnya hingga seterusnya proses *generation* berjalan.
5.  Mengubah kembali output berbentuk token menjadi teks.

Masalahnya, langkah-langkah ini sangat spesifik terhadap detil implementasi dan memerlukan proses *setup* yang merepotkan setiap kali ingin menyiapkan LLM lalu menggunakannya untuk menghasilkan jawaban.

Seorang AI Engineer memandang LLM dengan cara yang pragmatis. Ketika menggunakan LLM, AI Engineer tidak terlalu memikirkan detail internal dari LLM yang ingin digunakan seperti proses *tokenization*, arsitektur yang digunakan, implementasi setiap *layer*, maupun mekanisme untuk melakukan *sampling*. Yang penting LLM bisa di-setup dengan mudah, diatur konfigurasi untuk menjalankannya, dan bisa dengan mudah dijalankan untuk menghasilkan sebuah jawaban.

Karena itu, sebuah LLM diabstraksi sehingga proses setup dan inference hanya sebatas:
*   Melakukan setup akses LLM (misalnya mendapatkan endpoint atau memuat library), lalu
*   Menjalankan LLM dengan input dan output berupa teks.

Proses menjalankan LLM ini diabstraksi menjadi sebuah "fungsi" yang memiliki detail sebagai berikut:
*   Parameter untuk menjalankan LLM seperti parameter dari proses *sampling* bisa diatur melalui sebuah *interface* untuk mengubah-ubah konfigurasinya, dan
*   Ketika menjalankan LLM, baik input maupun outputnya sudah langsung berbentuk teks.

Semua proses teknis lainnya dikerjakan secara internal oleh implementasi fungsi tersebut menggunakan konfigurasi yang telah diberikan.

![Gambar 3. Ilustrasi LLM black box](https://lms.sdmdigital.id/pluginfile.php/635455/mod_page/content/25/image35.jpg)
**Gambar 3.** Ilustrasi LLM yang dipandang sebagai *black box* yang menerima input teks, output teks, serta bisa diatur melalui konfigurasi.

Dalam industri, umumnya ada dua bentuk utama abstraksi untuk mengakses dan menjalankan LLM:
*   Sebuah layanan API yang bisa dipanggil menggunakan HTTP Request dan sejenisnya, atau
*   Sebagai sebuah fungsi / library dalam suatu bahasa pemrograman.

Alih-alih harus secara manual me-load parameter model yang sudah dilatih dan menginisialisasi arsitektur model yang ingin digunakan, proses setup menjadi jauh lebih sederhana karena seluruh detail internal tersebut sudah dibungkus dalam abstraksi.

---

## 4.1.3 Menggunakan API LLM

Salah satu metode yang umum dipakai untuk mengakses LLM adalah melalui API. Cara ini ideal terutama untuk developer aplikasi yang ingin mengembangkan aplikasi AI tanpa perlu memiliki *hardware* dengan spesifikasi yang tinggi.

### Bagaimana LLM bisa diakses melalui API?

Karena LLM membutuhkan *resource* yang cukup signifikan bahkan hanya untuk menjalankannya, beberapa penyedia layanan atau sebut saja vendor berinisiatif membuat layanan LLM yang bisa diakses melalui API. Mereka melakukan *deployment* LLM sendiri yang mereka bungkus dalam sebuah sistem yang dikhususkan untuk menjalankan LLM. Pengguna layanan bisa mengakses dan menjalankan LLM menggunakan API yang disediakan oleh vendor.

Untuk mengakses sebuah layanan API, dibutuhkan dua hal. Yang pertama adalah **autentikasi**, yang umum dimiliki berbagai layanan API dalam bentuk **API Key** untuk membatasi hanya pihak tertentu saja yang diperbolehkan menggunakan API tersebut. Yang kedua adalah **endpoint** atau simpelnya bisa disebut sebagai URL atau alamat yang bisa digunakan untuk memanggil API tersebut.

Kalau dibuat analogi, interaksi dengan LLM menggunakan API bisa dianalogikan sebagai mengirim surat ke sebuah LLM yang berisi **input** yang ingin diproses serta **deskripsi bagaimana input tersebut harus diproses**. **Endpoint** merupakan **alamat** yang kita tuju untuk mengirimkan surat tersebut. Untuk masuk ke alamat tersebut, kita perlu memberikan **"kata sandi"** dalam bentuk **API Key**. Setelah mengirimkan surat, kita menerima respon dari LLM yang selanjutnya bisa digunakan dalam aplikasi yang dibuat.

Setiap vendor memiliki endpoint masing-masing dan juga memerlukan autentikasi yang unik untuk setiap penggunanya. Output dari LLM akan diterima melalui HTTP response yang dikirimkan kembali oleh API LLM yang kemudian tinggal diproses oleh aplikasi AI yang menggunakan API tersebut. Umumnya format ini menggunakan JSON yang strukturnya ditentukan oleh penyedia layanan.

![Gambar 4. Detil API OpenAI](https://lms.sdmdigital.id/pluginfile.php/635455/mod_page/content/25/image47.png)
**Gambar 4.** Detil mengenai penggunaan API openAI untuk mengakses LLM mereka yang bisa dibaca pada dokumentasi resmi mereka.

Ada banyak vendor yang menyediakan API LLM, misalnya OpenAI, Anthropic, Gemini, Mistral, dan masih banyak lagi. Vendor agregator seperti **OpenRouter** mengabstraksi pengaksesan berbagai macam API LLM ini dengan memberikan satu endpoint dan format data yang bisa digunakan untuk berinteraksi dengan LLM dari berbagai vendor dengan mudah.

---

## 4.1.4 Hands on: Mendapatkan Akses dan Menjalankan LLM Menggunakan API Gemini

Beberapa vendor menyediakan *free tier* atau layanan gratis dengan pilihan model dan batasan penggunaan tertentu. Salah satunya adalah Gemini milik Google.

Langkah pertama adalah membuka website [https://aistudio.google.com](https://aistudio.google.com) kemudian melakukan registrasi atau log in.

![Gambar 5. Tampilan utama Google AI Studio](https://lms.sdmdigital.id/pluginfile.php/635455/mod_page/content/25/image29.png)
**Gambar 5.** Tampilan utama Google AI Studio.

Pilih `Get API key` yang berada pada pojok kiri bawah tampilan layar untuk membuka dashboard manajemen API key.

![Gambar 6. Dashboard manajemen API Key](https://lms.sdmdigital.id/pluginfile.php/635455/mod_page/content/25/image40.png)
**Gambar 6.** Dashboard manajemen API Key.

Gunakan tombol `Create API key` untuk membuat key baru. Anda akan diminta mengisi nama dan memilih/membuat project.

![Gambar 7. Menu membuat API key](https://lms.sdmdigital.id/pluginfile.php/635455/mod_page/content/25/image44.png)
**Gambar 7.** Menu untuk membuat API key baru (atas) dan menu untuk membuat sebuah project (bawah).

![Gambar 8. Tampilan detil API key](https://lms.sdmdigital.id/pluginfile.php/635455/mod_page/content/25/image19.png)
**Gambar 8.** Tampilan detil API key yang sudah selesai dibuat.

### Contoh Penggunaan dengan Library `genai`

```python
from google import genai

client = genai.Client(api_key="GANTI DENGAN API KEY")

response = client.models.generate_content(
    model="gemini-2.5-flash", contents="Siapa yang menemukan nasi uduk?"
)
print(response.text)
```

**Output:**
> Tidak ada satu orang pun yang secara spesifik diakui sebagai "penemu" nasi uduk. Seperti kebanyakan hidangan tradisional yang sudah lama ada, nasi uduk diperkirakan berkembang secara bertahap seiring waktu di tengah masyarakat, khususnya di wilayah Jakarta dan sekitarnya... (dst)

### Memanggil API melalui Endpoint Langsung (curl)

```bash
curl "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent" \
  -H "x-goog-api-key: $GEMINI_API_KEY" \
  -H 'Content-Type: application/json' \
  -X POST \
  -d '{
    "contents": [
      {
        "parts": [
          {
            "text": "Siapa yang menemukan nasi uduk?"
          }
        ]
      }
    ]
  }'
```

### Memanggil API melalui Python `requests`

```python
import requests

header = {
    "x-goog-api-key": "GANTI DENGAN API KEY",
    "Content-Type": "application/json",
}

body = {"contents": [{"parts": [{"text": "Siapa yang menemukan nasi uduk?"}]}]}

endpoint = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"

response = requests.post(
    endpoint,
    json=body,
     headers=header
)

print(response.json()["candidates"][0]["content"]["parts"][0]["text"])
```

![Gambar 9. Dokumentasi API Gemini](https://lms.sdmdigital.id/pluginfile.php/635455/mod_page/content/25/image10.png)
**Gambar 9.** Contoh dokumentasi salah satu endpoint pada API Gemini yang menunjukkan endpoint serta detail cara penggunaan.

---

## 4.1.5 Mengenal OpenAI API Format

Terdapat sebuah "standar" yang umum digunakan, yaitu **OpenAI API Format**. Sebagian besar vendor lain dan software *self-hosting* mendukung format ini untuk memudahkan developer berpindah vendor tanpa mengubah banyak kode.

Endpoint utama yang umum dipakai:
1. `GET /models`: Melihat model tersedia.
2. `POST /chat/completions`: Endpoint utama interaksi chat.

### Contoh Menggunakan Library `openai` untuk Gemini

```python
from openai import OpenAI

client = OpenAI(
    api_key="GANTI DENGAN API KEY",
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

response = client.chat.completions.create(
    model="gemini-2.5-flash",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "Siapa yang menemukan nasi uduk?"
        }
    ]
)

print(response.choices[0].message.content)
```

---

## 4.1.6 Self-hosting LLM

Akses API memiliki batasan (internet, latency, kuota). Untuk model custom atau privasi tinggi, anda bisa melakukan *self-hosting* menggunakan *inference server* sendiri.

### Komponen Utama:
1. **Hardware**: RAM (untuk CPU) atau VRAM (untuk GPU).
2. **Software**: Inference server (Ollama, vLLM, dll).
3. **Model**: File model (biasanya dari Hugging Face).

#### Kebutuhan Hardware & Quantization
**Quantization** adalah teknik kompresi model dengan mengubah tipe data parameter (misal float32 ke int4) untuk mengurangi kebutuhan memori dan mempercepat proses tanpa kehilangan performa yang signifikan.

**Tabel 1. Petunjuk sederhana kapasitas memori vs ukuran model**

| Kapasitas memory (RAM/VRAM) | Jumlah parameter model (quantized 4-bit) |
| :--- | :--- |
| 4 GB | 3B |
| 8 GB | 7-8B |
| 16 GB | 13-14B |
| 32 GB + | ~30B ke atas |

---

## 4.1.7 Hands-on: Self hosting Menggunakan Ollama

Ollama adalah pilihan termudah untuk eksplorasi lokal.

**Tabel 2. Tooling untuk menjalankan LLM**

| Software | Kelebihan | Kekurangan |
| :--- | :--- | :--- |
| **Ollama** | Sangat mudah, beginner friendly, fitur lengkap. | Kurang fleksibel untuk produksi besar. |
| **llama.cpp** | Ringan, portabel, dukungan CPU terbaik. | Fokus pada penggunaan lokal sederhana. |
| **vLLM** | Production-ready, optimasi GPU tinggi. | Overkill untuk prototyping, dukungan CPU kurang. |

### Mendapatkan Model dari Hugging Face

Anda bisa mencari model di [huggingface.co/models](https://huggingface.co/models).

![Gambar 10. Tampilan halaman models pada Hugging Face](https://lms.sdmdigital.id/pluginfile.php/635455/mod_page/content/25/image36.png)
**Gambar 10.** Tampilan halaman `models` pada Hugging Face.

![Gambar 11. Tampilan model card](https://lms.sdmdigital.id/pluginfile.php/635455/mod_page/content/25/image27.png)
**Gambar 11.** Tampilan model card pada sebuah repository model.

### Menjalankan Ollama
Setelah instalasi, cek versi:
```bash
ollama -v
```

Menjalankan model (contoh: Smollm2):
```bash
ollama run smollm2:360m
```

![Gambar 16. Tampilan output ollama](https://lms.sdmdigital.id/pluginfile.php/635455/mod_page/content/25/image46.png)
**Gambar 16.** Tampilan output setelah menjalankan perintah run.

Untuk melihat daftar model lokal:
```bash
ollama ls
```

![Gambar 17. Tampilan hasil ollama ls](https://lms.sdmdigital.id/pluginfile.php/635455/mod_page/content/25/image24.png)
**Gambar 17.** Tampilan hasil perintah `ollama ls`.

---

## 4.1.8 Pertimbangan dalam Memilih Cara Mengakses LLM

1.  **Gunakan API sebagai default**: Jika internet reliabel, tidak ada data sangat sensitif, dan ingin kemudahan.
2.  **Self-host jika**:
    *   Sangat sensitif terhadap latency/internet.
    *   Membutuhkan privasi data tinggi (organisasi/bisnis).
    *   Menggunakan model sangat custom.
    *   Eksplorasi/Prototyping mandiri dalam jumlah besar (hemat biaya).

---

## 4.1.9 Pertimbangan dalam Memilih Model LLM

### Kapabilitas yang Diperlukan:
*   **Modalitas**: Teks saja, atau butuh Gambar/Audio?
*   **Fitur Umum**: Coding, bahasa tertentu (Indonesia), tool calling, atau reasoning.
*   **Ukuran vs Resource**: Model besar lebih pintar tapi butuh hardware kuat dan lambat.

### Evaluasi Performa:
*   **Jangan hanya percaya Benchmark**: Skor tinggi belum tentu cocok untuk kasus spesifik anda.
*   **Evaluasi Spesifik Kebutuhan**: Buat skenario dunia nyata (misal: simulasi komplain customer e-commerce) dan nilai hasilnya secara kualitatif maupun kuantitatif.

---

## 4.1.10 Interaksi Dasar dengan LLM

### Apa itu Prompt?
Input teks yang berisi instruksi dan konteks untuk mengarahkan LLM mencapai tujuan tertentu.

### System Prompt dan User Prompt
*   **System Prompt**: Bagian statis yang mengatur perilaku, gaya bahasa, dan aturan (misal: "Kamu adalah asisten yang sopan").
*   **User Prompt**: Bagian dinamis berisi input spesifik dari pengguna.

![Gambar 20. System dan User Prompt](https://lms.sdmdigital.id/pluginfile.php/635455/mod_page/content/25/image28.png)
**Gambar 20.** Berbagai vendor memisahkan system prompt dan user prompt.

---

## 4.1.11 Framework untuk Mengembangkan Aplikasi LLM

Framework membantu orkestrasi workflow yang kompleks (seperti RAG atau Agentic system).

**Tabel 3. Framework populer untuk membangun aplikasi AI**

| Nama | Fitur Utama | Cocok Untuk |
| :--- | :--- | :--- |
| **LangChain** | Orkestrasi workflow luas, integrasi banyak tool. | Prototyping cepat, sistem multi-agent. |
| **LlamaIndex** | Indexing data dan query engine. | Sistem yang berfokus pada Retrieval (RAG). |
| **Pydantic-ai** | Validasi output terstruktur. | Ekstraksi data yang reliabel dan predictable. |

---

## 4.1.12 Hands on: Berbagai Pola Interaksi Sederhana dengan LLM

### Contoh Interaksi dengan Gaya Bahasa (Jaksel Style)

```python
response = client.chat.completions.create(
    model="gemini-2.5-flash",
    messages=[
        {"role": "system", "content": "Respond to user's queries in Indonesian. Use jaksel-style language. Output must be in plain text only, no formatting."},
        {"role": "user", "content": "Jelaskan padaku secara singkat, apa itu AI?"}
    ]
)
print(response.choices[0].message.content)
```
**Hasil:** "AI itu basically, Artificial Intelligence. Nah, itu tuh kayak gimana ya... It's pretty game-changing, sih..."

### Contoh Memproses Gambar (Multimodal)

![Gambar 21. Gambar kota tua](https://lms.sdmdigital.id/pluginfile.php/635455/mod_page/content/25/image7.jpg)
**Gambar 21.** Gambar yang diambil dari wikipedia kota tua.

Untuk memproses gambar, kita perlu mengubahnya menjadi format **Base64** lalu mengirimkannya ke model.

```python
# (Proses konversi gambar ke base64_url)
response = client.chat.completions.create(
    model="gemini-2.5-flash",
    messages=[
        {"role": "system", "content": "Respond in Indonesian. Use jaksel-style language."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Coba deskripsikan gambar apa ini?"},
                {"type": "image_url", "image_url": {"url": base64_url}},
            ],
        },
    ],
)
print(response.choices[0].message.content)
```

**Hasil:** "OMG, ini kayaknya Museum Fatahillah di Kota Tua, sih. The vibes-nya literally historical banget gitu..."

---

### Mini Quiz
*(Elemen H5P Interaktif)*

> **Refleksi Singkat**
> Setelah membaca penjelasan panjang tentang berbagai cara mengakses LLM, alasan memilih API atau self-host, serta langkah memilih model yang sesuai dengan kebutuhan aplikasi, bagaimana kamu akan menentukan metode akses LLM dan model yang tepat untuk proyekmu sendiri, dan faktor apa yang menurutmu paling kritis untuk dipertimbangkan berdasarkan konteks aplikasi yang ingin kamu bangun?