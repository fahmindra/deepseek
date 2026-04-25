---
slug: modul-10-3
title: modul-10-3
---
Berikut adalah isi konten yang telah dirapikan ke dalam format Markdown sesuai dengan hirarki dan instruksi Anda:

***

# 6.3 Pemanfaatan JSON Mode dan Constrained Decoding

---

Di Subtopik 6.2, kita menggunakan Prompt Engineering (instruksi dan contoh) untuk membujuk model agar menghasilkan JSON. Namun, "bujukan" ini tidak menjamin hasil 100%.

*   Model mungkin lupa menutup kurung kurawal `}`.
*   Model mungkin tiba-tiba "ngobrol" di akhir JSON: `"...} Terima kasih!"` (ini merusak fungsi `json.loads()`).
*   Model mungkin salah menaruh koma.

Dalam sistem produksi, "mungkin berhasil" tidak cukup. Kita butuh determinisme. Untungnya, penyedia model modern (OpenAI, Anthropic, Ollama) kini menyediakan fitur di level inference engine yang disebut **JSON Mode**, atau secara teknis dikenal sebagai **Constrained Decoding**.

Mengapa disebut Constrained Decoding? Istilah ini merujuk pada proses membatasi (*constraining*) pilihan kata model saat melakukan generasi teks (*decoding*). Hubungannya dengan apa yang ia lakukan sangat harfiah: alih-alih membiarkan model bebas memilih sembarang kata dari kamus (probabilistik), mekanisme ini menerapkan batasan keras berupa aturan tata bahasa (*grammar*) JSON. la secara aktif "memblokir" model untuk memilih token apa pun yang akan merusak struktur JSON.

Di subtopik ini, kita akan membedah cara kerja mekanisme pemblokiran ini "di balik layar" dan cara mengimplementasikannya.

### 6.3.1 Konsep: JSON Mode (API-Level Enforcement)

JSON Mode adalah fitur yang disediakan oleh penyedia API seperti OpenAI GPT-4o atau Groq yang menjamin output model adalah string JSON yang valid.

**Bagaimana cara kerja JSON Mode?**
Sebuah parameter konfigurasi API (misal: `response_format={"type": "json_object"}`) yang mengaktifkan filter internal pada server model. Saat aktif, server akan menolak untuk menghentikan generasi (*stop generation*) sampai struktur JSON (kurung buka dan tutup) lengkap dan valid.

**Peringatan Penting:** Meskipun JSON Mode menjamin sintaks valid (kurungnya lengkap, komanya benar), ia tidak menjamin skema (isi key-nya benar).
*   Anda tetap harus memberi instruksi di prompt: "Extract fields 'name' and 'age'".
*   Jika tidak, model mungkin menghasilkan JSON valid tapi isinya ngawur: `{"random": "data"}`.

### 6.3.2 Di Balik Layar: Constrained Decoding (Grammar-Based Sampling)

Bagaimana JSON Mode bisa "memaksa" model? Apakah dia mengecek setelah selesai (*post-processing*)? Tidak, itu tidak efisien. la melakukannya saat proses generasi token demi token. Teknik ini disebut **Constrained Decoding**.

**Mekanisme Kerja (Logit Masking):**
Setiap kali model hendak memilih token berikutnya, algoritma "Penjaga Tata Bahasa" (*Grammar Guard*) akan mengecek: "Token mana saja yang VALID menurut aturan JSON saat ini?"

1.  Jika kita baru saja mencetak kurung kurawal buka `{`, token berikutnya harus berupa tanda kutip `"` (untuk memulai Key).
2.  Algoritma ini akan melihat daftar jumlah token di vokabulari (dimisalkan kita punya 50.000 token).
3.  la akan secara paksa memblokir semua token lain (seperti huruf atau angka) dengan memberikan nilai skor terendah yang mungkin (Minus Tak Terhingga).

**Dampaknya:** Saat skor ini diubah menjadi persentase (probabilitas), peluang untuk token-token terblokir tersebut menjadi 0%. Artinya, mustahil bagi model untuk memilihnya. Satu-satunya pilihan yang "hidup" hanyalah tanda kutip `"`. Akibatnya, saat masuk Softmax, probabilitas token selain `"` menjadi 0%. Model terpaksa memilih `"`.

Agar lebih mudah dalam memahami mekanisme kerja di atas, berikut adalah ilustrasinya.

![Mekanisme Constrained Decoding](https://lms.sdmdigital.id/pluginfile.php/635481/mod_page/content/12/image.png)
**Gambar 4. Mekanisme Constrained Decoding**

Diagram ini mengilustrasikan mekanisme penyaringan otomatis. Meskipun model sebenarnya 'ingin' memprediksi kata "Budi" (karena skor prediksinya cukup tinggi), mekanisme Constrained Decoding bertindak sebagai penjaga gerbang yang ketat. la tahu bahwa menurut aturan tata bahasa JSON, sebuah kata tidak boleh muncul tepat setelah nilai string tanpa pemisah.

Oleh karena itu, sistem secara paksa memblokir pilihan "Budi" (membuat peluangnya menjadi nol). Akibatnya, model tidak punya pilihan lain selain memilih opsi yang diizinkan aturan, yaitu tanda koma (`,`) atau tutup kurawal (`}`). Ini menjamin bahwa apa pun yang dipilih model, hasilnya pasti JSON yang sah.

Setelah memahami konsep "penjaga gerbang" ini secara visual, mari kita lihat bagaimana cara mengaktifkan fitur ini secara praktis menggunakan kode Python.

### 6.3.3 Implementasi Kode: OpenAI & Ollama

Dalam praktik engineering, kita tidak hanya mengandalkan prompt ("Tolong buat JSON"). Kita menggunakan parameter API khusus untuk mengaktifkan Constrained Decoding. Berikut adalah implementasi pada dua platform paling populer: OpenAI (Cloud) dan Ollama (Local/Open Source).

**1. Implementasi OpenAI (JSON Mode)**
Kode ini menunjukkan cara menggunakan parameter `response_format` yang diperkenalkan pada model GPT.

```python
from openai import OpenAI
import os
import json

# Pastikan API Key sudah diset di environment variable
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 1. System Prompt: Tetap WAJIB instruksikan JSON secara eksplisit
# JSON Mode akan gagal/error jika prompt tidak mengandung kata "JSON".
system_prompt = "You are a helpful assistant designed to output JSON."

# 2. User Prompt: Data tidak terstruktur
user_prompt = "Daftar buah: Apel merah, Pisang kuning, Anggur ungu."

print("Mengirim request ke OpenAI...")
response = client.chat.completions.create(
  model="gpt-3.5-turbo-0125", # Pastikan model support JSON Mode
  messages=[
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt}
  ],
  # --- KUNCI UTAMA: Mengaktifkan JSON Mode ---
  # Ini memberitahu API untuk memblokir token yang bukan sintaks JSON valid
  response_format={ "type": "json_object" }
)

# 3. Parsing Hasil
raw_content = response.choices[0].message.content
print(f"Output Mentah:\n{raw_content}")

# Kita bisa langsung parse karena dijamin valid secara sintaksis
json_data = json.loads(raw_content)
print(f"\nOutput Ter-parse (Python Dict): {json_data}")
```

**Analisis Kode (OpenAI):**
*   `response_format={ "type": "json_object" }`: Ini adalah baris terpenting. Tanpa ini, model mungkin menjawab dengan teks pembuka ("Tentu, ini JSON-nya..."). Dengan ini, model dipaksa untuk hanya mengeluarkan objek JSON murni `{...}`.
*   **Validitas:** Output dijamin valid secara sintaks (bisa di-`json.loads`), tetapi struktur kuncinya (*schema*) masih tergantung pada interpretasi model terhadap prompt.

**2. Implementasi Ollama (Local LLM)**
Ollama memungkinkan kita menjalankan model open-source (Llama 3, Mistral) secara lokal.

```python
import requests
import json

url = "http://localhost:11434/api/generate"

payload = {
  "model": "llama3",
  "prompt": "Extract info: Budi is 25 years old living in Jakarta.",
  "stream": False,
 
  # --- KUNCI UTAMA: Format JSON ---
  # Ollama akan menggunakan 'grammar constraints' (biasanya GBNF)
  # untuk memaksa model hanya menghasilkan token JSON yang valid.
  "format": "json"
}

print("Mengirim request ke Ollama Local...")
response = requests.post(url, json=payload)

# Ollama mengembalikan JSON respons yang berisi field 'response'
result_text = response.json()['response']
print(f"Output Mentah:\n{result_text}")

# Parsing
data = json.loads(result_text)
print(f"Umur (Parsed): {data.get('age', 'Tidak ditemukan')}")
```

**Analisis Kode (Ollama):**
Parameter `"format": "json"` pada Ollama bekerja dengan cara membatasi sampling model hanya pada token-token yang mematuhi tata bahasa JSON standar. Ini membuat model kecil (seperti Llama-3-8B) jauh lebih andal untuk tugas ekstraksi data dibandingkan jika hanya menggunakan prompting biasa.

### Mini Quiz
*(Konten interaktif tersedia pada platform LMS)*

> **Refleksi Singkat**
>
> Kita telah membahas bahwa fitur JSON Mode atau Constrained Decoding sangat ampuh untuk mencegah syntax error (seperti lupa tanda kurung atau koma). Namun, refleksikan: Apakah "JSON yang valid secara sintaksis" sama dengan "Data yang benar"? Jika Anda meminta model mengekstrak umur, dan karena halusinasi model menulis `{"age": "dua ratus tahun"}`, JSON tersebut valid secara sintaksis (bisa di-parse), tetapi tidak valid secara logika bisnis (schema database Anda mungkin butuh integer, bukan string). Apakah JSON Mode bisa menyelamatkan Anda dari kesalahan tipe data atau halusinasi nilai ini? Atau kita butuh lapisan pertahanan lain setelah ini?