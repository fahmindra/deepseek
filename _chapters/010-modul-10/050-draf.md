---
slug: modul-10-5
title: modul-10-5
---
Berikut adalah isi konten yang telah dirapikan ke dalam format Markdown sesuai dengan hirarki dan instruksi Anda:

***

## 6.5 Implementasi Library Ekstraksi: Pipeline Otomatis

---

Di subtopik sebelumnya, kita melakukan validasi secara manual:
1.  Kirim prompt.
2.  Dapat JSON string.
3.  Coba parse dengan Pydantic.
4.  Jika error, print error.

Dalam aplikasi nyata, proses "Jika error -> Coba lagi" (Retry Logic) ini sangat rumit untuk ditulis dari nol setiap saat. Kita perlu menghitung berapa kali boleh retry, menyimpan history error agar LLM tidak mengulangi kesalahan yang sama, dan mengelola parsing secara efisien.

Untungnya, komunitas open-source telah membangun framework orkestrasi yang menangani siklus Prompt -> Parse -> Validate -> Retry ini secara otomatis.

![Pipeline Otomatis Ekstraksi](https://lms.sdmdigital.id/pluginfile.php/635483/mod_page/content/15/image.png)
**Gambar 5. Pipeline Otomatis Ekstraksi**

Di subtopik ini, kita akan membahas dua pendekatan populer: LangChain Output Parsers (pendekatan klasik/agnostik) dan Instructor (pendekatan modern/Pydantic-native).

### 6.5.1 Konsep: The Extraction Loop (Self-Correction)

Sistem ekstraksi data yang tangguh (*robust*) tidak hanya mengandalkan satu kali percobaan (*single-shot*). la menerapkan siklus koreksi diri (*self-correction loop*). Ide dasarnya adalah memperlakukan pesan error dari validator (Pydantic) bukan sebagai kegagalan sistem, melainkan sebagai umpan balik (*feedback*) bagi model untuk memperbaiki outputnya sendiri.

![Alur Self-Correction Loop](https://lms.sdmdigital.id/pluginfile.php/635483/mod_page/content/15/image%20%281%29.png)
**Gambar 6. Alur Self-Correction Loop**

Gambar di atas mengilustrasikan mekanisme "ketahanan" (*resilience*) dalam pipeline ekstraksi data. Alih-alih berhenti saat terjadi kesalahan, sistem dirancang untuk berputar (*loop*) dan memperbaiki diri sendiri. Alur prosesnya adalah sebagai berikut:

1.  **Input Awal:** Proses dimulai dengan masuknya data tidak terstruktur (misal: teks email yang berantakan) dan instruksi skema (Pydantic Model).
2.  **Langkah 1: LLM Generation (Percobaan Pertama)**
    *   LLM membaca input dan mencoba menghasilkan output JSON.
    *   Status: Output ini masih berupa "Kandidat", belum tentu benar. Bisa jadi sintaksnya salah atau ada field yang hilang.
3.  **Langkah 2: Pydantic Validation (Pos Pemeriksaan)**
    *   Output kandidat diperiksa oleh validator Pydantic. Ini adalah "gerbang logika".
    *   Apakah JSON valid? Apakah tipe datanya benar? Apakah semua field wajib ada?
4.  **Cabang Keputusan (Decision Diamond):**
    *   **Jalur SUKSES (Valid):** Jika tidak ada error, data dianggap bersih dan langsung dikirim ke output akhir (misal: disimpan ke Database). Proses selesai.
    *   **Jalur GAGAL (Invalid):** Jika ditemukan error (misal: `ValidationError: Field 'age' missing`), proses masuk ke jalur koreksi.
5.  **Langkah 3: Error Formatting (Penerjemahan Masalah)**
    *   Sistem mengambil pesan error teknis dari Python dan mengubahnya menjadi instruksi bahasa alami yang bisa dipahami LLM.
    *   Contoh: Dari `ValueError: missing field` menjadi "Error: Kamu lupa mengisi field 'age'. Tolong generate ulang JSON-nya dan pastikan field 'age' terisi."
6.  **Langkah 4: Feedback Loop (Umpan Balik)**
    *   Instruksi perbaikan ini dimasukkan kembali ke LLM sebagai prompt baru (biasanya ditambahkan ke history percakapan).
    *   LLM melakukan Generasi Ulang (Retry) dengan mempertimbangkan kesalahannya sebelumnya.

Siklus ini berulang sampai data valid atau batas maksimum percobaan (*max retries*) tercapai. Ini mengubah LLM dari mesin yang "sekali jalan" menjadi agen yang mampu melakukan refleksi dan perbaikan diri.

Konsep loop koreksi diri ini sangat powerful, namun mengimplementasikan aliran data yang berputar balik (*cyclical*) secara manual membutuhkan penanganan state yang rumit. Bagaimana jika ada library yang bisa membungkus seluruh diagram rumit di atas menjadi satu rantai eksekusi sederhana? Inilah yang ditawarkan oleh framework orkestrasi seperti LangChain.

### 6.5.2 Pendekatan LangChain: Pydantic OutputParser

LangChain adalah framework orkestrasi LLM yang paling populer. la memiliki modul khusus `PydanticOutputParser` yang secara otomatis menyuntikkan instruksi format ke dalam prompt dan menangani parsing. Berikut adalah contoh implementasi kodenya:

```python
# !pip install Langchain Langchain-openai
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from pydantic import BaseModel, Field

# 1. Definisikan Data Model
class UserProfile(BaseModel):
    name: str = Field(description="Nama lengkap user")
    age: int = Field(description="Umur user")

# 2. Siapkan Parser Dasar
parser = PydanticOutputParser(pydantic_object=UserProfile)

# 3. Hubungkan ke LLM
model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# --- REVISI: Menambahkan Mekanisme Retry/Fixing ---
# Kita membungkus parser dasar dengan 'OutputFixingParser'.
# Jika parser dasar gagal, parser ini akan mengirim error + output salah
# kembali ke LLM untuk diperbaiki.
fixing_parser = OutputFixingParser.from_llm(parser=parser, llm=model)

# 4. Siapkan Prompt
prompt = PromptTemplate(
    template="Extract info.\n{format_instructions}\nInput: {query}\n",
    input_variables=["query"],
    # Kita gunakan instruksi dari parser yang sudah dibungkus
    partial_variables={"format_instructions": fixing_parser.get_format_instructions()},
)

# 5. Buat Chain (Menggunakan fixing_parser)
chain = prompt | model | fixing_parser

# 6. Eksekusi dengan Simulasi Data "Rusak"
try:
    # Misal input yang memancing jawaban ambigu
    input_query = "Namanya Sarah, dia lahir tahun 1995 (hitung umurnya sekarang)."
    
    # Invoke chain
    # Jika LLM awalnya salah format, fixing_parser akan otomatis Loop memperbaikinya
    output = chain.invoke({"query": input_query})
    
    print(f"Sukses Parsing:")
    print(f"Tipe Objek: {type(output)}")
    print(f"Data: {output}")
except Exception as e:
    print(f"X Gagal total setelah retry: {e}")
```

**Bedah Kode & Analisis (LangChain):**
1.  **PydanticOutputParser:** Ini adalah otak utamanya. Saat kita memberikan objek Pydantic (`UserProfile`) ke parser ini, ia tidak hanya diam. la menganalisis seluruh field dan tipe datanya.
2.  **`parser.get_format_instructions()`:** Inilah "sihir"-nya. Fungsi ini menghasilkan string instruksi yang sangat panjang (bisa 500+ token) yang berisi aturan JSON ketat. String ini disuntikkan ke variabel `{format_instructions}` di dalam prompt. Implikasi: Anda tidak perlu menulis "Please output JSON..." secara manual. LangChain menuliskannya untuk Anda dengan bahasa yang sangat teknis agar model patuh.
3.  **`chain.invoke(...)`:** Saat ini dipanggil, LangChain mengirim prompt + instruksi format ke LLM. Output mentah dari LLM kemudian ditangkap oleh parser. Jika formatnya cocok, ia langsung dikonversi menjadi objek `UserProfile`. Jika tidak, `OutputFixingParser` (komponen tambahan LangChain) bisa dipanggil untuk melakukan retry.
4.  **Mekanisme Retry & Latensi (`OutputFixingParser`):**
    *   **Implementasi:** Di baris `OutputFixingParser.from_llm(...)`, kita mengaktifkan "jaring pengaman".
    *   **Cara Kerja:** Saat parser dasar gagal memvalidasi output (misal: JSON tidak lengkap), `fixing_parser` tidak langsung crash. la menangkap error tersebut, lalu secara otomatis mengirim prompt baru ke LLM: "Hei, kamu menghasilkan JSON yang salah: [Error Message]. Tolong perbaiki."
    *   **Trade-off:** Ini menjamin Output yang Benar (100%), tetapi jika error terjadi, pengguna harus menunggu 2x lipat waktu (karena ada panggilan LLM kedua untuk perbaikan).

Meskipun LangChain sangat populer, pendekatannya memiliki kelemahan: ia sangat bergantung pada prompt engineering yang berat (instruksi teks panjang). Ini boros token dan kadang kurang stabil pada model yang tidak terlalu pintar mengikuti instruksi kompleks.

Apakah ada cara yang lebih "bersih" dan efisien yang memanfaatkan fitur native dari model modern (seperti OpenAI Function Calling) alih-alih hanya memanipulasi teks prompt? Inilah filosofi di balik library Instructor.

### 6.5.3 Pendekatan Modern: Instructor (Patched Client)

Instructor adalah library yang mengadopsi filosofi "Python-Native". Alih-alih menyuntikkan teks instruksi panjang ke dalam prompt, Instructor memanfaatkan fitur API Function Calling / Tool Use yang ada pada model seperti GPT-3.5/4. Ini membuat ekstraksi data terasa bukan seperti "mengobrol dengan bot", melainkan seperti memanggil fungsi Python biasa.

```python
import instructor
from openai import OpenAI
from pydantic import BaseModel, Field

# 1. Patching Client (The Magic)
client = instructor.from_openai(OpenAI())

# 2. Definisi Skema (Sama seperti sebelumnya)
class MovieReview(BaseModel):
    title: str = Field(description="Judul film")
    rating: float = Field(description="Skala 1-10")
    sentiment: str = Field(description="Positif, Negatif, atau Netral")

# 3. Eksekusi Ekstraksi
# Perhatikan parameter 'response_model' dan 'max_retries'
try:
    review = client.chat.completions.create(
        model="gpt-3.5-turbo",
        # KUNCI 1: Langsung minta objek Python, bukan teks
        response_model=MovieReview,
        # KUNCI 2: Auto-Retry Loop
        # Jika validasi Pydantic gagal, Instructor otomatis mengirim
        # error message ke LLM dan meminta ulang, maksimal 3x.
        max_retries=3,
        messages=[
            {"role": "user", "content": "Saya baru nonton Dune 2, visualnya gila banget! 9/10 deh."}
        ]
    )
    
    print(f"Judul: {review.title}")
    print(f"Rating: {review.rating}")
    print(f"Tipe Data Rating: {type(review.rating)}") # Float, bukan string!
except Exception as e:
    print(f"Gagal setelah retries: {e}")
```

**Bedah Kode & Analisis (Instructor):**
1.  **`instructor.from_openai(...)`:** Langkah ini memodifikasi metode `create` standar OpenAI. la menyuntikkan logika validasi di balik layar.
2.  **`response_model=MovieReview`:** Kita tidak lagi meminta "JSON". Kita meminta Instance Kelas Python. Di balik layar, Instructor mengubah skema Pydantic menjadi spesifikasi JSON Schema dan mengirimkannya lewat parameter `tools` atau `functions` di API OpenAI. Ini jauh lebih stabil daripada menaruhnya di system prompt.
3.  **`max_retries=3`:** Ini mengimplementasikan Gambar 4 secara otomatis. Jika LLM mengembalikan rating "sembilan" (string), Pydantic akan error. Instructor menangkap error itu, lalu mengirim request baru ke LLM: "Error: field 'rating' must be float. Fix it.". LLM kemudian mengembalikan 9.0. Proses ini transparan bagi pengguna. Kita hanya terima hasil bersih atau error final.

Dengan menggunakan library seperti Instructor atau LangChain, kita telah mengubah LLM dari sekadar "chatbot pintar" menjadi komponen perangkat lunak yang dapat diandalkan. Kita memindahkan beban validasi dan perbaikan error dari manusia ke mesin, memungkinkan otomatisasi skala besar yang sebelumnya tidak mungkin dilakukan dengan data tidak terstruktur.

**Tabel Perbandingan: LangChain vs Instructor**

| Fitur | LangChain Output Parsers | Instructor (Pydantic-Native) |
| :--- | :--- | :--- |
| **Pendekatan Utama** | Prompt Engineering: Menyuntikkan instruksi format teks panjang ke dalam prompt. | API Patching / Tool Use: Memanfaatkan fitur Function Calling native dari model modern. |
| **Kompleksitas Kode** | Tinggi: Membutuhkan definisi Parser, Template, dan Chain. | Rendah: Terasa seperti memanggil fungsi Python biasa; sangat "Pythonic". |
| **Ketergantungan Model** | Agnostik: Bisa bekerja dengan model lama (GPT-3, model lokal polos) yang tidak punya fitur function calling. | Modern-Only: Bekerja optimal dengan model cerdas yang mendukung Tool Use (GPT-4, Llama-3, Mistral). |
| **Efisiensi Token** | Boros: Instruksi format teks memakan banyak token di input context. | Efisien: Skema JSON dikirim lewat protokol API terpisah (Tools definition), lebih hemat token prompt. |
| **Best Practice (Saat Ini)** | Gunakan jika Anda harus mendukung berbagai jenis model (termasuk model lama/lemah) atau jika Anda sudah terlanjur menggunakan ekosistem LangChain yang luas. | Rekomendasi Utama: Gunakan untuk aplikasi produksi modern yang menggunakan OpenAI atau Open-Source LLM terbaru (via Ollama/vLLM). Lebih bersih, lebih stabil, dan lebih mudah di-debug. |

### Mini Quiz
*(Konten interaktif tersedia pada platform LMS)*

> **Refleksi Singkat**
>
> Kita telah melihat betapa mudahnya menggunakan `max_retries = 3` di Instructor untuk memperbaiki kesalahan model secara otomatis. Namun, sebagai Al Engineer yang sadar biaya, refleksikan: Apa dampak finansial dan latensi dari fitur ini? Setiap kali retry terjadi, kita melakukan pemanggilan API baru (mengonsumsi token input + output lagi). Jika Anda memproses 1 juta dokumen dan model Anda memiliki tingkat kesalahan awal 20% (membutuhkan 1x retry), berapa persen kenaikan biaya dan waktu proses yang Anda tanggung? Apakah lebih baik memperbaiki prompt agar benar dalam sekali coba (one-shot), atau mengandalkan retry loop?