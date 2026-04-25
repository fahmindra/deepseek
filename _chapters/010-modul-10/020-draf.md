---
slug: modul-10-2
title: modul-10-2
---
Berikut adalah isi konten yang telah dirapikan ke dalam format Markdown sesuai dengan hirarki dan instruksi Anda:

***

# 6.2 Prompting untuk Output Terstruktur

---

Di subtopik sebelumnya, kita telah memetakan "medan perang": data tidak terstruktur yang kacau dan risiko halusinasi model. Sekarang, kita akan membahas senjata pertahanan pertama kita: Prompt Engineering.

Sebelum kita menulis satu baris kode Python untuk validasi, kita harus bisa menulis instruksi bahasa alami yang cukup kuat untuk "mengekang" kreativitas model. Dalam konteks ekstraksi data, Prompt Engineering bukanlah tentang mencari "mantra ajaib" agar model menjadi pintar. Ini lebih mirip seperti menulis instruksi kerja (SOP) yang detail untuk karyawan baru.

Bayangkan LLM sebagai seorang asisten cerdas yang sangat patuh, namun butuh arahan spesifik.

*   **Instruksi Buruk:** Jika Anda hanya berkata, "Tolong ambilkan datanya dari dokumen ini," asisten tersebut akan menebak-nebak apa yang Anda mau. Hasilnya bisa berupa rangkuman paragraf, daftar poin, atau tabel yang tidak rapi. Outputnya menjadi tidak pasti (acak).
*   **Instruksi Baik:** Sebaliknya, jika Anda berkata, "Cari Nama dan Umur dari teks ini. Tuliskan hasilnya dalam format JSON. Gunakan kunci 'nama' dan 'umur'. Jika data umur tidak ditemukan, wajib tulis 'null', jangan dikosongkan." Dengan instruksi ini, asisten memiliki panduan yang kaku. Hasilnya akan konsisten, terstruktur, dan dapat diprediksi (deterministik).

Di subtopik ini, kita akan membahas hierarki teknik memberikan "instruksi kerja" ini: mulai dari perintah dasar (Zero-Shot), pemberian contoh cara kerja (Few-Shot), hingga memberikan formulir isian lengkap (Schema Definition).

### 6.2.1 Format Instructions & Zero-Shot Prompting

Teknik paling dasar adalah memberikan instruksi format yang eksplisit dalam System Prompt. Ini sering disebut sebagai Format Instructions. Contohnya adalah Zero-Shot Prompting.

> **Apa maksud dari Zero-Shot Prompting (untuk Struktur)?**
> Memberikan instruksi kepada model untuk melakukan tugas tanpa memberikan contoh konkret input-output sebelumnya. Model harus mengandalkan pemahaman instruksinya saja.

**Elemen Kunci dalam Instruksi Format:**

1.  **Role Play:** Tetapkan persona (You are a precise data extraction algorithm).
2.  **Format Output:** Tentukan format secara eksplisit (JSON, CSV, XML).
3.  **Constraints (Batasan):** Apa yang harus dilakukan jika data hilang? (Misal: "Use 'N/A' for missing fields").
4.  **No-Chatter:** Instruksi untuk diam (Misal: "Output JSON only. Do not include explanations or preamble").

**Contoh Prompt (Zero-Shot):**

```text
SYSTEM:
You are an API that converts unstructured text into valid JSON.
Extract the following fields: "product_name", "price", "currency".
If a field is missing, return null.
Do NOT output markdown backticks (json). Just the raw JSON string.

USER:
Saya baru saja membeli laptop Gaming X15 harganya 15 juta rupiah.
```

Meskipun efektif untuk model cerdas (GPT-4/Claude 3), teknik ini sering gagal pada model kecil (Llama-3-8B) yang mungkin masih "bawel" (menambahkan teks pembuka: "Here is the JSON you asked for..."), yang akan merusak proses parsing di kode kita.

Meskipun teknik Zero-Shot sangat efisien dari segi penggunaan token, ia menuntut model untuk memiliki kemampuan penalaran abstrak yang tinggi. Model seringkali gagal menangkap nuansa format yang kompleks atau aturan implisit hanya dari instruksi verbal semata. Untuk menjembatani kesenjangan ini dan meningkatkan konsistensi pada model yang lebih kecil, kita perlu memberikan model sesuatu yang lebih konkret daripada sekadar aturan: kita perlu memberikan contoh nyata.

### 6.2.2 Few-Shot Prompting

Ketika instruksi saja tidak cukup, kita akan menggunakan contoh. Ini memanfaatkan kemampuan In-Context Learning (yang kita bahas di Modul 3.1). Teknik ini disebut Few-Shot Prompting.

> **Apa itu Few-Shot Prompting?**
> Menyertakan beberapa pasang contoh (Input Teks -> Output JSON yang diinginkan) di dalam prompt sebelum memberikan input data yang sebenarnya.

**Mengapa ini Krusial untuk Struktur?**
Contoh mengajarkan hal-hal implisit yang sulit dijelaskan dengan kata-kata:
*   Bagaimana menangani tanggal ("12 Jan" -> "2024-01-12")?
*   Bagaimana menangani nested structure (JSON bersarang)?
*   Bagaimana menangani kapitalisasi?

![Anatomi Few-Shot Terstruktur](https://lms.sdmdigital.id/pluginfile.php/635480/mod_page/content/11/image.png)
**Gambar 3. Anatomi Few-Shot Terstruktur**

Gambar di atas menunjukkan anatomi prompt yang robust. Dengan menyisipkan Blok Tengah (Few-Shot Examples), kita memanfaatkan mekanisme attention model untuk "meniru pola" (pattern matching). Model melihat contoh JSON yang valid di context history-nya, sehingga probabilitas ia menghasilkan JSON yang valid untuk input baru meningkat drastis.

Namun, menulis contoh JSON secara manual di dalam string prompt ("hard-coding") itu melelahkan, rentan error, dan sulit dikelola jika struktur data berubah. Dalam software engineering, kita terbiasa mendefinisikan struktur data menggunakan Class atau Schema, bukan string contoh. Bisakah kita memberikan definisi skema formal ini langsung kepada model sebagai instruksi?

### 6.2.3 Konsep: Output Schema Definition (Blueprint Data)

Di level engineering yang lebih matang, kita tidak lagi menulis JSON manual di dalam string prompt. Kita mendefinisikan Skema Data.

> **Apa itu Skema Data?**
> Sebuah definisi formal (biasanya menggunakan standar seperti JSON Schema atau definisi Class Pydantic) yang menjelaskan struktur data, tipe data (string/int), dan deskripsi setiap field.

Alih-alih berkata: "Ekstrak nama dan umur", kita memberikan model sebuah Blueprint:

```json
{
  "type": "object",
  "properties": {
    "nama": {"type": "string", "description": "Nama lengkap pelanggan"},
    "umur": {"type": "integer", "description": "Umur dalam tahun"}
  },
  "required": ["nama"]
}
```

Menyuntikkan skema ini ke dalam prompt (atau ke fitur function calling model) memberikan konteks semantik yang jauh lebih kaya bagi model untuk memahami apa yang sebenarnya kita cari. Ini adalah langkah awal menuju penerapan Guardrails (pagar pengaman) yang akan memastikan data yang keluar valid secara tipe.

Meskipun prompting yang baik (Few-Shot + Schema) bisa meningkatkan akurasi secara signifikan, ia tidak menjamin 100% determinisme. Model masih bisa "terpleset" dan menghasilkan JSON yang cacat. Untungnya, penyedia model modern telah menyadari masalah ini dan menyediakan fitur di level API untuk memaksa model agar patuh pada format. Fitur inilah yang disebut JSON Mode dan Constrained Decoding yang akan kita bahas di subtopik berikutnya.

### Mini Quiz
*(Konten interaktif tersedia pada platform LMS)*

> **Refleksi Singkat**
>
> Kita telah membahas Zero-Shot (Instruksi saja) dan Few-Shot (Instruksi + Contoh). Few-Shot terbukti jauh lebih akurat untuk format struktur. Namun, setiap contoh yang Anda masukkan ke dalam prompt memakan biaya token (context window). Jika Anda harus mengekstrak data dari 1 juta dokumen, dan Anda menyertakan 5 contoh panjang di setiap prompt, biaya API Anda bisa membengkak 5x lipat. Sebagai Al Engineer, bagaimana Anda menyeimbangkan trade-off antara Akurasi Format (butuh banyak contoh) dan Efisiensi Biaya (butuh sedikit token)? Apakah ada titik optimal, atau adakah teknik lain (seperti Fine-Tuning spesifik untuk format) yang lebih hemat jangka panjang?