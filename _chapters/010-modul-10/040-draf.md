---
slug: modul-10-4
title: modul-10-4
---
Berikut adalah isi konten yang telah dirapikan ke dalam format Markdown sesuai dengan hirarki dan instruksi Anda:

***

# 6.4 Validasi dan Parsing Output dengan Pydantic

---

Kita sudah berhasil memaksa model mengeluarkan JSON. Tapi JSON itu masih berupa data mentah yang berisiko.

*   Bagaimana jika model mengembalikan `{"age": "dua puluh"}` padahal database kita butuh Integer?
*   Bagaimana jika model lupa menyertakan field email yang wajib ada?
*   Bagaimana jika ada field tambahan yang tidak kita minta?

Dalam software engineering Python modern, standar emas untuk validasi data adalah **Pydantic**. Pydantic memungkinkan kita mendefinisikan "bentuk data" yang kita inginkan sebagai Class Python, dan ia akan otomatis memvalidasi serta mengonversi (*parse*) data JSON yang masuk agar sesuai dengan bentuk tersebut.

Di subtopik ini, kita akan belajar menggunakan Pydantic sebagai lapisan Validasi Struktural (*Gatekeeper*) terakhir sebelum data masuk ke sistem kita.

Istilah "penjaga gerbang" di sini berbeda dengan konsep Guardrails pada umumnya:
*   **AI Guardrails** biasanya berfokus pada penyaringan keamanan konten (misal: mencegah ujaran kebencian atau topik terlarang).
*   **Pydantic Gatekeeper** berfokus pada integritas tipe data. la memastikan bahwa output model mematuhi format JSON Schema yang telah ditentukan secara ketat.

Sebagai contoh, jika sistem database Anda mengharapkan format: `{"id": 123, "active": true}` Namun LLM menghasilkan: `{"id": "123", "active": "yes"}`. Pydantic akan bertindak sebagai gerbang yang mencegat data tersebut, memvalidasi strukturnya, dan (jika mungkin) memperbaikinya agar sesuai dengan standar skema sebelum diproses lebih lanjut.

### 6.4.1 Konsep: Data Validation & Schema Enforcement

Validasi data dengan LLM berbeda dengan validasi input user biasa. LLM seringkali "benar tapi salah format".

> **Apa itu Pydantic?**
> Pydantic adalah library Python untuk validasi data. Kita mendefinisikan class yang mewarisi `BaseModel`, dan kita mendeklarasikan tipe data untuk setiap field dengan memanfaatkan fitur Type Hinting.

**Lalu apa itu Type Hinting?**
Type Hinting adalah sintaks standar Python yang memungkinkan kita menuliskan secara eksplisit tipe data yang diharapkan (seperti `str` untuk teks, `int` untuk angka, atau `List` untuk daftar) langsung pada variabel kode. Pydantic menggunakan petunjuk tipe ini sebagai aturan validasi utama untuk menerima atau menolak data dari LLM.

**Keunggulan Pydantic untuk LLM:**
1.  **Type Coercion (Pemaksaan Tipe Cerdas):** Jika model memberikan string "25" untuk field `age: int`, Pydantic akan otomatis mengubahnya menjadi integer 25.
2.  **Schema Definition:** Pydantic bisa menghasilkan skema JSON otomatis dari kode Python, yang bisa kita umpankan balik ke LLM sebagai instruksi (prompt).
3.  **Error Handling:** Jika data tidak valid, Pydantic memberikan pesan error yang sangat spesifik (misal: "Field 'email' missing"), yang bisa kita gunakan untuk auto-retry (meminta model memperbaiki dirinya sendiri).

Memahami kemampuan Pydantic untuk memvalidasi dan memaksa tipe data adalah langkah awal yang krusial. Namun, agar "penjaga gerbang" ini tahu siapa yang boleh masuk dan siapa yang harus ditolak, kita perlu memberinya instruksi spesifik. Kita harus menerjemahkan kebutuhan bisnis yang abstrak (misal: "saya butuh resep") menjadi definisi kode Python yang kaku dan dapat dieksekusi.

### 6.4.2 Implementasi Kode: Mendefinisikan Skema Ekstraksi

Mari kita lihat bagaimana kita mengubah kebutuhan bisnis menjadi kode Pydantic yang ketat.

**Skenario:** Kita ingin mengekstrak data dari Resep Makanan. Data yang dibutuhkan:
*   Nama Resep (String)
*   Daftar Bahan (List of Strings)
*   Kalori (Integer, Opsional)
*   Apakah Pedas? (Boolean)

```python
# 1. Instalasi Pydantic (biasanya sudah ada di environment modern)
# !pip install pydantic
from pydantic import BaseModel, Field
from typing import List, Optional
import json

# 2. Mendefinisikan Skema Data
# Kita membuat Class yang mewarisi BaseModel
class RecipeExtraction(BaseModel):
    recipe_name: str = Field(
        ...,
        description="Nama resmi dari hidangan makanan."
    )
    ingredients: List[str] = Field(
        ...,
        description="Daftar bahan-bahan utama yang disebutkan."
    )
    calories: Optional[int] = Field(
        None,
        description="Total kalori per porsi. Isi None jika tidak disebutkan."
    )
    is_spicy: bool = Field(
        False,
        description="Boolean: True jika makanan terindikasi pedas, False jika tidak."
    )

# 3. Mengintip Skema JSON (Untuk Prompting)
# Pydantic bisa otomatis membuat JSON Schema untuk kita berikan ke LLM!
schema_json = RecipeExtraction.model_json_schema()
print("--- Skema Otomatis Pydantic ---")
print(json.dumps(schema_json, indent=2))
```

**Analisis Kode:**
*   `Field(..., description="...")`: Deskripsi ini sangat penting! Nanti, deskripsi ini akan dibaca oleh LLM sebagai instruksi semantik untuk memahami apa yang harus diisi di field tersebut.
*   `Optional[int]`: Memberi tahu Pydantic (dan LLM) bahwa field ini boleh kosong (null).
*   `List[str]`: Memaksa output menjadi array string, bukan satu string panjang yang dipisah koma.

Sekarang kita telah memiliki "cetak biru" (blueprint) yang kokoh dalam bentuk kelas Python. Namun, definisi skema ini hanyalah dokumen pasif sampai kita mengujinya dengan data nyata. Tantangan sebenarnya dimulai ketika skema yang kaku ini berhadapan dengan output LLM yang seringkali tidak terduga, kotor, atau sedikit melenceng dari aturan. Mari kita lihat bagaimana Pydantic menangani kekacauan tersebut secara otomatis.

### 6.4.3 Implementasi Kode: Validasi Output LLM

Sekarang, mari kita simulasikan bagaimana kita memvalidasi output (JSON mentah) yang datang dari LLM menggunakan kelas Pydantic yang sudah kita buat.

```python
import json
from pydantic import ValidationError

# Simulasi Output dari LLM (JSON String)
# Kasus 1: Output Sempurna
llm_output_perfect = """
{
    "recipe_name": "Nasi Goreng Gila",
    "ingredients": ["Nasi", "Telur", "Sosis", "Cabai Rawit"],
    "calories": 600,
    "is_spicy": true
}
"""

# Kasus 2: Output "Kotor" (Tipe data salah, tapi bisa diperbaiki Pydantic)
# Perhatikan: calories berupa string "500", is_spicy berupa string "false"
llm_output_dirty = """
{
    "recipe_name": "Bubur Ayam",
    "ingredients": ["Beras", "Ayam", "Kecap"],
    "calories": "500",
    "is_spicy": false
}
"""

# Kasus 3: Output Invalid (Gagal Validasi)
# Missing field 'ingredients'
llm_output_invalid = """
{
    "recipe_name": "Teh Manis",
    "calories": 100
}
"""

print("\n--- Test Validasi ---")

# Fungsi Validasi Sederhana
def validate_and_parse(json_str):
    try:
        # Parse string JSON ke Dict
        data_dict = json.loads(json_str)
       
        # Validasi dengan Pydantic
        # Ini adalah langkah "Magic"-nya
        recipe = RecipeExtraction(**data_dict)
       
        print(f"✅ Validasi SUKSES: {recipe.recipe_name}")
        print(f"   Data Terstruktur: {recipe}")
        print(f"   Tipe Kalori: {type(recipe.calories)}") # Bukti konversi tipe
       
    except json.JSONDecodeError:
        print("❌ Error: Output bukan JSON valid.")
    except ValidationError as e:
        print(f"❌ Error Validasi Pydantic:\n{e}")

print("1. Test Perfect Output:")
validate_and_parse(llm_output_perfect)

print("\n2. Test Dirty Output (Auto-Correction):")
validate_and_parse(llm_output_dirty)
# Perhatikan Pydantic akan otomatis mengubah "500" (str) jadi 500 (int)

print("\n3. Test Invalid Output:")
validate_and_parse(llm_output_invalid)
```

Mari kita telusuri bagaimana Pydantic bertindak sebagai "penjaga gerbang" yang cerdas:

1.  **Fungsi `validate_and_parse` (Mekanisme Pertahanan):** Fungsi ini membungkus proses ekstraksi dalam blok `try...except` yang berlapis.
    *   **Lapis 1 (`json.loads`):** Mengecek apakah string yang masuk adalah JSON valid secara sintaksis. Jika LLM lupa menutup kurung `}`, error akan ditangkap di `JSONDecodeError`.
    *   **Lapis 2 (`RecipeExtraction(**data_dict)`):** Ini adalah inti validasi. Kita membongkar dictionary (`**data_dict`) dan mencoba memaksanya masuk ke dalam cetakan class `RecipeExtraction`. Jika ada data yang tidak sesuai tipe atau field wajib yang hilang, Pydantic akan melempar `ValidationError`.

2.  **Analisis Kasus "Dirty Output" (The Magic of Coercion):** Perhatikan input pada Kasus 2: `"calories": "500"` (String).
    *   Secara teknis, ini salah tipe (karena kita minta `int`).
    *   Namun, Pydantic memiliki fitur **Type Coercion** cerdas. Ia "tahu" bahwa string "500" bisa diubah jadi angka 500.
    *   **Hasil:** Data lolos validasi dan dikonversi otomatis menjadi tipe yang benar. Ini membuat aplikasi jauh lebih tahan banting (*robust*) terhadap variasi output LLM.

3.  **Analisis Kasus "Invalid Output" (Error Reporting):** Pada Kasus 3, field `ingredients` hilang.
    *   Pydantic melempar error: `Field required`.
    *   Pesan error ini sangat berharga. Dalam sistem produksi, kita tidak hanya mencetak error ini ke log. Kita akan mengirim pesan error ini kembali ke LLM ("Hei, kamu lupa field 'ingredients', tolong perbaiki JSON ini") agar LLM melakukan koreksi diri (*self-correction*). Teknik otomatisasi inilah yang akan kita bahas di subtopik selanjutnya.

Dengan Pydantic, kita mengubah output LLM yang tidak pasti menjadi objek Python yang aman, bertipe kuat (*strongly typed*), dan siap dimasukkan ke database. Jika validasi gagal (seperti di kasus 3), Pydantic memberikan pesan error yang jelas, yang nantinya bisa kita gunakan untuk meminta LLM memperbaiki jawabannya (*self-correction*).

### Mini Quiz
*(Konten interaktif tersedia pada platform LMS)*

> **Refleksi Singkat**
>
> Kita telah melihat bahwa Pydantic sangat hebat dalam melakukan Type Coercion (misal: mengubah string "500" menjadi integer 500). Namun, refleksikan batasan dari fitur ini: Jika LLM mengekstrak harga produk sebagai "500 ribu rupiah" untuk field `price: int`, Pydantic akan memunculkan `ValidationError` karena string tersebut tidak bisa dikonversi langsung menjadi integer. Sebagai Al Engineer, apa strategi error handling yang akan Anda bangun jika Pydantic melempar error tersebut? Apakah Anda akan membuang datanya? Atau Anda akan mengirim pesan error tersebut kembali ke LLM ("Kamu salah format, tolong perbaiki nilai '500 ribu rupiah' menjadi angka saja") untuk melakukan Auto-Correction?