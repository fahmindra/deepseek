---
slug: modul-12-4
title: modul-12-4
---
Berikut adalah isi konten dari subtopik 8.4 yang telah dirapikan ke dalam format Markdown sesuai hirarki tanpa mengurangi isinya:

---

# 8.4 Pola LLM-as-a-Judge: Evaluasi Otomatis yang Berskala

Di subtopik sebelumnya (8.3), kita sudah bersusah payah membangun **Golden Set** (Soal Ujian + Kunci Jawaban). Sekarang, kita menghadapi masalah logistik: **Siapa yang akan memeriksa ribuan lembar jawaban ini?**

Jika Anda merekrut manusia (*annotator*), biayanya mahal dan lambat. Jika Anda menggunakan metrik BLEU/ROUGE (8.1), hasilnya tidak akurat.
Solusi modernnya adalah merekrut "mesin" untuk menilai "mesin". Inilah paradigma **LLM-as-a-Judge**.

Evaluasi otomatis tradisional (*String Matching*) gagal menangkap nuansa. Evaluasi manusia sangat akurat tapi tidak *scalable*.

**LLM-as-a-Judge** adalah jalan tengah yang manis. Konsepnya sederhana: Kita menggunakan model LLM yang sangat kuat dan cerdas (misal: GPT-4o atau Claude 3.5 Sonnet) sebagai **"Juri"** untuk mengevaluasi output dari sistem aplikasi kita (misal: RAG berbasis Llama-3-8B).

Mengapa ini berhasil? Karena model *state-of-the-art* (SOTA) telah terbukti memiliki korelasi yang sangat tinggi dengan penilaian manusia dalam tugas-tugas evaluasi. Mereka bisa membaca instruksi penilaian (rubrik), membandingkan dua teks, dan memberikan alasan logis (*reasoning*) mengapa satu jawaban lebih baik dari yang lain.

Di subtopik ini, kita akan membedah arsitektur penilaian ini dan menulis kode untuk "Juri AI" kita sendiri.

## 8.4.1 Konsep Arsitektur Penilaian

Dalam pola ini, "Juri" bukanlah sekadar model kosong. Ia adalah model yang diberi **konteks penuh**.
Input untuk Juri terdiri dari 4 komponen wajib:

1.  **Pertanyaan Asli (Input):** Apa yang ditanyakan user?
2.  **Jawaban Model (Candidate):** Apa yang dihasilkan aplikasi kita?
3.  **Kunci Jawaban (Reference):** Apa jawaban yang seharusnya (dari Golden Set)?
4.  **Rubrik Penilaian (Evaluation Prompt):** Instruksi kriteria penilaian (misal: "Nilai dari 1-5 berdasarkan akurasi fakta").

Agar lebih mudah memahami, berikut adalah ilustrasi bagaimana LLM-as-a-Judge bekerja:

![Arsitektur LLM-as-a-Judge](https://lms.sdmdigital.id/pluginfile.php/635506/mod_page/content/8/image25.jpg)
*Gambar 3. Arsitektur LLM-as-a-Judge*

Diagram ini menjelaskan mengapa **LLM-as-a-Judge** jauh lebih canggih daripada sekadar mencocokkan kata (BLEU/ROUGE). Agar sang Juri (GPT-4) bisa memberikan penilaian yang adil, ia membutuhkan **Konteks Lengkap (Triangulasi)**:

1.  **User Input vs. Candidate:** Juri mengecek **Relevansi**. Apakah jawaban model *nyambung* dengan pertanyaan user? (Tanpa melihat pertanyaan, kita tidak tahu apakah jawaban itu relevan).
2.  **Reference vs. Candidate:** Juri mengecek **Akurasi**. Juri membandingkan jawaban model dengan Kunci Jawaban (Reference). Apakah faktanya sama? Apakah ada halusinasi?
3.  **Rubrik (System Prompt):** Ini adalah "kacamata" Juri. Tanpa rubrik, Juri bingung: "Saya harus menilai apanya? Tata bahasanya? Atau faktanya?". Rubrik memaksa Juri fokus pada kriteria spesifik (misal: "Hukum berat jika ada fakta salah, abaikan gaya bahasa").

Hasil akhirnya bukan cuma angka (Skor), tapi juga **Alasan (Reasoning)**. Ini krusial untuk *debugging*. Jika skor rendah, kita bisa baca alasannya: "Oh, ternyata model saya sopan tapi faktanya salah."

Memahami arsitekturnya mudah, tetapi keajaiban sesungguhnya terletak pada **Prompt Penilaian (Rubrik)**. Jika Anda memberi instruksi yang samar kepada Juri ("Nilai apakah ini bagus"), hasilnya akan inkonsisten. Kita harus memprogram Juri tersebut layaknya kita memprogram fungsi Python: dengan input yang jelas, logika penilaian yang eksplisit, dan format output yang terstruktur.

## 8.4.2 Implementasi Kode: Membangun Juri AI dengan OpenAI

Berikut adalah implementasi fungsi `evaluate_answer` yang *robust*. Kita menggunakan teknik **JSON Mode** (dari Modul 2.6) untuk memastikan Juri memberikan skor yang bisa diolah secara programatis.

```python
from openai import OpenAI
import json
import os


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def evaluate_answer(question, reference, candidate_answer):
    """
    Fungsi Juri AI untuk menilai akurasi jawaban RAG.
    """
   
    # 1. Definisi Rubrik (System Prompt)
    # Kita menggunakan teknik 'Chain-of-Thought' di dalam instruksi
    # agar model berpikir dulu sebelum memberi nilai.
    judge_system_prompt = """
    Anda adalah evaluator AI yang objektif untuk sistem RAG.
    Tugas Anda: Bandingkan JAWABAN KANDIDAT dengan REFERENSI.
   
    Kriteria Penilaian (Skala 1-5):
    1 (Buruk): Jawaban salah total atau halusinasi.
    3 (Cukup): Jawaban benar sebagian tapi ada detail penting yang hilang.
    5 (Sempurna): Jawaban akurat, lengkap, dan sesuai dengan Referensi.
   
    FORMAT OUTPUT WAJIB (JSON):
    {
        "reasoning": "Penjelasan singkat analisis perbandingan...",
        "score": <integer 1-5>
    }
    """
   
    # 2. Menyusun Input User
    user_content = f"""
    [PERTANYAAN]: {question}
    [REFERENSI]: {reference}
    [JAWABAN KANDIDAT]: {candidate_answer}
   
    Berikan penilaian Anda dalam JSON.
    """
   
    # 3. Panggil API Juri (GPT-4)
    response = client.chat.completions.create(
        model="gpt-4-turbo", # Gunakan model terkuat untuk jadi juri
        messages=[
            {"role": "system", "content": judge_system_prompt},
            {"role": "user", "content": user_content}
        ],
        temperature=0, # Wajib 0 agar konsisten
        response_format={ "type": "json_object" } # Paksa JSON valid (Modul 2.6)
    )
   
    # 4. Parse Hasil
    return json.loads(response.choices[0].message.content)


# --- Simulasi Penggunaan ---
q = "Apa syarat pengajuan cuti sakit?"
ref = "Harus melampirkan surat dokter dan diajukan maksimal H+2 setelah masuk kerja."
cand = "Anda perlu surat dokter, tapi batas waktunya tidak disebutkan."


# Jalankan Evaluasi
hasil = evaluate_answer(q, ref, cand)


print(f"Skor: {hasil['score']}/5")
print(f"Alasan: {hasil['reasoning']}")
```

Skrip di atas bukan sekadar pemanggilan API biasa. Ia menerapkan beberapa **pola desain (*design patterns*)** krusial untuk evaluasi AI yang andal:

1.  **Strategi "Reasoning-First" (Chain-of-Thought)**: Perhatikan urutan dalam JSON yang diminta: `reasoning` dulu, baru `score`.
    *   **Mengapa?** Jika kita meminta skor terlebih dahulu (misal: `{"score": 5, "reasoning": "..."}`), model cenderung "menebak" angka secara intuitif, baru kemudian mengarang alasan untuk membenarkan angka tersebut (*rationalization*).
    *   **Solusi:** Dengan meminta penjelasan terlebih dahulu, kita memaksa model untuk melakukan "berpikir langkah demi langkah" (*Chain-of-Thought*). Ia menganalisis bukti, menimbang pro-kontra, dan baru kemudian menyimpulkan skor. Riset menunjukkan ini meningkatkan korelasi penilaian dengan manusia secara signifikan.
2.  **Determinisme Evaluasi (temperature=0)**: Parameter `temperature=0` adalah wajib hukumnya untuk evaluasi.
    *   **Masalah:** Evaluasi harus *reprodusibel*. Jika Anda menilai jawaban yang sama hari ini dan besok, skornya harus sama.
    *   **Solusi:** `temperature=0` mematikan kreativitas model. Kita ingin Juri yang dingin, kaku, dan konsisten, bukan Juri yang "kreatif" atau "*moody*".
3.  **Struktur "Role-Playing" (System Prompt)**: Kita mendefinisikan persona: "Anda adalah evaluator AI yang objektif."
    *   Ini bukan sekadar pemanis. Dalam *latent space* LLM, persona ini mengaktifkan pola neuron yang terkait dengan audit, kritik, dan objektivitas, mengurangi kemungkinan model menjadi terlalu "baik hati" atau penjilat (*sycophancy bias*).
4.  **Otomatisasi Penuh (response_format JSON)**: Kita menggunakan fitur **JSON Mode** (yang dipelajari di Modul 2.6).
    *   Tanpa ini, Juri mungkin akan menjawab: "Analisis saya adalah... Skornya 3." Teks narasi ini sulit diolah oleh komputer secara massal.
    *   Dengan JSON, hasil penilaian langsung menjadi data terstruktur yang bisa kita simpan ke *database*, hitung rata-ratanya, atau tampilkan di *dashboard*. Ini mengubah evaluasi dari tugas kualitatif menjadi kuantitatif.

**Kesimpulan:** Fungsi `evaluate_answer` ini adalah blok bangunan dasar. Dalam praktiknya, Anda akan memanggil fungsi ini di dalam *loop* untuk menilai ratusan baris data dari **Golden Set** Anda secara otomatis.

---

### Mini Quiz
*(Konten interaktif tersedia pada platform LMS)*

---

> **Refleksi Singkat**
>
> Kita menggunakan GPT-4 untuk menilai Llama-3. Secara filosofis, ini memunculkan pertanyaan: **"Siapa yang menjaga penjaga?"** Jika model Juri (GPT-4) memiliki bias tertentu (misalnya, lebih suka jawaban yang panjang dan bertele-tele), maka skor evaluasi kita juga akan bias.
>
> Sebagai *AI Engineer*, bagaimana Anda memvalidasi bahwa Juri AI Anda adil? Apakah Anda akan mempercayai skor Juri 100%, atau Anda perlu melakukan *sampling* manual (manusia menilai ulang penilaian Juri) secara berkala?