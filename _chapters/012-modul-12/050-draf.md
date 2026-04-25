---
slug: modul-12-5
title: modul-12-5
---
Berikut adalah isi konten dari subtopik 8.5 yang telah dirapikan ke dalam format Markdown sesuai hirarki tanpa mengurangi isinya:

---

# 8.5 Kelemahan LLM-as-a-Judge: Bias dan Reliabilitas

Menggunakan LLM sebagai juri sangatlah efisien, tetapi apakah mereka objektif? Riset (seperti *Judging LLM-as-a-Judge*, 2023) menemukan bahwa LLM memiliki preferensi implisit yang kuat.

Mereka cenderung memberikan nilai tinggi bukan karena jawabannya benar, tapi karena jawabannya **panjang**, berada di **urutan pertama**, atau memiliki **gaya bahasa** yang mirip dengan dirinya sendiri.

Sebagai *AI Engineer*, tugas kita bukan hanya menggunakan juri ini, tetapi juga **mengaudit sang juri**. Di subtopik ini, kita akan membedah jenis-jenis bias utama dan teknik statistik untuk mengukur apakah juri kita bisa dipercaya (*reliable*).

## 8.5.1 Anatomi Bias Evaluator

Berikut adalah tiga bias paling umum yang merusak validitas skor evaluasi otomatis:

**1. Verbosity Bias (Bias Kepanjangan)**
*   **Fenomena:** Juri AI cenderung memberikan skor lebih tinggi pada jawaban yang lebih panjang, bertele-tele, dan tampak "pintar", meskipun isinya kurang akurat atau berulang-ulang (*yapping*).
*   **Dampak:** Model yang ringkas dan padat (efisien) seringkali dihukum dengan skor rendah.

**2. Position Bias (Bias Posisi)**
*   **Fenomena:** Dalam evaluasi perbandingan (*Pairwise Comparison*), jika kita memberikan dua jawaban: `[Jawaban A, Jawaban B]`, LLM memiliki kecenderungan statistik untuk memenangkan **Jawaban A** (yang muncul pertama).
*   **Dampak:** Jika kita menukar posisinya menjadi `[Jawaban B, Jawaban A]`, pemenangnya bisa berubah. Ini tanda ketidakkonsistenan fatal.

**3. Self-Preference Bias (Narsisme Model)**
*   **Fenomena:** GPT-4 cenderung memberi nilai lebih tinggi pada teks yang digenerate oleh GPT-4. Claude cenderung menyukai gaya Claude.
*   **Dampak:** Evaluasi menjadi tidak adil jika Juri dan Kandidat berasal dari keluarga model yang sama.

Berikut adalah ilustrasi dari salah satu Bias tersebut yaitu Bias Posisi:

![Position Bias pada LLM](https://lms.sdmdigital.id/pluginfile.php/635507/mod_page/content/17/image21.png)
**Gambar 4. Position Bias pada LLM**

Gambar di atas mengilustrasikan salah satu bias paling persisten dalam LLM yang disebut **Position Bias**. Diagram ini menunjukkan eksperimen sederhana:

1.  **Skenario 1 (Kiri):** Juri AI diberikan dua jawaban kandidat, A dan B. Urutannya adalah `[A, B]`. Juri memutuskan bahwa **A lebih baik**.
2.  **Skenario 2 (Kanan):** Juri diberikan dua jawaban yang sama persis, tetapi urutannya dibalik menjadi `[B, A]`. Secara logis, jika A lebih bagus, Juri harus tetap memilih A. Namun, dalam skenario bias ini, Juri malah memilih **B**.

**Kesimpulan:** Diagram ini membuktikan bahwa keputusan Juri tidak didasarkan pada **konten** jawaban, melainkan pada **posisi** jawaban tersebut dalam *prompt*. Seringkali, model memiliki *primacy bias* (lebih menyukai opsi pertama) atau *recency bias* (lebih menyukai opsi terakhir). Ini menandakan evaluasi yang tidak konsisten dan tidak valid.

Mengetahui keberadaan bias kognitif seperti *verbosity* dan *position bias* ini cukup mengkhawatirkan. Jika "hakim" kita ternyata tidak adil, apakah seluruh konsep *LLM-as-a-Judge* harus dibuang? Tidak perlu. Dalam *engineering*, setiap *noise* bisa diredam jika kita memiliki algoritma mitigasi yang tepat.

## 8.5.2 Mitigasi Bias

Berikut adalah teknik standar industri untuk menetralkan bias dan meningkatkan keadilan penilaian otomatis.

**1. Position Swapping (Pertukaran Posisi)**
Ini adalah solusi langsung untuk masalah pada Gambar 4. Jangan pernah menilai sekali. Lakukan penilaian dua kali dengan urutan yang dibalik.
*   **Mekanisme:**
    *   Pass 1: `Prompt("Bandingkan Jawaban A dan Jawaban B")` -> Hasil: A Menang.
    *   Pass 2: `Prompt("Bandingkan Jawaban B dan Jawaban A")` -> Hasil: A Menang.
*   **Keputusan:** Jika hasil konsisten (A menang di kedua posisi), maka penilaian valid. Jika hasil bertentangan (A menang di Pass 1, B menang di Pass 2), maka hasilnya adalah *Tie* (Seri) atau tidak valid.

**2. Reference-Guided Grading (Jangkar Kebenaran)**
Bias sering muncul ketika model "bingung" atau tidak punya standar.
*   **Solusi:** Jangan biarkan juri menilai di ruang hampa. Selalu sertakan *Ground Truth* (Referensi) dari Golden Set.
*   **Prompt:** "Jangan nilai mana yang terdengar lebih pintar. Nilai mana yang faktanya paling mendekati Referensi Kunci ini."
*   Ini memaksa model melakukan pencocokan fakta, bukan penilaian gaya bahasa.

**3. Chain-of-Thought (CoT) Requirement**
Memaksa model untuk menjelaskan alasan sebelum memberikan skor terbukti mengurangi bias *verbosity*.
*   **Mekanisme:** Jika model diminta langsung memberi skor, ia cenderung memilih teks panjang. Jika diminta *reasoning* dulu, ia terpaksa membaca isinya.
*   Seringkali model akan menulis: "Meskipun Jawaban B panjang, isinya berulang-ulang dan tidak menjawab poin utama." -> Skor rendah.

Setelah menerapkan strategi mitigasi di atas, kita bisa merasa lebih percaya diri dengan Juri AI kita. Namun, "rasa percaya diri" bukanlah metrik teknik. Kita membutuhkan angka. Bagaimana kita mengukur secara statistik apakah penilaian Juri AI ini benar-benar selaras (*aligned*) dengan standar penilaian manusia (kita), atau apakah ia masih memiliki agenda tersembunyi?

## 8.5.3 Implementasi Kode: Mengukur Reliabilitas (Agreement Score)

Salah satu cara mengukur kualitas Juri AI adalah membandingkannya dengan penilaian Manusia (*Human-AI Agreement*). Salah satu metrik statistiknya adalah *Cohen's Kappa* atau sekadar *Agreement Rate*.

Berikut kode untuk menghitung *Agreement Score* sederhana:

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

Kode di atas menghitung dua metrik vital untuk memvalidasi apakah Juri AI kita "sehati" dengan penilai manusia. Berikut adalah rincian formula dan target skor yang diharapkan:

**1. Exact Agreement Rate (Akurasi Mentah)**
*   **Formula:**
    $$Agreement = \frac{\text{Jumlah Nilai Sama}}{\text{Total Data}}$$
*   **Logika:** Seberapa sering AI dan Manusia memberikan skor yang persis sama (misal: Manusia beri 5, AI beri 5).
*   **Kelemahan Fatal:** Metrik ini rentan terhadap "faktor kebetulan". Jika 90% data Anda adalah skor 5, sebuah model bodoh yang selalu menebak angka 5 akan memiliki Agreement Rate 90%, padahal ia tidak "berpikir" sama sekali.

**2. Cohen's Kappa Score (Standard Emas)**
*   **Formula:**
    $$\kappa = \frac{p_{o} - p_{e}}{1 - p_{e}}$$
*   **Komponen:**
    *   $p_{o}$: *Observed Agreement* (Kesepakatan yang teramati, sama dengan metrik no 1).
    *   $p_{e}$: *Expected Agreement* (Kesepakatan yang terjadi murni karena kebetulan statistik).
*   **Logika:** Kappa menghapus faktor keberuntungan. Skor tinggi hanya diberikan jika AI setuju dengan manusia lebih sering daripada sekadar menebak acak.

### Panduan Interpretasi Skor (Threshold)
Nilai Kappa berkisar dari -1 sampai 1. Berikut adalah acuan untuk *AI Engineer*:

| Range Skor ($\kappa$) | Interpretasi Kualitas | Tindakan yang Disarankan |
| :---: | :--- | :--- |
| < 0.20 | Buruk (Poor) | **JANGAN DIPAKAI.** Juri AI Anda menilai secara acak atau bias. Perbaiki *Prompt* Rubrik Anda segera. |
| 0.21 - 0.40 | Kurang (Fair) | Tidak bisa diandalkan untuk otomatisasi penuh. Butuh revisi besar pada definisi kriteria penilaian. |
| 0.41 - 0.60 | Sedang (Moderate) | Boleh digunakan untuk *screening* awal, tapi wajib ada *Human-in-the-loop* (verifikasi sampel). |
| 0.61 - 0.80 | Bagus (Substantial) | **TARGET MINIMAL PRODUKSI.** Juri AI sudah cukup selaras dengan manusia. Aman untuk evaluasi skala besar. |
| > 0.81 | Sempurna (Perfect) | Sangat jarang terjadi. Juri AI setara dengan pakar manusia. |

**Tabel 1. Panduan Interpretasi Skor Kappa**

Kita telah belajar bahwa Juri AI tidak sempurna, tetapi dengan teknik mitigasi (*Swapping*) dan validasi statistik (Kappa Score), kita bisa menjadikannya alat yang andal.

Namun, menulis kode Python manual untuk melakukan *Position Swapping*, menghitung Kappa, dan mengelola *Golden Set* sangatlah melelahkan (*boilerplate code*). Di dunia *engineering* modern, kita tidak membangun alat evaluasi dari nol. Kita menggunakan *framework* yang sudah matang yang membungkus semua praktik terbaik ini (termasuk mitigasi bias) menjadi satu paket siap pakai.

Inilah saatnya kita berkenalan dengan RAGAS dan DeepEval, *tools* yang menjadi standar industri untuk audit kualitas LLM yang akan kita bahas di subtopik penutup kita berikutnya.

---

### Mini Quiz
*(Konten interaktif tersedia pada platform LMS)*

---

> **Refleksi Singkat**
>
> Kita tahu LLM memiliki *Verbosity Bias* (suka jawaban panjang). Jika Anda sedang mengevaluasi model untuk aplikasi SMS Singkat atau Notifikasi Push (di mana keringkasan adalah segalanya), bagaimana bias ini bisa membahayakan hasil evaluasi Anda? Langkah spesifik apa yang harus Anda tambahkan ke dalam *System Prompt* Juri AI Anda untuk mematikan bias ini?