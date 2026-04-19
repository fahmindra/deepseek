---
slug: modul-7-7
title: modul-7-7
---
# Praktikum Modul 2.3: Strategi Efisiensi Fine-Tuning dan Data Engineering

**Judul Praktikum:** Perancangan Strategi Fine-Tuning: Dari Hardware hingga Data
**Tujuan Praktikum:** Peserta mampu menganalisis kebutuhan sumber daya untuk pelatihan LLM, memilih teknik PEFT yang tepat (LoRA/QLoRA/Prompt Tuning), dan merancang strategi persiapan data (Chat Template) yang valid untuk kasus penggunaan spesifik.
**Deskripsi Singkat Aktivitas:** Anda berperan sebagai AI Solutions Architect. Perusahaan Anda ingin melakukan fine-tuning model open-source untuk berbagai departemen dengan keterbatasan hardware yang ketat. Anda harus mengevaluasi skenario teknis, memilih metode adaptasi yang paling efisien, dan memvalidasi format data sebelum pelatihan dimulai untuk mencegah kegagalan.

---

## Langkah-Langkah Praktikum

### 1. Siapkan Dokumen Jawaban
- Buat file dokumen baru: "Unjuk Kerja Modul 2.3 - [Nama Anda]".

### 2. Analisis Kendala Hardware & Metode (Konsep 2.3.1 & 2.3.2)
- **Skenario:** Tim IT memiliki satu server dengan GPU NVIDIA A10G (24GB VRAM). Mereka ingin melakukan fine-tuning pada model Llama-3-8B untuk tugas analisis sentimen internal.
- **Pertanyaan:**
    - **a.** Jelaskan secara matematis (estimasi kasar) mengapa Full Fine-Tuning (FFT) tidak mungkin dilakukan di GPU 24GB ini. (Petunjuk: Hitung estimasi beban Model + Gradien + Optimizer State).
    - **b.** Anda mengusulkan menggunakan QLoRA. Jelaskan mekanisme apa dalam QLoRA yang memungkinkan model tersebut muat dan bisa dilatih di GPU ini.

### 3. Analisis Strategi Data & Format (Konsep 2.3.5)
- **Skenario:** Anda menerima data mentah dari tim Customer Service berupa ribuan file log percakapan dalam format teks berantakan: `User: Halo. CS: Ya ada yang bisa dibantu? User: Komputer saya mati.`
- **Pertanyaan:**
    - **a.** Format standar industri mana (Alpaca atau ShareGPT) yang akan Anda pilih untuk merapikan data ini agar model bisa belajar konteks percakapan timbal-balik (multi-turn)? Jelaskan alasannya.
    - **b.** Mengapa sekadar mengubah data ke JSON tidak cukup? Jelaskan peran krusial Chat Template dan Special Tokens (seperti `<|im_start|>`) agar model tidak bingung membedakan mana pertanyaan User and mana jawaban CS.

![Perbedaan Format Alpaca dan ShareGPT](https://lms.sdmdigital.id/pluginfile.php/635443/mod_page/content/11/alpaca%20vs%20sharegpt.png)
*Gambar: Visualisasi perbedaan antara format JSON Alpaca dan ShareGPT*
*(Catatan: Gambar di atas adalah representasi visual untuk perbandingan format data)*

### 4. Analisis Efisiensi & Skalabilitas (Konsep 2.3.3)
- **Skenario:** Perusahaan ingin membuat fitur "Personalized AI" di mana setiap user (ada 10.000 user) bisa memiliki gaya bahasa bot mereka sendiri.
- **Pertanyaan:** Jika Anda menggunakan LoRA, Anda harus menyimpan 10.000 file adapter. Sebagai alternatif yang lebih ekstrem hemat penyimpanannya, teknik adaptasi ringan apa (dari Subtopik 2.3.3) yang hanya memanipulasi input layer tanpa menyentuh bobot internal sama sekali? Apa trade-off performanya dibandingkan LoRA?

### 5. Simpan & Kumpulkan
- Simpan file jawaban Anda dalam format .md, .txt, atau .pdf.

---

## Tools & Resource yang Dibutuhkan
- **Kalkulator VRAM (Opsional):** Bisa menggunakan referensi online atau hitungan manual dari materi modul untuk estimasi memori.
- **Dokumentasi Hugging Face:** Untuk referensi format Chat Templates.
- **Aplikasi Pengolah Kata:** Untuk menyusun laporan analisis.

---

## Contoh Output atau Hasil Akhir
Satu dokumen laporan yang berisi analisis logis:
- Perhitungan estimasi memori yang menunjukkan kegagalan FFT dan keberhasilan QLoRA.
- Justifikasi pemilihan format ShareGPT dan penjelasan teknis tentang fungsi Special Tokens.
- Perbandingan strategis antara LoRA dan Soft Prompting untuk kasus 10.000 user.

---

## Kriteria Keberhasilan / Penilaian Mandiri (Unjuk Kerja Modul 2.3)
1. Mampu memberikan estimasi perhitungan VRAM yang logis (menjelaskan bahwa training butuh memori jauh lebih besar dari ukuran model) dan menjelaskan secara teknis bagaimana QLoRA (melalui tipe data NF4) dan Paged Optimizers memecahkan bottleneck tersebut.
2. Mampu memberikan justifikasi yang tepat untuk pemilihan format ShareGPT dalam konteks percakapan multi-turn, serta menjelaskan fungsi teknis Special Tokens (seperti `<|im_start|>`) sebagai "pagar pembatas" agar model mengenali struktur dialog.
3. Mampu membandingkan trade-off penyimpanan antara LoRA (file adapter menengah) dan Soft Prompting (vektor input sangat kecil) dalam skenario ribuan pengguna, serta memilih solusi yang paling hemat storage namun tetap efektif.
4. Jawaban ditulis dengan argumen teknis yang runtut, menggunakan terminologi yang tepat (misal: Quantization, Token Embedding, Inference Latency), dan memberikan solusi konkret untuk setiap skenario.

---

## Petunjuk Troubleshooting
- **Bingung menghitung memori FFT?** Ingat "Aturan Jempol": Untuk training FP16, memori yang dibutuhkan sekitar 4x ukuran model. Model 8B FP16 $\approx$ 16GB. $16 \times 4 = 64$GB. Apakah 24GB cukup?
- **Bingung tentang Chat Template?** Bayangkan model sebagai mesin tik yang butuh tanda khusus untuk tahu kapan harus ganti baris dan ganti peran. Tanpa template, percakapan terlihat seperti satu paragraf novel yang datar.
- **Ragu antara Alpaca vs ShareGPT?** Alpaca = Satu Tanya, Satu Jawab. ShareGPT = Tanya, Jawab, Tanya lagi, Jawab lagi. Log CS biasanya tipe yang mana?

**Durasi Praktik Disarankan:** 60 Menit.

---

> **Aksi Lanjutan atau Refleksi (Opsional)**
> 
> Bayangkan jika di masa depan GPU menjadi 100x lebih murah dan VRAM bukan lagi masalah. Apakah teknik seperti PEFT/LoRA akan punah dan semua orang kembali ke Full Fine-Tuning, atau apakah PEFT tetap memiliki nilai guna (misalnya dalam hal portabilitas, modularitas adapter, atau keamanan bobot asli)?

---

*Klik [HANDS-ON: Modul 2.3 - Fine-Tuning Llama-3 dengan Unsloth](https://github.com/Modul-LLM/Module-07-2.3-Intermediate) untuk membuka sumber.*

---

# Penutup: Modul 3 Finetuning LLM

---

Jika Modul 2 adalah tentang "Menambah Pengetahuan" (CPT) yang mahal dan berisiko, Modul 3 adalah tentang "Mengasah Keterampilan" (Fine-Tuning) yang efisien dan taktis. Kita telah belajar bahwa untuk membuat model yang berguna di industri, kita tidak perlu selalu menggunakan "palu godam" (Full Fine-Tuning) atau GPU seharga rumah mewah.

Dalam modul ini, kita telah mendemokratisasi akses ke pengembangan LLM:

- **Efisiensi adalah Kunci (3.1):** Kita membuktikan bahwa PEFT memungkinkan kita mengadaptasi model dengan hanya melatih <1% parameter, mengubah paradigma penyimpanan dari "satu model per klien" menjadi "satu adapter per klien".
- **Inovasi Matematika (3.2):** Kita membedah LoRA (dekomposisi matriks rank-rendah) dan QLoRA (kuantisasi 4-bit), teknik yang memungkinkan model 7B-70B berjalan di GPU konsumen/gratisan.
- **Alternatif Ringan (3.3):** Kita mengeksplorasi teknik non-invasif seperti Soft Prompting dan Prefix Tuning yang memanipulasi input alih-alih bobot.
- **Anatomi MoE (3.4):** Kita melihat langsung ke dalam kode model Mixture of Experts, memahami bagaimana Router dan Experts diimplementasikan sebagai lapisan Linear yang kompleks namun efisien.
- **Data adalah Raja (3.5):** Kita mempelajari bahwa keberhasilan SFT sangat bergantung pada format data (Alpaca vs ShareGPT), penggunaan Chat Template yang benar, dan teknik efisiensi seperti Sequence Packing.
- **Kecepatan Eksekusi (3.6):** Kita menutup dengan Unsloth, library yang membuktikan bahwa optimasi kernel GPU bisa melipatgandakan kecepatan pelatihan tanpa mengubah arsitektur model.

### Tujuan Pembelajaran yang Dicapai:
- Anda kini dapat menerapkan strategi PEFT/LoRA untuk melatih model besar pada infrastruktur terbatas, mengatasi hambatan biaya hardware.
- Anda mampu menyiapkan dan memvalidasi dataset instruksi yang berkualitas tinggi, memahami pentingnya Chat Template dan Masking untuk mencegah model yang "rusak".
- Anda dapat mengoperasikan library modern (Transformers, PEFT, Unsloth) untuk menjalankan siklus fine-tuning dari awal hingga akhir (end-to-end).

---

> **Refleksi Akhir**
> 
> Sebagai penutup, mari kita renungkan pergeseran paradigma yang baru saja Anda pelajari. Dahulu, kekuatan AI diukur dari siapa yang punya GPU terbesar. Hari ini, dengan teknik seperti QLoRA dan Unsloth, hambatan hardware telah runtuh. Siapa pun dengan laptop gaming atau Google Colab bisa melatih model canggih.
> 
> Jika hardware bukan lagi pembeda utama, maka apa "parit pertahanan" (moat) Anda sebagai seorang AI Engineer? Jawabannya adalah Data. Kemampuan Anda untuk mengkurasi dataset instruksi yang unik, bersih, dan spesifik-domain (seperti di 2.3.5) kini jauh lebih berharga daripada kemampuan menyewa server mahal.
> 
> Apakah Anda siap beralih fokus dari "Model-Centric AI" (mencari model terbesar) menjadi "Data-Centric AI" (mencari data terbaik untuk fine-tuning)?

---

## Referensi
1. Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). Qlora: Efficient finetuning of quantized llms. Advances in neural information processing systems, 36, 10088-10115. [https://arxiv.org/pdf/2305.14314](https://arxiv.org/pdf/2305.14314)
2. Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., ... & Chen, W. (2022). Lora: Low-rank adaptation of large language models. ICLR, 1(2), 3. [https://arxiv.org/pdf/2106.09685](https://arxiv.org/pdf/2106.09685)
3. Li, X. L., & Liang, P. (2021). Prefix-tuning: Optimizing continuous prompts for generation. [https://aclanthology.org/2021.acl-long.353.pdf](https://aclanthology.org/2021.acl-long.353.pdf)
4. Houlsby, N., Giurgiu, A., Jastrzebski, S., Morrone, B., De Laroussilhe, Q., Gesmundo, A., ... & Gelly, S. (2019, May). Parameter-efficient transfer learning for NLP. In International conference on machine learning (pp. 2790-2799). PMLR. [https://arxiv.org/pdf/1902.00751](https://arxiv.org/pdf/1902.00751)