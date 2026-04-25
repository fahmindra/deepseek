---
slug: modul-3-6
title: modul-3-6
---
### Analisis Konseptual: Sampling, Multimodal, dan Arsitektur Efisien
---

**Tujuan Praktikum:**
Peserta mampu menganalisis konsep arsitektur Decoder-Only, membedakan berbagai strategi *sampling* (Temperature, Top-K/P), dan mengidentifikasi arsitektur canggih (Multimodal dan MoE) untuk kasus penggunaan spesifik.

**Deskripsi Singkat Aktivitas:**
Aktivitas ini meminta Anda bertindak sebagai AI Engineer yang merancang dan mengontrol sistem LLM. Anda akan menganalisis skenario untuk menentukan arsitektur yang tepat (VLM, MoE) dan, yang terpenting, memprediksi serta mengontrol output model menggunakan parameter *sampling* yang berbeda.

### **Langkah-Langkah Praktikum**

**1. Siapkan Dokumen Jawaban**
*   Buat sebuah file teks (misalnya menggunakan Notepad, Google Docs, atau editor Markdown) untuk menuliskan jawaban Anda.
*   Beri judul seperti "Unjuk Kerja Modul 3 - [Nama Anda]".

**2. Analisis Arsitektur & Pelatihan (Konsep 3.1: Decoder-Only)**
*   **Baca skenario:** Anda membandingkan BERT (Encoder-Only) dan GPT (Decoder-Only).
*   Tuliskan jawaban singkat (1-2 poin) yang menjelaskan mengapa arsitektur **Decoder-Only (GPT-style)** jauh lebih unggul untuk tugas *chatbot* dan *prompting* (In-Context Learning) dibandingkan Encoder-Only (BERT). (Petunjuk: Pikirkan tentang objektif pelatihan *autoregressive vs masked*).

**3. Analisis Kontrol Sampling (Konsep 3.2: Sampling)**
Perhatikan distribusi probabilitas (hasil Softmax) berikut untuk prompt "Restoran ini...":
`{"enak": 75%, "oke": 15%, "mahal": 8%, "tutup": 2%}`

*   **a.** Tuliskan kata apa yang akan dipilih jika menggunakan **Greedy Search**.
*   **b.** Jelaskan apa yang akan terjadi pada probabilitas "tutup" (2%) jika Anda menaikkan **Temperature menjadi 1.8**. Apakah kemungkinannya terpilih naik atau turun?
*   **c.** Tuliskan token mana saja yang masuk kandidat jika menggunakan **Top-K Sampling** dengan **K=2**.
*   **d.** Tuliskan token mana saja yang masuk kandidat jika menggunakan **Top-P (Nucleus) Sampling** dengan **P=0.9 (90%)**. Berikan perhitungan singkat Anda.

**4. Analisis Arsitektur Lanjutan (Konsep 3.3 & 3.4: VLM & MoE)**
Baca dua skenario berikut:
*   **(A)** Anda perlu membangun fitur di aplikasi Anda yang memungkinkan pengguna mengunggah foto dan bertanya, "Objek apa ini?"
*   **(B)** Perusahaan Anda ingin melatih model 1 Triliun parameter, tetapi biaya inferensi (komputasi) per *query* harus dijaga agar setara dengan model 150 Miliar parameter.

**Pertanyaan:**
*   **a.** Untuk skenario **(A)**, komponen arsitektur krusial apa (dari 3.3) yang harus ditambahkan ke LLM agar bisa "melihat"?
*   **b.** Untuk skenario **(B)**, arsitektur apa (dari 3.4) yang menggunakan "aktivasi jarang" (sparse activation) untuk mencapai ini?

**5. Simpan & Kumpulkan**
Simpan file jawaban Anda dalam format .md, .txt, atau .pdf.

---

### **Tools & Resource yang Dibutuhkan**
*   Akses ke materi Modul 3 (terutama Subtopik 3.1, 3.2, 3.3, dan 3.4).
*   Sebuah *text editor* atau *word processor* (Notepad, Google Docs, VS Code, dll.) untuk menulis jawaban.

---

### **Kriteria Keberhasilan / Penilaian Mandiri**
*   Mampu menjelaskan mengapa arsitektur Decoder-Only (*autoregressive*) cocok untuk ICL.
*   Mampu menentukan hasil Greedy Search, menjelaskan efek Temperature, mengidentifikasi kandidat Top-K, dan mengidentifikasi kandidat Top-P dengan benar.
*   Mampu memilih arsitektur VLM (Projector) dan MoE (Router) yang tepat untuk skenario yang diberikan.
*   Semua jawaban ditulis dengan jelas dan terstruktur dalam satu file dokumen.

---

### **Petunjuk Troubleshooting**
*   **Ragu tentang Decoder-Only vs Encoder-Only?** Ingat analogi "Si Penulis" (GPT) vs "Si Analis" (BERT). Chatbot adalah tugas "menulis" (3.1).
*   **Lupa cara kerja Top-P Sampling?** Baca kembali Subtopik 3.2. Jumlahkan probabilitas dari yang tertinggi (75%) hingga totalnya mencapai atau melebihi P (90%). Token apa saja yang Anda perlukan untuk mencapai 90%?
*   **Tertukar antara VLM and MoE?** Ingat VLM (3.3) adalah tentang menambah modalitas (penglihatan) melalui Projector. MoE (3.4) adalah tentang efisiensi komputasi (skala) melalui Router.

**Durasi Praktikum Disarankan:** 60 menit.

**Aksi Lanjutan atau Refleksi (opsional):**
Setelah selesai, coba pikirkan: Dari semua konsep di Modul 3 (Sampling/Temperature, VLM/Projectors, MoE/Sparse Activation), konsep mana yang menurut Anda paling krusial untuk dipahami seorang AI Engineer saat mengontrol atau menskalakan LLM di dunia nyata? Mengapa?

---
**HANDS-ON:** [Kontrol Perilaku dan Multimodal](https://github.com/Modul-LLM/Module-03-1.3-Fundamental)

---

### Penutup
### Modul 3 Fundamental LLM: Komponen Inti dan Arsitektur LLM Modern
---

Jika Modul 2 adalah tentang "anatomi" dasar LLM (komponen-komponen pembangunnya), Modul 3 adalah tentang "fisiologi" dan "arsitektur lanjutan". Kita telah beralih dari apa komponennya, menjadi bagaimana arsitektur modern berperilaku, dikontrol, dan berevolusi.

Dalam modul ini, kita telah berfokus pada arsitektur dan konsep yang mendefinisikan lanskap AI Generatif modern:

*   Kita membedah **arsitektur Decoder-Only**, memahami mengapa desain yang hanya "melihat ke masa lalu" atau Masked MHA ini menjadi dominan karena efisiensi pelatihan *self-supervised*-nya dan kemampuan *emergent* In-Context Learning (ICL).
*   Kita menganalisis **Tombol Pengatur Tingkat Kreativitas LLM: Proses Sampling**. Kita melacak alur dari Logits lalu menuju Softmax (probabilitas), dan mempelajari cara mengontrol output menggunakan parameter krusial seperti Temperature, Top-K, dan Top-P (Nucleus).
*   Kita melihat bagaimana LLM "belajar melihat" melalui **arsitektur Multimodal (VLM)**, memahami peran *Vision Encoder* (ViT) sebagai "mata" dan Projector sebagai "penerjemah" yang menjembatani domain gambar dan teks.
*   Kita membahas solusi untuk *Scaling Laws* yang mahal: **Mixture of Experts (MoE)**. Kita memahami konsep (*sparse activation*) di mana Router hanya mengaktifkan sebagian kecil "ahli" (FFN) untuk efisiensi komputasi.
*   Terakhir, kita menganalisis pesaing Transformer seperti **Mamba (Model State-Space)** yang menawarkan skalabilitas linear ($O(N)$) dan **Diffusion LLM** yang menawarkan pendekatan *non-autoregressive*.

Dengan menyelesaikan materi dalam modul ini, Anda telah berhasil mencapai tujuan pembelajaran berikut:
*   Anda kini dapat menjelaskan arsitektur Decoder-Only (*Autoregressive*) sebagai fondasi LLM generatif modern.
*   Anda mampu menguraikan proses sampling dari logits ke Softmax, dan menganalisis bagaimana parameter kontrol (seperti Temperature, Top-K, dan Top-P) memengaruhi perilaku output model.
*   Anda dapat mengidentifikasi dan membedakan evolusi serta pendekatan arsitektur lanjutan, termasuk Multimodal Projectors (untuk VLM), Mixture of Experts (MoE) (untuk efisiensi), Mamba (sebagai alternatif Transformer), dan Model Difusi.

> **Refleksi Akhir**
>
> Setelah mempelajari Sampling di Modul 3.2, Multimodal di Modul 3.3, dan MoE di Modul 3.4, bagaimana pandangan Anda tentang peran AI Engineer berubah?
>
> Apakah Anda menyadari bahwa menjadi AI Engineer modern bukan hanya tentang memilih model misalkan Llama 3 vs GPT-4o, tetapi lebih tentang bagaimana Anda mengontrol perilakunya (sampling), mengadaptasi modalitasnya (projector), dan mengoptimalkan biayanya (MoE) untuk kebutuhan bisnis di dunia nyata?


1. Kaplan, J., McCandlish, S., Henighan, T., Brown, T. B., Chess, B., Child, R., ... & Amodei, D. (2020). Scaling laws for neural language models. arXiv preprint arXiv:2001.08361. [https://arxiv.org/pdf/2001.08361/1000](https://arxiv.org/pdf/2001.08361/1000)
2. Jiang, A. Q., Sablayrolles, A., Roux, A., Mensch, A., Savary, B., Bamford, C., ... & Sayed, W. E. (2024). Mixtral of experts. arXiv preprint arXiv:2401.04088. [https://arxiv.org/pdf/2401.04088](https://arxiv.org/pdf/2401.04088)
3. Bengio, Y., Goodfellow, I., & Courville, A. (2017). Deep learning (Vol. 1, pp. 23-24). Cambridge, MA, USA: MIT press. [https://virtualmmx.ddns.net/gbooks/DeepLearning.pdf](https://virtualmmx.ddns.net/gbooks/DeepLearning.pdf)
4. Holtzman, A., Buys, J., Du, L., Forbes, M., & Choi, Y. (2019). The curious case of neural text degeneration. arXiv preprint arXiv:1904.09751. [https://arxiv.org/pdf/1904.09751](https://arxiv.org/pdf/1904.09751)