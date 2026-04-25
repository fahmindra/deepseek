---
slug: modul-3-4
title: modul-3-4
---
### 3.4 Mixture of Experts (MoE)
---

Kita tahu dari *Scaling Laws* bahwa menambah parameter (membuat model lebih besar) adalah resep yang terbukti untuk membuat model lebih "pintar". Namun, kita menghadapi halangan dalam proses komputasi. Model *dense* (padat) seperti GPT-3 yang memiliki 175B parameter sangat mahal untuk dilatih dan bahkan untuk dijalankan atau proses inferensi, karena 175 miliar parameter tersebut diaktifkan untuk setiap token yang diproses.

Ini memicu pertanyaan: Apakah mungkin kita memiliki model dengan total parameter yang sangat besar misal, 1 Triliun parameter, tetapi untuk satu *query* atau *prompt* yang spesifik, kita hanya menggunakan sebagian kecil misal, cukup dengan 100 Miliar parameter yang relevan?

![Mixture of Experts](https://lms.sdmdigital.id/pluginfile.php/635377/mod_page/content/11/image48.jpg)
**Gambar 8. Mixture of Experts**

Ini adalah ide inti di balik **Mixture of Experts (MoE)**. Alih-alih satu otak raksasa yang menangani segalanya, MoE adalah "kabinet" berisi banyak otak "ahli" yang lebih kecil, dan kita hanya mengaktifkan ahli yang kita butuhkan.

#### 3.4.1 Arsitektur MoE: "Kabinet Ahli"

Arsitektur MoE bukanlah arsitektur yang sama sekali baru. Ini adalah modifikasi cerdas dari blok Transformer yang sudah kita kenal. Ingat, Blok Decoder-Only memiliki dua komponen: (1) Masked Self-Attention dan (2) Feed-Forward Network (FFN).

Dalam model MoE, kita mengganti lapisan FFN dengan sesuatu yang jauh lebih kompleks:

> **Apa itu MoE?**
> Mixture of Experts (MoE) adalah sebuah blok arsitektur yang menggantikan satu lapisan FFN dengan dua komponen utama:
> 1. **Sekumpulan *"Experts"* atau Pakar:** Ini adalah beberapa lapisan FFN yang independen.
> 2. **Router:** Sebuah *neural network* kecil yang bertugas untuk mengatur sebagaimana layaknya seorang "manajer" atau "operator telepon".

**Ilustrasi Alur Kerja MoE**

![Alur Kerja Lapisan Mixture of Experts (MoE)](https://lms.sdmdigital.id/pluginfile.php/635377/mod_page/content/11/image57.jpg?time=1766639790098)
**Gambar 9. Alur Kerja Lapisan Mixture of Experts (MoE)**

Alur Kerjanya adalah sebagai berikut:
1. **Input:** Sebuah token (setelah diproses *Self-Attention*) masuk ke lapisan MoE.
2. **Routing:** Token tersebut pertama kali masuk ke Router (Gate). Tugas Router adalah melihat token itu dan memutuskan: "Untuk token ini, siapa 1 atau 2 'ahli' (dari 4 ahli yang ada) yang paling relevan untuk menanganinya?"
3. **Sparse Activation:** Keadaan di mana Router hanya mengirimkan token tersebut ke ahli yang dipilih (misal, *Expert* 1 dan *Expert* 3). Dua ahli lainnya (*Expert* 2 dan 4) tidak diaktifkan. Mereka "diam" dan tidak menggunakan komputasi sama sekali untuk token ini.
4. **Processing:** *Expert* 1 dan *Expert* 3 memproses token tersebut secara independen (paralel).
5. **Output:** Hasil dari kedua ahli tersebut digabungkan untuk menghasilkan vektor output akhir dari lapisan MoE.

**Analogi MoE: Tim Spesialis di Rumah Sakit**
* **Model Dense (Non-MoE):** Ketika seorang pasien (token) datang dengan keluhan sakit kaki, semua dokter di rumah sakit (Ahli Bedah Saraf, Ahli Jantung, Ahli Mata, Ahli Ortopedi) harus berkumpul dan memeriksa pasien tersebut. Ini sangat boros waktu.
* **Model MoE:** Pasien (token) datang ke Router (Manajer Triase). Manajer melihat keluhannya ("sakit kaki") dan berkata, "Anda hanya perlu ke Ahli Ortopedi (Expert 3)." Hanya satu dokter yang diaktifkan; sisanya bisa lanjut bekerja.

Arsitektur MoE (Gambar 9) yang mengandalkan *Router* dan *sparse activation* ini bukanlah sekadar konsep teoretis. Ini adalah inovasi yang secara fundamental mengubah paradigma *trade-off* dalam *Scaling Laws*. Dengan mengaktifkan sebagian kecil saja kemampuan dari total parameter yang sangat besar, akan membuat komputasinya jauh lebih hemat. Namun cara kerja ini juga membawa tantangan arsitektural baru yang akan kita bahas.

#### 3.4.2 Keuntungan dan Tantangan Sparse Activation

Model MoE seperti Mixtral dari Mistral AI, telah menjadi sangat populer karena keunggulan komputasinya. Namun apa sebenarnya letak keunggulannya? Keunggulannya ada di *Sparse Activation*.

Ini adalah keuntungan utama dari *Sparse Activation*:
1. **Parameter Skala Besar:** Kita bisa memiliki model dengan jumlah total parameter yang masif. Misalnya, model dengan 8 keahlian, di mana masing-masingnya memiliki 20 Miliar parameter, atau total memiliki 160 Miliar parameter.
2. **Inferensi Cepat serta Komputasi Ringan:** Meskipun totalnya 160 Miliar, jika Router hanya memilih 2 keahlian, maka untuk setiap token, kita hanya menjalankan komputasi setara 40 Miliar parameter. Modelnya terasa "sekecil" 40 Miliar, tetapi memiliki "pengetahuan" dari 160 Miliar.

Ini memungkinkan kita melatih model dengan triliunan parameter (pengetahuan) sambil menjaga biaya inferensi tetap terkendali. Namun di balik itu semua, terdapat tantangan yang cukup menyulitkan.

**Apa Saja Tantangan Arsitektural MoE?**
1. **Beban Memori (VRAM):** Meskipun kita hanya menggunakan 2 keahlian, semua atau 8 keahlian harus dimuat ke dalam memori (VRAM) GPU setiap saat. Meskipun saat ini ada penelitian aktif untuk mengefisienkan hal ini, pemahaman dasarnya tetap menjadi acuan yaitu MoE tetap membutuhkan GPU dengan VRAM yang sangat besar, meskipun komputasinya (FLOPs) lebih sedikit.
2. **Pelatihan *Load Balancing*:** Tantangan terbesarnya adalah melatih *Router*. Jika tidak dilatih dengan benar, *Router* bisa menjadi "malas". Ia mungkin memutuskan bahwa Expert 1 adalah yang terbaik, dan akhirnya mengirim semua token ke Expert 1. Ini membuat 7 *expert* lainnya "mati" atau tidak pernah belajar apa-apa. Oleh karena itu, pelatihan MoE membutuhkan *loss function* khusus yang disebut *load balancing loss* untuk "memaksa" *Router* mendistribusikan pekerjaan secara merata.

#### Mini Quiz
*(Silakan akses Mini Quiz interaktif melalui platform LMS)*

---

> **Refleksi Singkat**
>
> Kita telah mempelajari bahwa model MoE seperti Mixtral menggunakan "sparse activation" untuk berjalan lebih cepat, misal hanya mengaktifkan 2 dari 8 "ahli". Namun, kita juga belajar bahwa kedelapan ahli tersebut harus tetap dimuat ke dalam memori VRAM GPU.
>
> Sebagai seorang AI Engineer yang bertugas men-deploy model, bagaimana trade-off spesifik ini (**komputasi inferensi cepat** vs. **kebutuhan VRAM sangat tinggi**) memengaruhi keputusan Anda dalam memilih antara model MoE (misal, Mixtral 8x7B) dan model dense (padat) yang lebih kecil (misal, Llama 7B)?