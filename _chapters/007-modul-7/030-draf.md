---
slug: modul-7-3
title: modul-7-3
---
# 3.3 Teknik Adaptasi Ringan Lainnya: Prompting & Adapters

Kita sudah membahas LoRA (3.2), yang bekerja dengan cara menyuntikkan matriks perubahan ke dalam bobot model $\Delta W$. Itu sangat efisien.

Tapi para peneliti bertanya: "Apakah kita benar-benar perlu menyentuh arsitektur model? Bisakah kita mengubah perilaku model hanya dengan **mengubah apa yang kita masukkan ke dalamnya**, tapi membuat input tersebut *bisa dilatih*?"

Ini melahirkan keluarga teknik **Prompt Tuning** dan **Prefix Tuning**. Alih-alih melakukan "bedah otak" (mengubah bobot), teknik ini lebih seperti memberikan "bisikan hipnotis" (vektor khusus) ke telinga model agar ia berperilaku sesuai keinginan kita.

Selain itu, ada juga metode klasik **Adapters** (Houlsby) yang secara fisik menambahkan lapisan baru. Di subtopik ini, kita akan membedakan teknik-teknik "Non-LoRA" ini.

---

## 3.3.1 Soft Prompting (Prompt Tuning)

Kita tahu bahwa *prompt engineering* (menulis instruksi teks: "Terjemahkan ini...") bisa mengubah perilaku model. **Soft Prompting** membawa ide ini ke level *neural*.

**Apa itu Soft Prompting?**

**Soft Prompting adalah** teknik di mana kita menambahkan serangkaian **vektor yang dapat dilatih (trainable vectors)** ke bagian depan *embedding* input.

*   **Hard Prompt (Manusia):** Teks nyata seperti "Translate this:". Sifatnya kaku (diskrit).
*   **Soft Prompt (AI):** Deretan angka (vektor) yang tidak merepresentasikan kata nyata, tapi dipelajari oleh AI melalui *backpropagation* untuk memicu perilaku tertentu.

**Analogi: Kertas Contekan Gaib**

Bayangkan Anda mau ujian.
*   **Hard Prompt:** Teman Anda membisikkan "Ingat rumus Pythagoras!".
*   **Soft Prompt:** Anda menanamkan sebuah *chip* memori abstrak di depan otak Anda yang berisi "intuisi matematika". Anda tidak bisa membacanya sebagai teks, tapi kehadirannya membuat Anda tiba-tiba jago matematika.

**Kelebihan:** Sangat hemat memori (hanya melatih <0.01% parameter).
**Kekurangan:** Seringkali kurang kuat untuk tugas yang kompleks dibandingkan LoRA.

Metode *Soft Prompting* ini sangat efisien, namun ia memiliki kelemahan struktural: instruksi hanya diberikan di "pintu masuk" (layer input). Seiring data mengalir naik melewati puluhan layer Transformer, pengaruh "bisikan" awal ini bisa memudar atau terdistorsi. Bagaimana jika kita bisa menyuntikkan instruksi secara konsisten di *setiap* lantai pemrosesan agar model tetap fokus?

---

## 3.3.2 Prefix Tuning

Jika *Soft Prompting* hanya menambahkan vektor di lapisan input (paling bawah), **Prefix Tuning** melakukan pendekatan yang jauh lebih agresif dan mendalam.

**Apa itu Prefix Tuning?**

**Prefix Tuning adalah** teknik adaptasi di mana kita menyisipkan serangkaian vektor yang dapat dilatih (disebut "Prefix") di setiap lapisan Transformer, secara spesifik ditempelkan di depan sekumpulan vektor **Key (K)** dan **Value (V)** dalam mekanisme *Multi-Head Attention*.

**Mengapa Ini "Lebih Kaya" dari Soft Prompting?**
Dalam arsitektur Transformer, setiap lapisan merepresentasikan tingkat abstraksi yang berbeda (Lapisan bawah = sintaksis, Lapisan atas = semantik/logika).

*   **Soft Prompting:** Hanya mempengaruhi lapisan bawah. Lapisan atas harus "mengingat" instruksi itu.
*   **Prefix Tuning:** Kita menyuntikkan "instruksi virtual" di setiap tingkat abstraksi. Kita memberi tahu lapisan sintaksis apa yang harus dilakukan, DAN kita juga memberi tahu lapisan semantik apa yang harus dilakukan secara langsung.

**Mekanisme: Virtual Tokens**
Secara teknis, *Prefix Tuning* membuat model merasa seolah-olah ada serangkaian "token virtual" yang selalu hadir di konteks masa lalu, membimbing model ke arah yang benar.

*   Tanpa Prefix: `[Konteks Asli] -> Output`
*   Dengan Prefix: `[Prefix Layer 1]...[Prefix Layer 12] + [Konteks Asli] -> Output`

**Analogi: Pemandu Jalan di Setiap Persimpangan**

*   **Soft Prompting:** Memberi peta di awal perjalanan, lalu membiarkan turis berjalan sendiri. Jika hutan lebat (model dalam), turis bisa tersesat.
*   **Prefix Tuning:** Menempatkan rambu penunjuk jalan di **setiap persimpangan** sepanjang perjalanan. Turis (data) terus-menerus diingatkan dan dibimbing di setiap langkah pemrosesan.

**Trade-off:** Prefix Tuning jauh lebih ekspresif dan kuat daripada Soft Prompting, namun juga sedikit lebih boros parameter (sekitar 0.1% - 3%) karena kita menyimpan vektor untuk setiap layer, bukan hanya layer input.

*Prefix Tuning* menawarkan kontrol yang jauh lebih kuat daripada *Soft Prompting*, namun ia tetap bekerja dalam paradigma "menambah konteks virtual". Ada pendekatan ketiga yang lebih arsitektural: alih-alih memanipulasi *state* internal via prefix, kita menyisipkan **modul pemrosesan (neural network) nyata** ke dalam arsitektur.

---

## 3.3.3 MLP Adapters (Houlsby Adapters)

Ini adalah teknik PEFT "OG" (*Original Gangster*) sebelum LoRA populer, yang diperkenalkan oleh Houlsby et al. (2019).

**Definisi: MLP Adapters**

Teknik ini menyisipkan lapisan Neural Network kecil (Bottleneck MLP) secara fisik di antara lapisan-lapisan Transformer yang asli.

*   Transformer Asli: `Layer 1 -> Layer 2`
*   Dengan Adapter: `Layer 1 -> Adapter -> Layer 2`

**Arsitektur Bottleneck:**

Adapter ini biasanya memiliki struktur "menyempit lalu melebar" (mirip autoencoder):
1.  Proyeksi ke dimensi kecil (*down-projection*).
2.  Aktivasi non-linear (ReLU/GELU).
3.  Proyeksi kembali ke dimensi besar (up-projection).
Struktur ini meminimalkan jumlah parameter tambahan.

**Perbedaan Kunci dengan LoRA:**

*   **LoRA:** Paralel (jalur samping). Bisa digabung (*merge*) saat inferensi sehingga **tidak ada latensi tambahan**.
*   **Adapters:** Serial (disisipkan di tengah). Data *harus* melewati adapter ini. Ini menambah langkah komputasi, sehingga secara fisik **memperlambat waktu inferensi**.

Tabel 1 ini meringkas perbedaan antara LoRA, Soft Prompt, Prefix Tuning, dan Adapters.

**Tabel 1. Perbandingan Teknik PEFT**

| Teknik | Apa yang Dilatih? | Letak Perubahan | Dampak Inferensi |
| :--- | :--- | :--- | :--- |
| **LoRA** | Matriks Rank-Rendah ($\Delta W$) | Paralel dengan bobot | **Nol** (bisa di-merge) |
| **Soft Prompt** | Vektor Input Tambahan | Hanya di Input Layer | Minimal |
| **Prefix Tuning** | Vektor Prefix Tambahan | Di setiap Layer Attention | Minimal |
| **Adapters** | Lapisan MLP Baru | Disisipkan *di antara* layer | **Ada Latensi** (Lebih lambat) |

Meskipun efektif secara akurasi, kelemahan fundamental berupa latensi inferensi (inference latency) inilah yang membuat metode Adapter tipe Houlsby perlahan ditinggalkan dan digantikan oleh pendekatan LoRA dalam implementasi modern, di mana kecepatan adalah segalanya.

---

### Mini Quiz

<iframe src="https://app.lumi.education/api/v1/run/kyn7mL/embed" width="1088" height="720" frameborder="0" allowfullscreen="allowfullscreen" allow="geolocation *; microphone *; camera *; midi *; encrypted-media *" style="width: 100%; height: 441px;"></iframe>

---

> **Refleksi Singkat**
>
> Teknik **Soft Prompting** memungkinkan kita menyimpan satu model beku (misal: GPT-3) dan ribuan "prompt vector" kecil (masing-masing hanya beberapa Kilobyte) untuk tugas berbeda. Jika Anda membangun platform SaaS di mana user bisa membuat "bot kepribadian" mereka sendiri, mengapa Soft Prompting mungkin menjadi arsitektur yang lebih menarik secara ekonomi daripada Fine-Tuning (bahkan LoRA) untuk setiap user?