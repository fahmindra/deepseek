---
slug: modul-3-2
title: modul-3-2
---

### 3.2 Proses Sampling pada LLM (Softmax, Temperature, Top-K/P)
---

Di Modul 2 (Tabel 1), kita melihat bahwa tugas model bahasa adalah menghasilkan **distribusi probabilitas** atas seluruh *vocabulary* atau kumpulan kosakata. Di Modul 3.1, kita melihat arsitektur Decoder-Only yang dilatih untuk melakukan ini.

Blok Transformer terakhir akan menghasilkan sebuah **vektor representasi akhir** (ada di modul 2 (*Embedding*), sering disebut *final hidden state*, misal, berdimensi 768) untuk token berikutnya. Namun, misalkan kita memiliki 50.000 token atau lebih dalam *vocabulary* kita. Bagaimana kita mengubah satu vektor berdimensi 768 menjadi 50.000 probabilitas?

Dan yang lebih penting, setelah kita memiliki daftar probabilitas itu, misalkan: "cerah" 60%, "hujan" 30%, "berawan" 10%, bagaimana kita sebenarnya memilih satu kata? Apakah kita selalu memilih yang 60%?

![Halusinasi LLM](https://lms.sdmdigital.id/pluginfile.php/635375/mod_page/content/20/image60.jpg?time=1767940816825)
**Gambar 4. Halusinasi LLM**

Subtopik ini adalah tentang "mekanisme pilihan" tersebut. Ini adalah jembatan krusial antara perhitungan matematis model dan kata-kata yang Anda lihat misalkan di chatbot GenAI. Menguasai ini adalah kunci untuk mengontrol apakah model Anda terdengar faktual dan repetitif, atau kreatif dan terkadang "berhalusinasi" yaitu keadaan di mana model terlalu yakin pada output namun salah secara faktual atau tidak sesuai konteks.

---

#### 3.2.1 Dari Logits ke Probabilitas (Fungsi Softmax)

Langkah pertama adalah mengubah satu vektor output menjadi daftar skor untuk setiap kata di *vocabulary*. Ini dilakukan oleh Linear Layer terakhir di atas tumpukan Decoder pada arsitektur Transformer asli.

Lapisan ini pada dasarnya adalah kebalikan dari *Embedding Layer* (2.3.2); ia memproyeksikan vektor internal (misal, 768 dimensi) ke dimensi vocabulary (misal, 50.000 dimensi). Hasil dari lapisan ini bukanlah probabilitas. Hasilnya disebut **Logits**.

**Apa itu Logits?**
Logits adalah skor numerik mentah yang bisa bernilai positif atau negatif yang dihasilkan oleh model untuk setiap token di vocabulary. Logits yang lebih tinggi berarti model "lebih yakin" pada token tersebut.

**Contoh Logits:**
* "cerah": 9.8
* "hujan": 8.1
* "berawan": 7.0
* "salju": 2.5
* sisa 49.996 token lainnya: < 0

Kita tidak bisa menggunakan skor ini secara langsung. Kita perlu mengubahnya menjadi probabilitas yang rapi, di mana totalnya adalah 100%. Untuk ini, kita menggunakan fungsi matematis fundamental: **Softmax**.

> **Lalu Apa Itu Softmax?**
> Softmax adalah sebuah fungsi matematis yang mengambil sebuah vektor berisi Logits dan mengubahnya menjadi sebuah distribusi probabilitas.
> 
> Secara konseptual, Softmax melakukan dua hal:
> 1. Menggunakan eksponensial ($e^x$) untuk membuat semua skor mentah menjadi nilai positif dan membuat perbedaan skor menjadi lebih jelas.
> 2. Kemudian, nilai-nilai eksponensial tersebut **dinormalisasi** dengan membaginya dengan **jumlah kolektif** dari semua nilai. Langkah ini mengubah skor-skor tersebut menjadi **persentase** yang jika dijumlahkan, hasilnya tepat 1 atau 100%.

Rumus dari Softmax adalah:

$$\text{softmax}(\boldsymbol{x})_i = \frac{\exp(x_i)}{\sum_{j=1}^{n} \exp(x_j)}$$

Berikut adalah cara "menerjemahkan" formula tersebut ke dalam konteks LLM kita:
* **softmax(x)ᵢ**: Ini adalah Probabilitas Akhir dari token ke-i, misalnya, token "cerah".
* **x**: Ini adalah vektor Logits secara keseluruhan, yang memiliki n angka. Dalam contoh kita: [9.8, 8.1, 7.0, 2.5, ...].
* **n**: Ini adalah ukuran total vocabulary Anda (misal: 50.000).

Untuk mengkonkretkan formula matematis tersebut, mari kita terapkan pada contoh *logits* yang kita perkenalkan di 3.2.1:
* **Penyebut (Denominator):** Kita jumlahkan semua nilai eksponensial:
Total = exp(9.8) + exp(8.1) + exp(7.0) + exp(2.5) + ... ≈ 18020 + 3294 + 1097 + 12 ≈ 22423.
* **Probabilitas "cerah":** Kita ambil nilai "cerah" dan bagi dengan total:
P("cerah") = exp(9.8) / Total ≈ 18020 / 22423 ≈ 0.804 atau 80.4%

**Tabel 1. Alur Konversi Logits ke Probabilitas (Softmax)**

| Token | Logits (Skor Mentah) | logit (Eksponensial) $e^x$ | Probabilitas (Softmax) |
| :--- | :--- | :--- | :--- |
| "cerah" | 9.8 | ≈ 18020 | 18020/22423 ≈ 0.804 (80.4%) |
| "hujan" | 8.1 | ≈ 3294 | 3294/22423 ≈ 0.147 (14.7%) |
| "berawan" | 7.0 | ≈ 1097 | 1097/22423 ≈ 0.049 (4.9%) |
| "salju" | 2.5 | ≈ 12 | 12/22423 ≈ 0.0005 (0.05%) |
| **Total** | | **≈ 22423** | **1.0 (100%)** |

Total ≈ 22423 ini tidak memiliki satuan. Ia adalah nilai skalar mentah yang merepresentasikan penyebut (denominator) dalam rumus Softmax. Perhatikan bagaimana fungsi eksponensial dalam Softmax memperkuat keyakinan model: skor *logit* tertinggi ("cerah") mendominasi distribusi probabilitas akhir (81.3%), sementara skor *logit* yang rendah ("salju") secara efektif ditekan mendekati nol (0.05%).

---

#### 3.2.2 Strategi Sampling: Mengontrol Kreativitas Model

Kita memiliki distribusi probabilitas (80.4% "cerah", 14.7% "hujan", ...). Bagaimana kita memilih kata berikutnya?

Metode pertama adalah **Greedy Search (Pencarian Serakah)**.
Greedy Search adalah sebuah strategi sampling di mana kita selalu memilih satu token dengan probabilitas tertinggi (nilai logit $x_i$ tertinggi).
* Dalam contoh Tabel 1: Kita akan selalu memilih "cerah".
* **Kelebihan:** Sangat cepat, aman, faktual, dan dapat diprediksi.
* **Kekurangan:** Sangat membosankan dan repetitif. Jika model memilih kata "sangat", kemungkinan besar "sangat" akan kembali menjadi pilihan teratas di langkah berikutnya.

Metode kedua adalah **Sampling dengan Temperature**. Untuk memperkenalkan faktor "keacakan" (*randomness*) yang terkontrol, kita menerapkan **Temperature** sebagai pembagi pada Logits ($x_i$) sebelum dimasukkan ke fungsi Softmax.

Matematisnya, kita membuat logits baru $x'_i$:
$$x'_i = \frac{x_i}{T}$$

Kemudian, kita masukkan logits baru $x'$ ini ke fungsi Softmax standar:
$$P(w_i) = \frac{\exp(x'_i)}{\sum_{j=1}^{V} \exp(x'_j)}$$

**Analogi: Tombol Pengatur Kreativitas**
* **Temperature Rendah (misal, T = 0.2):**
    * Kita membagi logit dengan angka kecil. Hasilnya: Perbedaan antara logits menjadi jauh lebih besar.
    * Distribusi probabilitas menjadi "lebih runcing".
    * Perilaku: Sangat deterministik, faktual, mirip *Greedy Search*.
* **Temperature Normal (T = 1.0):**
    * Hasil Softmax sesuai Tabel 1 (80.4%).
* **Temperature Tinggi (misal, T = 2.0):**
    * Kita membagi logit dengan angka besar. Hasilnya: Perbedaan antara logits menjadi jauh lebih kecil.
    * Distribusi probabilitas menjadi "lebih datar".
    * Perilaku: Sangat kreatif, acak, dan "gila". Model lebih mungkin memilih kata berprobabilitas rendah seperti "salju".

**Tabel 2. Perbandingan Efek Temperature pada Sampling**

| Pengaturan (Temperatur) | Proses Logits | Hasil Distribusi | Perilaku Model |
| :--- | :--- | :--- | :--- |
| T Rendah (0.2) | $logit\_baru = 9.8 / 0.2 = 49$ (Perbedaan diperbesar) | Sangat Runcing (Prob "cerah" ~ 99.9%) | Deterministik, Faktual, Mirip Greedy Search. |
| T Normal (1.0) | $logit\_baru = 9.8 / 1.0 = 9.8$ (Tidak ada perubahan) | Normal (Prob "cerah" = 80.4%) | Seimbang. |
| T Tinggi (2.0) | $logit\_baru = 9.8 / 2.0 = 4.9$ (Perbedaan diratakan) | Sangat Datar (Prob "cerah" ~ 40%) | Kreatif, Acak. Token prob. rendah (misal "salju") ikut terangkat. |

![Diagram Temperature](https://lms.sdmdigital.id/pluginfile.php/635375/mod_page/content/20/image31.jpg)
**Gambar 5. Diagram Temperature**

Gambar 5 menunjukkan Temperature Rendah (kiri) "mempertajam" distribusi, sementara Temperature Tinggi (kanan) "meratakan" distribusi, meningkatkan kemungkinan token berprobabilitas rendah untuk terpilih.

> **Catatan Penting:**
> *Temperature* (T) diterapkan pada logits, bukan probabilitas akhir. Jika *Temperature* terlalu tinggi, kata-kata yang benar-benar aneh (seperti "logaritma" dalam konteks cuaca) mendapatkan probabilitas yang cukup tinggi, menyebabkan output tidak koheren.

---

#### 3.2.3 Metode Sampling Cerdas (Top-K dan Top-P)

Untuk mendapatkan "jalan tengah" antara kreatif dan koheren, kita **memfilter** daftar probabilitas terlebih dahulu untuk menciptakan "daftar kandidat" yang lebih aman.

**Apa itu Top-K Sampling?**
**Top-K Sampling** memfilter vokabulari dan hanya menyimpan **K** token dengan probabilitas tertinggi.
* **K** adalah variabel integer jumlah kandidat teratas.
* **Contoh K = 3**: Kita hanya ambil {"cerah", "hujan", "berawan"}. Kita buang "salju" dan token lainnya, lalu sampling dari 3 token itu.

**Apa itu Top-P Sampling?**
**Top-P atau Nucleus Sampling** adalah strategi dinamis. Kita memilih jumlah token minimal yang total probabilitas kumulatifnya (setelah diurutkan) mencapai **P**.
* **P** adalah variabel ambang (*threshold*) probabilitas kumulatif.

**Mengapa disebut "Nucleus"?** Karena metode ini menemukan "inti" (*nucleus*) distribusi dan membuang "ekor" (*tail*) yang berisi kata-kata tidak relevan.

Contoh P = 0.9 (90%):
1. Urutkan: "cerah" (80.4%) -> Kumulatif = 80.4%. (Belum mencapai 90%)
2. Tambahkan: "hujan" (14.7%) -> Kumulatif = 95.1%. (Sudah mencapai 90%)
3. Berhenti. Nucleus kita adalah {"cerah", "hujan"}.

Top-P dianggap sebagai metode *default* terbaik karena ia adaptif terhadap tingkat keyakinan model.

---

#### Mini Quiz
*(Silakan akses Mini Quiz interaktif melalui platform LMS)*

---

> **Refleksi Singkat**
> Kita baru saja membahas "Tombol Pengatur Tingkat Kreativitas": **Temperature, Top-K, dan Top-P**. Coba refleksikan: Jika Anda membangun sebuah chatbot untuk layanan pelanggan bank (yang harus **faktual, aman,** dan **tidak mengada-ada**), parameter sampling apa yang akan Anda prioritaskan? 
> Apakah Anda akan menggunakan Temperature rendah atau tinggi? Greedy Search atau Top-P yang lebar? Mengapa demikian?