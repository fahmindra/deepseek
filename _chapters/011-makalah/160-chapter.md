---
slug: ch-11-16
title: "NSA"
layout: chapter
---
## Bab: Atensi Jarang Alami (Native Sparse Attention/NSA) — Atensi Jarang yang Selaras dengan Perangkat Keras dan Dapat Dilatih secara Alami

Pemodelan konteks panjang (*long-context modeling*) merupakan aspek krusial bagi model bahasa generasi berikutnya. Namun, biaya komputasi yang tinggi dari mekanisme atensi standar menghadirkan tantangan besar. **Atensi Jarang** (*Sparse Attention*) menawarkan arah yang menjanjikan untuk meningkatkan efisiensi tanpa mengorbankan kemampuan model. 

Kami memperkenalkan **NSA (Native Sparse Attention)**, sebuah mekanisme atensi jarang yang dapat dilatih secara alami (*natively trainable*) yang mengintegrasikan inovasi algoritma dengan optimasi yang selaras dengan perangkat keras (*hardware-aligned*). NSA menggunakan strategi jarang hierarkis dinamis, menggabungkan kompresi token berbutir kasar (*coarse-grained*) dengan seleksi token berbutir halus (*fine-grained*) untuk mempertahankan kesadaran konteks global sekaligus presisi lokal. 

Pendekatan kami memajukan desain atensi jarang melalui dua inovasi utama:
1.  **Peningkatan Kecepatan Substansial**: Melalui desain algoritma yang menyeimbangkan intensitas aritmatika, dengan optimasi implementasi untuk perangkat keras modern.
2.  **Pelatihan Ujung-ke-Ujung (*End-to-End*)**: Memungkinkan pelatihan penuh, mengurangi komputasi prapelatihan tanpa mengorbankan performa model. 

Eksperimen menunjukkan bahwa model yang dilatih dengan NSA mempertahankan atau melampaui model **Atensi Penuh** (*Full Attention*) pada berbagai tolok ukur umum, tugas konteks panjang, dan penalaran berbasis instruksi. NSA mencapai percepatan komputasi yang signifikan pada urutan sepanjang 64k di seluruh tahap pengodean (*decoding*), perambatan maju (*forward propagation*), dan perambatan mundur (*backward propagation*).

---

## 1. Pendahuluan

Komunitas riset semakin mengakui pemodelan konteks panjang sebagai kemampuan krusial, didorong oleh aplikasi dunia nyata mulai dari penalaran mendalam hingga pembuatan kode tingkat repositori dan sistem agen otonom multi-giliran. Terobosan terbaru seperti model seri-o dari OpenAI, DeepSeek-R1, dan Gemini 1.5 Pro memungkinkan model untuk memproses seluruh basis kode dan dokumen panjang. Namun, kompleksitas tinggi dari mekanisme atensi vanilla (Transformer tradisional) menjadi hambatan laten seiring bertambahnya panjang urutan. Estimasi teoritis menunjukkan bahwa komputasi atensi menyumbang 70–80% dari total latensi saat melakukan pengodean konteks sepanjang 64k.

Sifat jarang (*sparsity*) inheren dalam atensi *softmax* memungkinkan kita untuk hanya menghitung pasangan kueri-kunci yang kritis. Meskipun banyak strategi telah diusulkan (seperti pengosongan *KV-cache* atau seleksi berbasis *hashing*), mereka sering kali gagal dalam penyebaran praktis. Banyak metode tidak mencapai percepatan nyata yang sebanding dengan keuntungan teoritisnya dan kurang memiliki dukungan waktu pelatihan (*training-time support*).

Untuk mengatasi batasan ini, NSA dirancang untuk menangani dua tantangan utama:
*   **Percepatan Inferensi Selaras Perangkat Keras**: Mengonversi pengurangan komputasi teoritis menjadi peningkatan kecepatan aktual melalui desain algoritma yang ramah perangkat keras (memitigasi hambatan akses memori).
*   **Desain Algoritma Sadar Pelatihan (*Training-aware*)**: Memungkinkan komputasi ujung-ke-ujung dengan operator yang dapat dilatih untuk mengurangi biaya pelatihan sambil menjaga performa.

---

## **Analisis Visual: Performa dan Efisiensi**

**Deskripsi Gambar 1:**
Gambar ini menyajikan dua grafik batang yang membandingkan NSA (warna jingga) dengan Atensi Penuh/Full Attention (warna abu-abu).

*   **Grafik Kiri (Performance on Benchmarks):**
    *   **Sumbu-X:** Menampilkan kategori tolok ukur: *General* (Umum), *LongBench* (Konteks Panjang), dan *Reasoning* (Penalaran).
    *   **Sumbu-Y:** Menampilkan *Score* (Skor) dari 0.0 hingga 0.5.
    *   **Tren:** Terlihat bahwa batang jingga (NSA) selalu sedikit lebih tinggi atau setara dengan batang abu-abu (Full Attention). Secara akademik, ini membuktikan bahwa meskipun NSA bersifat "jarang", ia tidak kehilangan kemampuan intelektualnya; bahkan sedikit melampaui model atensi penuh pada rata-rata tolok ukur umum dan penalaran.
*   **Grafik Kanan (Speed on Stages):**
    *   **Sumbu-X:** Tahapan komputasi: *Decode* (Pengodean), *Forward* (Maju), dan *Backward* (Mundur).
    *   **Sumbu-Y:** *Speedup Ratio* (Rasio Percepatan) dari 1.0 hingga 13.0.
    *   **Data Poin:** Pada tahap *Decode*, NSA mencapai percepatan **11.6x**. Pada tahap *Forward*, mencapai **9.0x**. Pada tahap *Backward*, mencapai **6.0x**.
    *   **Kesimpulan Akademis:** NSA memberikan efisiensi yang masif di seluruh siklus hidup model, terutama pada tahap inferensi (*decoding*) yang biasanya menjadi hambatan utama pada urutan panjang (64k).

---

## 2. Memikirkan Ulang Metode Atensi Jarang

Metode atensi jarang modern sering kali mengalami "Ilusi Inferensi Efisien". Meskipun operasi matematikanya berkurang, latensi inferensi tetap tinggi karena:
1.  **Sparsitas Terbatas Fase**: Beberapa metode hanya mempercepat pengodean (*decoding*) tapi lambat di fase prapengisian (*prefilling*), atau sebaliknya.
2.  **Ketidakcocokan dengan Arsitektur Lanjut**: Banyak metode gagal beradaptasi dengan *Grouped-Query Attention* (GQA) yang berbagi KV di antara beberapa kepala kueri. Pola akses memori yang terpencar merusak desain efisiensi memori dari arsitektur modern.

Dari sisi pelatihan, muncul "Mitos Sparsitas yang Dapat Dilatih":
1.  **Degradasi Performa**: Menerapkan sparsitas hanya setelah pelatihan (*post-hoc*) memaksa model menyimpang dari trajektori optimasi aslinya.
2.  **Komponen Tidak Dapat Dilatih**: Operasi diskret seperti pengelompokan (*clustering*) menciptakan diskontinuitas dalam graf komputasi, mencegah aliran gradien.
3.  **Perambatan Mundur yang Tidak Efisien**: Strategi seleksi per-token menyebabkan akses memori non-kontigu, sehingga tidak bisa memanfaatkan teknik cepat seperti *FlashAttention*.

---

## 3. Metodologi

### 3.1. Latar Belakang

Mekanisme atensi dalam pemodelan bahasa menghitung skor relevansi antara token kueri $q_t$ terhadap semua kunci sebelumnya $k_{:t}$ untuk menghasilkan jumlah tertimbang dari nilai $v_{:t}$. Secara formal, untuk urutan dengan panjang $t$:

$$ o_t = \text{Attn}(q_t, k_{:t}, v_{:t}) = \sum_{i=1}^{t} \alpha_{t,i} v_i \quad (1) $$
$$ \alpha_{t,i} = \frac{e^{\frac{q_t^\top k_i}{\sqrt{d_k}}}}{\sum_{j=1}^{t} e^{\frac{q_t^\top k_j}{\sqrt{d_k}}}} \quad (2) $$

**Penjelasan Rumus:**
*   $o_t$: Output atensi pada langkah waktu $t$.
*   $\alpha_{t,i}$: Bobot atensi (perhatian) antara kueri pada posisi $t$ dan kunci pada posisi $i$.
*   $d_k$: Dimensi fitur dari kunci, digunakan sebagai faktor penskalaan $\sqrt{d_k}$ untuk menjaga stabilitas gradien saat menghitung produk titik (*dot product*).
*   **Hubungan Logis**: Seiring bertambahnya panjang urutan $t$, biaya perhitungan penyebut (sumasi) tumbuh secara linear untuk setiap token baru, yang secara total menyebabkan kompleksitas kuadratik $O(t^2)$.

### 3.2. Kerangka Kerja Keseluruhan NSA

NSA mengusulkan untuk mengganti pasangan kunci-nilai asli yang besar dengan set representasi yang lebih padat informasi dan kompak $\tilde{K}_t, \tilde{V}_t$. Output atensi yang dioptimalkan didefinisikan sebagai:

$$ o^*_t = \sum_{c \in C} g_t^c \cdot \text{Attn}(q_t, \tilde{K}_t^c, \tilde{V}_t^c) \quad (5) $$

**Penjelasan Variabel Rumus (5):**
*   $o^*_t$: Output atensi akhir dari NSA.
*   $C$: Himpunan strategi pemetaan, yang terdiri dari $\{cmp, slc, win\}$ (Kompresi, Seleksi, dan Jendela Geser).
*   $g_t^c$: Skor gerbang (*gate score*) bernilai $[0, 1]$ yang menentukan kontribusi relatif dari setiap cabang strategi. Skor ini dihasilkan oleh MLP dengan aktivasi *sigmoid*.
*   $\tilde{K}_t^c, \tilde{V}_t^c$: Pasangan kunci-nilai yang telah dipetakan ulang sesuai strategi $c$.

---

## **Analisis Visual: Arsitektur NSA**

**Deskripsi Gambar 2:**
Gambar ini menjelaskan struktur hierarkis NSA.

*   **Bagian Kiri (Overview):**
    *   Input kunci ($k_{:t}$) dan nilai ($v_{:t}$) dibagi menjadi blok-blok kontinu.
    *   Terdapat tiga cabang paralel:
        1.  **Compression (Jejaring Hijau):** Mengompresi blok menjadi token tunggal untuk menangkap pola kasar.
        2.  **Selection (Jejaring Biru):** Memilih blok-blok paling penting (Top-n) berdasarkan skor kepentingan.
        3.  **Sliding (Jejaring Hijau Muda):** Menangani token tetangga terdekat secara langsung.
*   **Bagian Kanan (Visualization of Masks):**
    *   Kotak-kotak hijau menunjukkan area skor atensi yang dihitung, kotak putih adalah area yang diabaikan.
    *   **Compressed Attention Mask:** Menunjukkan pola pindaian global yang kasar.
    *   **Selected Attention Mask:** Menunjukkan pola titik-titik (blok) yang tersebar namun penting.
    *   **Sliding Attention Mask:** Menunjukkan pola diagonal di sekitar kueri (konteks lokal).
*   **Kesimpulan Akademis:** Dengan membagi beban kerja ke tiga cabang ini, model tetap bisa "melihat" masa lalu yang sangat jauh melalui kompresi, "mengingat" detail spesifik melalui seleksi, dan "memahami" konteks saat ini melalui jendela geser.

---

### 3.3. Desain Algoritma

#### 3.3.1. Kompresi Token (*Token Compression*)
Kami mengagregasi blok sekuensial kunci/nilai menjadi representasi tingkat blok. Secara formal:

$$ \tilde{K}_t^{cmp} = f_K^{cmp}(k_{:t}) = \{ \phi(k_{id+1:id+l}) \mid 0 \le i \le \lfloor \frac{t-l}{d} \rfloor \} \quad (7) $$

**Penjelasan Variabel Rumus (7):**
*   $l$: Panjang blok (misalnya 32).
*   $d$: Langkah geser (*stride*) antar blok tetangga.
*   $\phi$: MLP yang dapat dilatih dengan pengodean posisi intra-blok untuk memetakan blok kunci menjadi satu kunci kompresi tunggal.
*   **Logika**: Strategi ini menangkap informasi semantik tingkat tinggi dengan resolusi yang lebih rendah, sehingga mengurangi beban komputasi secara signifikan.

#### 3.3.2. Seleksi Token (*Token Selection*)
Menggunakan hanya kunci terkompresi dapat menghilangkan detail halus. NSA menggunakan **Seleksi Blok** (*Blockwise Selection*) karena GPU modern memiliki *throughput* lebih tinggi untuk akses blok kontinu dibandingkan pembacaan indeks acak.

Skor kepentingan blok dihitung dengan memanfaatkan hasil antara dari atensi kompresi:

$$ p_t^{cmp} = \text{Softmax}(q_t^\top \tilde{K}_t^{cmp}) \quad (8) $$

Jika skema pemblokan kompresi dan seleksi berbeda, skor dipetakan secara spasial:

$$ p_t^{slc}[j] = \sum_{m=0}^{\frac{l'}{l}-1} \sum_{n=0}^{\frac{l'}{d}-1} p_t^{cmp} \left[ \frac{l'}{d}j - m - n \right] \quad (9) $$

**Penjelasan Rumus (9):**
Rumus ini menjumlahkan skor kepentingan dari token kompresi yang tumpang tindih secara spasial dengan blok seleksi berukuran $l'$. Ini memastikan bahwa jika suatu area dianggap penting pada tingkat kasar, area tersebut akan dipilih untuk pemrosesan tingkat halus.

Untuk model GQA, skor ini dijumlahkan di seluruh kepala dalam satu grup:
$$ p_t^{slc'} = \sum_{h=1}^{H} p_t^{slc,(h)} \quad (10) $$
Hal ini menjamin pemilihan blok yang konsisten, meminimalkan pemuatan ulang KV-cache.

#### 3.3.3. Jendela Geser (*Sliding Window*)
Cabang ini secara eksplisit menangani konteks lokal. Token terbaru $\tilde{k}_t^{win}$ dan $\tilde{v}_t^{win}$ dijaga dalam jendela $w$. Cabang ini mencegah pola lokal mendominasi proses pembelajaran pada cabang kompresi dan seleksi.

---

### 3.4. Desain Kernel

Untuk mencapai percepatan setingkat *FlashAttention*, kami mengimplementasikan kernel atensi jarang selaras perangkat keras menggunakan **Triton**. Optimasi utamanya adalah:
1.  **Pemuatan Data Berpusat pada Grup**: Memuat semua kueri dalam satu grup GQA secara bersamaan ke dalam SRAM karena mereka berbagi blok KV jarang yang sama.
2.  **Shared KV Fetching**: Memuat blok KV secara sekuensial berdasarkan indeks $I_t$ ke SRAM untuk meminimalkan beban memori.

**Deskripsi Gambar 3 (Kernel Design):**
Diagram ini menunjukkan alur data dari HBM (Memori Utama) ke SRAM (Memori Cepat di GPU).
*   **Grid Loop:** Membagi kueri $Q$ menjadi grup-grup.
*   **Inner Loop:** Untuk setiap kueri, ia melakukan "Select In" (Memilih blok KV yang relevan), lalu "Load" (Memindahkan data ke SRAM), kemudian "Compute on SRAM".
*   **Hasil:** Output dikembalikan ke HBM. Warna hijau menunjukkan operasi di SRAM yang sangat cepat.
*   **Kesimpulan Akademis:** Desain ini mencapai intensitas aritmatika yang hampir optimal dengan menghilangkan transfer KV yang redundan.

---

## 4. Eksperimen

### 4.1. Pengaturan Prapelatihan
Model menggunakan tulang punggung 27B parameter (3B aktif) dengan konfigurasi GQA dan *Mixture-of-Experts* (MoE).
*   30 lapisan, dimensi tersembunyi 2560.
*   GQA: 4 grup, total 64 kepala atensi.
*   MoE: 72 pakar terurut, 2 pakar bersama, memilih 6 pakar terbaik per token.

### 4.2. Perbandingan Performa

**Tabel 1: Performa Prapelatihan pada Tolok Ukur Umum**

| Model | MMLU | BBH | GSM8K | MATH | HumanEval | **Avg.** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Full Attn | **0.567** | 0.497 | 0.486 | 0.263 | 0.335 | 0.443 |
| **NSA** | 0.565 | **0.521** | **0.520** | **0.264** | **0.348** | **0.456** |

**Analisis Akademik:** Meskipun NSA sangat jarang (mengurangi banyak komputasi), ia mencapai performa rata-rata yang lebih tinggi (**0.456 vs 0.443**). Keuntungan signifikan terlihat pada tugas penalaran seperti BBH (+0.024) dan GSM8K (+0.034). Ini menunjukkan bahwa mekanisme atensi jarang berfungsi sebagai filter derau, memaksa model fokus pada informasi yang paling relevan.

---

## 5. Analisis Efisiensi

### 5.1. Kecepatan Pelatihan

**Deskripsi Gambar 6:**
Gambar ini membandingkan latensi (ms) antara NSA dan FlashAttention-2.
*   **Sumbu-X:** Panjang konteks (8k, 16k, 32k, 64k).
*   **Sumbu-Y:** Waktu komputasi dalam milidetik (0-2500 ms).
*   **Tren:** Seiring bertambahnya panjang konteks, batang biru (FlashAttention) tumbuh secara kuadratik (sangat cepat). Batang jingga (NSA) tumbuh jauh lebih lambat (linear).
*   **Kesimpulan:** Pada konteks 64k, NSA memberikan percepatan **9.0x** pada tahap *Forward* dan **6.0x** pada tahap *Backward*.

### 5.2. Kecepatan Pengodean (*Decoding*)

**Tabel 4: Volume Akses Memori saat Pengodean**

| Panjang Konteks | 8192 | 16384 | 32768 | 65536 |
| :--- | :---: | :---: | :---: | :---: |
| Full Attention | 8192 | 16384 | 32768 | 65536 |
| NSA | 2048 | 2560 | 3584 | 5632 |
| **Expected Speedup** | **4x** | **6.4x** | **9.1x** | **11.6x** |

**Analisis Akademik:** Karena pengodean adalah proses yang terbatas pada bandwidth memori (*memory-bound*), percepatan NSA berbanding lurus dengan pengurangan volume akses memori. Pada konteks 65k, NSA hanya perlu memuat data setara 5632 token, sementara Atensi Penuh harus memuat 65536 token.

---

## 6. Diskusi dan Visualisasi

### 6.1. Tantangan Strategi Alternatif
Kami menguji strategi berbasis klaster (*Key-Clustering*), namun menemukan tiga masalah:
1.  *Overhead* komputasi tinggi dari mekanisme pengelompokan dinamis.
2.  Ketidakseimbangan beban antar klaster yang menyulitkan optimasi operator.
3.  Kebutuhan untuk pengelompokan ulang periodik yang mengganggu alur pelatihan sekuensial.

### 6.2. Visualisasi Peta Atensi

**Deskripsi Gambar 8:**
Visualisasi ini menunjukkan peta panas (*heatmap*) dari skor atensi pada model Full Attention.
*   **Sumbu-X:** Posisi KV.
*   **Sumbu-Y:** Posisi Kueri.
*   **Pola:** Terlihat bercak-bercak terang yang membentuk klaster-klaster blok.
*   **Kesimpulan Akademis:** Skor atensi secara alami menunjukkan karakteristik pengelompokan blok (*blockwise clustering*). Token yang berdekatan cenderung memiliki tingkat kepentingan yang sama bagi suatu kueri. Hal ini secara empiris memvalidasi desain NSA yang memilih blok kontinu daripada token individual.

---

## 8. Kesimpulan

Kami memperkenalkan **NSA**, arsitektur atensi jarang yang selaras dengan perangkat keras untuk pemodelan konteks panjang yang efisien. Dengan mengintegrasikan kompresi token hierarkis dengan seleksi token tingkat blok dalam arsitektur yang dapat dilatih, sistem kami mencapai akselerasi pelatihan dan inferensi yang signifikan sambil tetap mempertahankan atau melampaui performa Atensi Penuh. NSA memajukan standar teknologi dengan menunjukkan bahwa performa tolok ukur umum, evaluasi konteks panjang, dan kemampuan penalaran dapat ditingkatkan secara bersamaan dengan pengurangan latensi komputasi yang terukur.

***(Tamat)***
