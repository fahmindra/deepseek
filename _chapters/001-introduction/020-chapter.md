---
slug: bab-02
title: "Mengatasi Masalah Kemacetan Kinerja"
layout: chapter
---

# Mengatasi Masalah Kemacetan Kinerja Menggunakan Memori Penyimpanan Sementara (Key-Value Cache)

## **Yang akan kita pelajari pada bab ini:**
*   Ketidakefisienan mendasar dari cara kerja AI yang menebak kata demi kata (inferensi autoregresif).
*   *Key-Value (KV) Cache* (Memori Kunci-Nilai): sebuah solusi cerdas yang sayangnya memakan biaya memori sangat besar.
*   *Multi-Query Attention* (MQA) dan *Grouped-Query Attention* (GQA): Solusi generasi pertama untuk mengatasi batas memori pada KV Cache.

Untuk benar-benar memahami inovasi luar biasa di dalam rancangan bangunan (arsitektur) DeepSeek, kita harus mulai dengan masalah teknis terberat yang sengaja mereka pecahkan. Perjalanan kita mengikuti peta jalan empat tahap yang telah dijelaskan di awal buku ini, dan bab ini didedikasikan sepenuhnya untuk **Tahap 1: Fondasi Key-Value Cache**.

Tahap ini membahas rintangan paling mendasar dalam penggunaan (inferensi) LLM modern: masalah mentoknya memori dan kemampuan komputer. Sebelum kita bisa mengagumi lompatan teknologi canggih seperti *Multi-Head Latent Attention* (MLA) atau *DeepSeek Sparse Attention* (DSA) buatan DeepSeek di Tahap 2, kita harus menguasai dulu mekanisme dasar dari mana teknologi tersebut berasal.

![Gambar 2.1 Peta jalan empat tahap untuk membangun model DeepSeek. Tahap 1 membangun Fondasi Key-Value Cache yang akan kita bahas di bab ini.](https://drek4537l1klr.cloudfront.net/dandekar/v-2/Figures/02__image001.png)

Seperti yang ditunjukkan pada peta jalan di atas, fondasi ini dibangun di atas dua konsep utama: *Key-Value (KV) Cache* itu sendiri, dan cara pengoptimalan generasi pertamanya, yaitu *Multi-Query Attention* (MQA) dan *Grouped-Query Attention* (GQA). Teknik-teknik ini membentuk dasar di mana arsitektur yang lebih canggih dibangun. Tahap 2 dari peta jalan kita nanti akan memperkenalkan inovasi inti dari DeepSeek (seperti MLA, Decoupled RoPE, dan Mixture-of-Experts). Namun sebelum membahas itu, kita akan membagi bab ini menjadi tiga bagian untuk membangun pemahaman dasar kita dari nol:

Pertama, kita akan membangun putaran (loop) pembuatan teks secara utuh. Kita akan melihat langsung bagaimana model bahasa menghasilkan teks satu per satu kata (token). Praktik langsung ini akan membuat kita melihat sendiri betapa tidak efisiennya cara kerja komputer pada pendekatan tradisional ini.

Kedua, kita akan membuat kode untuk *KV Cache*. Ini adalah solusi cerdas yang memecahkan masalah kelambatan awal tadi. Melalui kode kita, kita akan menunjukkan betapa drastisnya peningkatan kecepatannya. Namun, kita juga akan mengungkap "sisi gelapnya": biaya memori yang sangat besar yang menciptakan masalah kemacetan baru, terutama jika AI harus membaca teks sepanjang 1 juta kata.

Terakhir, kita akan membangun lapisan kode PyTorch yang berfungsi untuk MQA dan GQA. Ini adalah arsitektur solusi generasi pertama di industri AI yang dirancang untuk mengurangi masalah memori pada *KV Cache*. Sangat penting untuk dipahami bahwa solusi ini tidaklah gratis; baik MQA maupun GQA secara sadar mengorbankan kualitas dan kecerdasan model demi menghemat memori komputer. MQA adalah bentuk paling ekstrem, yang mengutamakan penghematan memori di atas segalanya. Sementara itu, GQA menawarkan jalan tengah yang lebih seimbang. Dengan membangun teknik-teknik ini dan memahami kelemahannya, kita akan memiliki dasar pengetahuan yang tepat untuk memahami inovasi unik DeepSeek di bab-bab selanjutnya, yang bertujuan untuk mendapatkan yang terbaik dari kedua sisi tanpa ada yang dikorbankan.

---

## **2.1 Putaran penggunaan LLM: Menghasilkan teks satu per satu kata**

Konsep pertama dan paling penting untuk dipahami adalah bahwa *KV cache* hanya berlaku selama tahap **inferensi** (tahap penggunaan) dari sebuah model bahasa. Perbedaan ini sangat penting, jadi mari kita perjelas dua fase utama dalam siklus hidup sebuah LLM.

### **2.1.1 Membedakan masa pelatihan (pre-training) dari masa penggunaan (inference)**

Setiap model bahasa besar, dari GPT-2 versi awal hingga DeepSeek-V4, melewati dua fase yang sangat berbeda:

1.  **Pelatihan (Pra-pelatihan & Pasca-pelatihan):** Ini adalah fase belajar yang sangat besar dan memakan biaya komputer yang sangat mahal. Model dilatih menggunakan sekumpulan data yang sangat luas (triliunan kata) untuk mempelajari tata bahasa, fakta, pola logika, dan hubungan antar kata. Selama fase ini, miliaran parameternya (bobot/jaringan sarafnya) secara aktif diubah dan disesuaikan. Setelah pelatihan selesai, parameter model tersebut dikunci (tidak berubah lagi).
2.  **Inferensi (Penggunaan):** Ini adalah fase "pemakaian". Model yang sudah dilatih dan dikunci tadi sekarang digunakan untuk melakukan tugas. Saat Anda mengobrol dengan ChatGPT atau meminta AI untuk "menulis kode Python," Anda sedang melakukan inferensi. Model tersebut tidak lagi belajar; ia murni menggunakan pengetahuannya yang sudah dikunci untuk menebak kata apa yang akan muncul selanjutnya dalam sebuah kalimat.

Seluruh pembahasan dalam bab ini—dan masalah kemacetan memori yang ingin kita pecahkan—berlaku secara eksklusif (khusus) untuk tahap inferensi. Kita mengasumsikan kita sudah memiliki model yang sudah selesai dilatih, dan tujuan kita adalah menggunakannya untuk menghasilkan teks secepat dan seefisien mungkin.

### **2.1.2 Proses autoregresif: Menambahkan kata untuk membangun konteks**

Selama tahap penggunaan, sebuah model bahasa menghasilkan teks tepat satu per satu kata (token). Meskipun tampilan layar di HP atau komputer Anda mungkin membuatnya terlihat seolah-olah seluruh paragraf muncul sekaligus, di balik layar, ada sebuah proses langkah demi langkah yang sedang berlangsung. Ini disebut **pembuatan teks autoregresif**.

Ide intinya sederhana tetapi sangat menguras tenaga komputer: setiap kata baru yang dihasilkan oleh model akan langsung dimasukkan kembali ke dalam kalimat awal, menjadi bagian dari konteks untuk menebak kata berikutnya. Hal ini menciptakan sebuah putaran yang terus berulang.

![Gambar 2.2 Dalam putaran pembuatan teks autoregresif, hasil keluaran model dari satu langkah ditambahkan ke masukan untuk langkah berikutnya, yang secara bertahap memperpanjang teks konteksnya.](https://drek4537l1klr.cloudfront.net/dandekar/v-2/Figures/02__image002.png)

Mari kita telusuri alur yang ditunjukkan dalam diagram di atas menggunakan sebuah kalimat pancingan (prompt) sederhana: *"The next day"* (Keesokan harinya).

Tugas model AI adalah menebak kata apa yang paling mungkin muncul setelah kalimat ini. Berikut adalah urutan kejadiannya:

1.  **Konteks Awal:** Kita mulai dengan memberikan pancingan awal kepada model: *"The next day."*
2.  **Tebakan Pertama:** Seluruh kalimat ini dimasukkan ke dalam pipa saluran inferensi LLM. Model memproses kalimat tersebut dan menebak kata selanjutnya yang paling mungkin. Dalam hal ini, model menebak kata *"is"* (adalah).
3.  **Tambahkan dan Ulangi:** Kata baru tersebut, *"is,"* ditambahkan ke bagian belakang kalimat. Masukan untuk langkah selanjutnya sekarang menjadi kalimat yang lebih panjang: *"The next day is."* Kalimat baru yang lebih panjang ini dimasukkan kembali ke dalam model dari awal.
4.  **Tebakan Kedua:** Model sekarang memproses *"The next day is"* dan menebak kata selanjutnya: *"bright"* (cerah). Proses ini terus berlanjut, dengan kalimat yang terus bertambah satu kata pada setiap langkahnya.

Putaran ini terus berlanjut sampai model menghasilkan sebuah tanda khusus bernama *end-of-sequence* (`<EOS>` / akhir dari kalimat) atau mencapai batas jumlah kata yang sudah ditentukan. Proses yang berulang dan saling menyambung ini adalah prinsip dasar tentang bagaimana LLM menyusun teks yang masuk akal. Ingatlah alur ini dengan baik, karena ini adalah kunci untuk memahami mengapa *KV cache* itu sangat diperlukan sekaligus sangat bermasalah.

### **2.1.3 Memvisualisasikan pembuatan autoregresif dengan GPT-2**

Kode pemrograman berikut ini mendemonstrasikan putaran autoregresif yang sedang bekerja menggunakan model GPT-2 yang sudah dilatih sebelumnya. Kode dimulai dengan kalimat pancingan awal dan masuk ke dalam sebuah putaran `for`. Di dalam putaran ini, kode melakukan tugas utamanya: ia mengirim kalimat *saat ini* ke model, mendapatkan tebakan untuk kata berikutnya, dan langsung menambahkan kata baru tersebut ke dalam kalimat untuk putaran selanjutnya.

Visualisasi sederhana ini memperjelas kenyataan yang menyakitkan: model komputer harus memproses ulang dan membaca *seluruh kalimat dari awal* untuk setiap satu kata baru yang ia hasilkan.

*(Catatan: Anda dapat menemukan kode ini, beserta kode pemrograman lainnya dalam buku ini, di penyimpanan GitHub resmi: [https://github.com/VizuaraAI/DeepSeek-From-Scratch](https://github.com/VizuaraAI/DeepSeek-From-Scratch).)*

##### Kode Pemrograman 2.1 Memvisualisasikan pembuatan autoregresif dengan GPT-2

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
 
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
 
prompt = "The next day is bright"
inputs = tokenizer(prompt, return_tensors="pt")
input_ids = inputs.input_ids
 
print(f"Prompt: '{prompt}'", end="")
 
for _ in range(20):
    # Model memproses seluruh rangkaian kata saat ini pada setiap putaran.
    outputs = model(input_ids)
    logits = outputs.logits
 
    # Kita hanya menggunakan skor (logits) dari kata yang paling terakhir 
    # untuk menebak kata berikutnya.
    next_token_logits = logits[:, -1, :]
    next_token_id = next_token_logits.argmax(dim=-1).unsqueeze(-1)
 
    # Kata yang baru ditebak kemudian ditambahkan ke kalimat masukan, 
    # yang kemudian dimasukkan kembali ke dalam model di putaran berikutnya. 
    # Ini adalah inti dari proses autoregresif.
    input_ids = torch.cat([input_ids, next_token_id], dim=-1)
 
    new_token = tokenizer.decode(next_token_id[0])
    print(new_token, end="", flush=True)
 
print("\n")
```

Menjalankan kode ini akan menghasilkan teks berikut, di mana teks yang muncul setelah kalimat pancingan awal dibuat satu per satu kata:

> *Prompt: 'The next day is bright' and sunny, and the sun is shining. The sun is shining, and the moon is shining.*

Demonstrasi ini memperjelas satu poin yang sangat penting: model komputer melakukan proses matematika penuh melalui seluruh arsitektur besarnya untuk *setiap satu kata baru*. Kita tahu hal ini terjadi karena perintah pemanggilan `outputs = model(input_ids)` berada di dalam kurung putaran `for`, dan data `input_ids` (yang berisi seluruh sejarah teks kalimat tersebut) dikirimkan ke model secara utuh pada setiap putarannya.

Pengamatan ini membawa kita pada sebuah pertanyaan kritis: apa yang sebenarnya terjadi di dalam perhitungan matematika tersebut, dan apakah menghitung ulang seluruh kalimat dari awal itu benar-benar diperlukan?

---

## **2.2 Tugas Inti: Menebak kata berikutnya**

Untuk memahami perhitungan apa saja yang mungkin kita ulang-ulang secara sia-sia selama proses inferensi, kita harus mengupas lapisan-lapisan arsitektur AI dan melihat ke bagian paling tengah dari blok Transformer: mekanisme **Multi-Head Attention** (Perhatian Banyak-Kepala).

Diagram di bawah ini memberikan gambaran umum dari perjalanan ini. Diagram ini menunjukkan bagaimana contoh kalimat kita, *"The next day is bright,"* mengalir melewati komponen-komponen utama. Ingatlah gambar ini karena ini mewakili seluruh proses yang akan kita bedah sebentar lagi.

![Gambar 2.3 Gambaran umum tingkat tinggi dari arsitektur blok Transformer. Diagram ini mengilustrasikan aliran data lengkap dari kata masukan awal ("The next day is bright") melewati proses embedding, multi-head attention, dan lapisan feed-forward, yang pada akhirnya menghasilkan skor logits yang digunakan untuk menebak kata berikutnya.](https://drek4537l1klr.cloudfront.net/dandekar/v-2/Figures/02__image003.png)

### **2.2.1 Dari Embedding Masukan hingga Vektor Konteks: Penjelasan Matematika**

Mari kita telusuri jalur kalimat masukan kita, *"The next day is,"* melalui satu perhitungan *Attention* (Perhatian) untuk melihat apa yang sebenarnya terjadi di dalam mesin komputer.

**Langkah 1: Mengubah Masukan Menjadi Query, Key, dan Value**

Setelah teks dipecah menjadi kata (tokenisasi) dan diubah menjadi angka (*embedding*), masukan kita diubah menjadi sebuah tabel angka matematika (matriks), yang akan kita sebut **X**. Untuk penjelasan ini, kita akan menggunakan ukuran tabel yang disederhanakan agar matematikanya mudah diikuti. Mari kita asumsikan tabel **X** kita memiliki ukuran (4, 8). Artinya, ada 4 kata, dan masing-masing kata diwakili oleh 8 angka (*dimensi embedding*). Pada model AI sungguhan, dimensi *embedding* ini jauh lebih besar (misalnya, berukuran 7168 pada DeepSeek-V3), tetapi rumus matematika dasarnya sama persis.

Langkah pertama yang dilakukan di dalam blok *attention* adalah memproyeksikan (mengubah) tabel masukan ini menjadi tiga tabel perwakilan yang berbeda: Tabel **Query (Q / Pertanyaan)**, **Key (K / Kunci)**, dan **Value (V / Nilai)**. Hal ini dilakukan dengan mengalikan tabel **X** dengan tiga tabel bobot pelajaran AI yang terpisah: $W_q$, $W_k$, dan $W_v$.

![Gambar 2.4 Matriks embedding masukan X diproyeksikan menjadi tiga matriks baru: Query, Key, dan Value. Setiap proyeksi adalah hasil perkalian matriks dengan matriks bobot unik yang telah dipelajari oleh AI.](https://drek4537l1klr.cloudfront.net/dandekar/v-2/Figures/02__image004.png)

Seperti yang diilustrasikan pada diagram:
*   Masukan **X** (ukuran 4x8) dikalikan dengan $W_q$ (ukuran 8x4) untuk menghasilkan matriks Query (ukuran 4x4).
*   Masukan **X** (ukuran 4x8) dikalikan dengan $W_k$ (ukuran 8x4) untuk menghasilkan matriks Key (ukuran 4x4).
*   Masukan **X** (ukuran 4x8) dikalikan dengan $W_v$ (ukuran 8x4) untuk menghasilkan matriks Value (ukuran 4x4).

Anggaplah ketiga matriks ini sebagai peran. Matriks **Query** (Pertanyaan) mewakili apa yang sedang "dicari" oleh setiap kata. Sedangkan matriks **Key** (Kunci) dan **Value** (Nilai/Isi) mewakili apa yang "ditawarkan" oleh setiap kata kepada kata-kata lain di sekitarnya.

**Langkah 2: Menghitung Skor Perhatian (Attention Scores)**

Selanjutnya, model AI harus menentukan seberapa penting hubungan antara satu kata dengan semua kata lainnya. Kita mengambil matriks Query dan melakukan perkalian matriks (disebut *dot product*) dengan kebalikan posisi dari matriks Key ($Q \times K^T$).

![Gambar 2.5 Perkalian dot product antara matriks Query dan matriks Key yang dibalik posisinya menghasilkan matriks Skor Perhatian (Attention Scores). Setiap elemen dalam matriks ini menunjukkan seberapa kuat hubungan/kepentingan dari satu kata ke kata lainnya.](https://drek4537l1klr.cloudfront.net/dandekar/v-2/Figures/02__image005.png)

Matriks Skor Perhatian yang dihasilkan (berukuran 4x4) mengukur hubungan antara setiap pasangan kata. Misalnya, angka yang berada di baris keempat dan kolom kedua mewakili seberapa besar kata *"is"* harus memberikan perhatian (fokus) kepada kata *"next."*

**Langkah 3: Dari Skor Menjadi Vektor Konteks**

Skor mentah ini kemudian diskalakan (diperkecil) agar proses pelatihannya stabil, lalu sebuah **topeng kausal** (*causal mask*) diterapkan. Karena proses pembuatan bahasa selalu bergerak maju ke depan, sebuah kata hanya bisa mengumpulkan informasi dari kata-kata *sebelumnya*. Topeng ini mencegah model untuk "mencontek" dengan melihat kata di masa depan. Caranya adalah dengan mengubah nilai pada segitiga bagian atas matriks skor perhatian menjadi nol. Terakhir, sebuah fungsi matematika bernama `softmax` mengubah skor-skor tersebut menjadi Bobot Perhatian (*attention weights*)—yaitu sekumpulan angka peluang/probabilitas yang jika dijumlahkan per baris hasilnya adalah 1.

Bobot perhatian akhir ini kemudian dikalikan dengan matriks Value (Nilai).

![Gambar 2.6 Bobot Perhatian dikalikan dengan matriks Value untuk menghasilkan Matriks Konteks akhir. Setiap baris sekarang merupakan bentuk perwakilan baru dari kata asli yang sudah sadar akan konteks di sekitarnya.](https://drek4537l1klr.cloudfront.net/dandekar/v-2/Figures/02__image006.png)

Perkalian terakhir ini menghasilkan **Matriks Konteks** (ukuran 4x4). Setiap baris dalam matriks ini adalah baris angka (vektor) baru yang sangat kaya akan informasi. Angka konteks untuk kata *"is,"* misalnya, kini merupakan jumlah gabungan dari semua angka Value (Nilai) kata-kata yang muncul sebelumnya, dan mengandung informasi makna yang sangat kaya tentang seluruh kejadian di kalimat sebelumnya.

**Langkah 4: Memperbesar menjadi Multi-Head Attention (Perhatian Banyak Kepala)**

Proses yang dijelaskan di atas baru menangani satu buah perhitungan perhatian tunggal. Namun, bahasa manusia itu rumit. Sebuah model AI perlu melacak aturan tata bahasa (apakah subjek dan kata kerjanya cocok) dan hubungan makna kata (semantik) secara bersamaan.

Di sinilah **Multi-Head Attention (MHA)** berperan. Daripada menggunakan satu set matriks perhitungan yang sangat besar, model ini menggunakan banyak set matriks yang lebih kecil dan bekerja secara mandiri—masing-masing disebut "kepala" (*head*).

![Gambar 2.7 Proyeksi Paralel pada Multi-Head Attention. Matriks embedding masukan X diproyeksikan ke matriks Query, Key, dan Value yang terpisah untuk setiap kepala perhatian (attention head).](https://drek4537l1klr.cloudfront.net/dandekar/v-2/Figures/02__image007.png)

Seperti yang ditunjukkan pada Gambar 2.7, jika model kita memiliki dua kepala, masukan awal (ukuran 4, 8) diproyeksikan secara bersamaan (paralel) menjadi enam matriks berukuran kecil (4, 2): $Q_1$, $K_1$, $V_1$ untuk Kepala 1, dan $Q_2$, $K_2$, $V_2$ untuk Kepala 2.

Setiap kepala kemudian menghitung skor perhatiannya sendiri secara sepenuhnya mandiri tanpa melihat kepala yang lain.

![Gambar 2.8 Setiap kepala perhatian menghitung matriks skor perhatiannya sendiri yang unik secara bersamaan (paralel).](https://drek4537l1klr.cloudfront.net/dandekar/v-2/Figures/02__image008.png)

Proses paralel (bekerja bersama-sama) inilah yang menjadi rahasia kekuatan Transformer. Karena setiap kepala memiliki angka acak awal yang unik, mereka belajar untuk melihat kalimat masukan dari sudut pandang yang berbeda. Kepala 1 mungkin menjadi ahli tata bahasa, sementara Kepala 2 menjadi ahli makna kalimat.

![Gambar 2.9 Skor perhatian untuk setiap kepala diproses secara mandiri untuk membuat bobot perhatian akhir.](https://drek4537l1klr.cloudfront.net/dandekar/v-2/Figures/02__image009.png)

Sama seperti kasus kepala tunggal sebelumnya, skor mentah dari setiap kepala diskalakan, ditutupi topeng, dan diubah menjadi angka peluang (probabilitas) melalui rumus *softmax*. Terakhir, setiap kepala mengalikan bobotnya sendiri dengan matriks Value miliknya sendiri untuk menghasilkan Matriks Konteks masing-masing.

![Gambar 2.10 Setiap kepala menghasilkan matriks konteksnya sendiri, yang mewakili sudut pandang uniknya terhadap kalimat masukan.](https://drek4537l1klr.cloudfront.net/dandekar/v-2/Figures/02__image010.png)

Pada titik ini, kita memiliki dua Matriks Konteks (ukuran 4, 2) yang terpisah. Untuk meneruskan data ini ke lapisan otak AI berikutnya, kita harus menyatukan kembali aliran yang terpisah ini.

![Gambar 2.11 Matriks konteks dari semua kepala digabungkan bersama (concatenate) untuk membentuk satu matriks tunggal yang lebih kaya, yang kemudian dilewatkan ke lapisan proyeksi akhir.](https://drek4537l1klr.cloudfront.net/dandekar/v-2/Figures/02__image011.png)

Pertama, kita menyambung (menggabungkan/concatenate) matriks-matriks tersebut di bagian ujungnya, mengubah dua matriks (4, 2) milik kita kembali menjadi satu matriks gabungan (4, 4). Kedua, matriks gabungan ini dilewatkan melalui sebuah **lapisan proyeksi keluaran** (sebuah lapisan matematika linier dengan bobot yang sudah dipelajari) yang berfungsi mencampurkan pemahaman dari berbagai kepala tersebut kembali ke dalam bentuk yang dapat dibaca oleh otak utama AI.

### **2.2.2 Dari Vektor Konteks menuju Logits**

Mekanisme Perhatian (*Attention*) kita telah berhasil menghasilkan Matriks Konteks yang kaya informasi. Untuk kalimat masukan kita *"The next day is,"* kita memiliki matriks (4, 4) di mana setiap baris mencerminkan pemahaman yang mendalam tentang kata tersebut beserta konteks sekitarnya. Sekarang, matriks ini harus melewati sisa blok Transformer untuk menghasilkan tebakan akhir.

**Langkah 1: Jaringan Feed-Forward (FFN)**
Matriks Konteks melewati Jaringan *Feed-Forward* (FFN). Berbeda dengan sistem *attention* (yang membandingkan satu kata dengan kata lainnya), FFN memproses data angka setiap kata secara mandiri dan sendirian. FFN bertindak sebagai pencocok pola yang rumit, menerapkan perubahan matematika tingkat lanjut tanpa mengubah ukuran matriksnya.

**Langkah 2: Mengulang Melalui Blok-Blok Transformer**
Hasil keluarannya melewati proses Normalisasi Lapisan dan sebuah sambungan sisa (menambahkan data masukan awal blok ke data keluaran akhirnya). Seluruh blok utuh ini (Attention + FFN + Normalisasi) kemudian diulang. Datanya mengalir ke blok berikutnya, mengulangi siklus tersebut sebanyak jumlah lapisan yang dimiliki oleh model AI.

**Langkah 3: Proyeksi Akhir Menjadi Logits**
Setelah keluar dari blok Transformer terakhir, matriks tersebut melewati **Lapisan Keluaran** (*Output Layer*). Lapisan linier ini mengubah kumpulan angka konteks kita menjadi ruang yang sangat luas sebesar jumlah total kosakata yang diketahui model AI tersebut.

> **Definisi: Apa itu Logits?**
> Logit adalah skor mentah yang belum diubah menjadi persentase pasti. Untuk setiap posisi kata dalam sebuah kalimat, model menghasilkan angka logit untuk setiap kata yang ada dalam kamusnya. Logit tertinggi sesuai dengan kata berikutnya yang paling mungkin muncul.

![Gambar 2.12 Perjalanan dari Matriks Konteks terakhir menuju Matriks Logits. Lapisan Keluaran memproyeksikan setiap baris angka yang kaya konteks menjadi barisan skor yang sangat panjang, di mana setiap skor mewakili satu kata dalam kamus kosakata AI.](https://drek4537l1klr.cloudfront.net/dandekar/v-2/Figures/02__image012.png)

Seperti yang ditunjukkan pada Gambar 2.12, setiap dari 4 baris dalam Matriks Konteks kita diubah menjadi barisan angka dengan panjang 50.257 (yaitu jumlah ukuran kosakata kamus milik model GPT-2). Hal ini menghasilkan sebuah **Matriks Logits** raksasa yang berukuran (4, 50257). Baris pertama berisi tebakan untuk kata apa yang muncul setelah *"The"*, baris kedua untuk apa yang muncul setelah *"next"*, dan seterusnya.

### **2.2.3 Fakta terpenting: Mengapa hanya baris terakhir yang berguna**

Kita sekarang memiliki Matriks Logits (4, 50257) yang sangat besar berisi tebakan untuk *setiap* posisi kata. Akan tetapi, tujuan asli kita sangatlah spesifik: kita *hanya* ingin menebak satu kata tunggal yang muncul *setelah* keseluruhan kalimat masukan kita, *"The next day is."*

**Ini berarti kita bisa membuang hampir seluruh informasi yang ada di dalam Matriks Logits tersebut.**

*   Baris 1 (tebakan kata setelah "The") tidak berguna bagi kita saat ini.
*   Baris 2 (tebakan kata setelah "next") tidak berguna.
*   Baris 3 (tebakan kata setelah "day") tidak berguna.

*Satu-satunya* baris yang kita pedulikan hanyalah baris paling akhir: baris logit yang berhubungan dengan kata *"is."*

Ini adalah fakta yang paling penting dan krusial dalam proses inferensi LLM: **Karena kita hanya menggunakan baris terakhir untuk membuat tebakan, maka menghitung ulang skor logits untuk semua baris sebelumnya pada setiap putaran pembuatan kata adalah tindakan yang sangat membuang-buang tenaga komputer.**

![Gambar 2.13 Langkah tebakan akhir. Barisan logits untuk kata terakhir diubah menjadi tabel peluang (probabilitas), dan kata dengan peluang tertinggi dipilih sebagai kata yang akan dimunculkan ke layar.](https://drek4537l1klr.cloudfront.net/dandekar/v-2/Figures/02__image013.png)

Untuk memilih kata berikutnya, kita melakukan hal berikut:
1.  **Pisahkan Barisan Logits Terakhir:** Kita potong bagian baris paling bawah saja (ukurannya menjadi 1, 50257).
2.  **Terapkan Softmax:** Kita mengubah skor mentah ini menjadi pembagian persentase (probabilitas) yang jika dijumlahkan hasilnya pasti 1 (100%).
3.  **Pilih Katanya:** Kita melakukan operasi komputer bernama `argmax` untuk mencari skor persentase yang paling tinggi, yang akhirnya mengungkapkan kata tebakan kita (misalnya, *"bright"*).

Seluruh jalur perhitungan yang rumit ini dijalankan untuk *setiap satu kata*. Menyadari bahwa tebakan akhir *hanya* bergantung pada vektor kata terakhir (yang di dalamnya sudah berisi ringkasan dari kata-kata masa lalu berkat *self-attention*), seharusnya segera menjadi tanda bahaya bagi kita. Jika kita terus-menerus mengirimkan kalimat yang semakin panjang kembali ke dalam model, apakah kita sedang melakukan perhitungan matematika tak berguna dalam jumlah yang sangat tidak masuk akal di blok *attention*?

Jawabannya adalah: Ya, sangat.

---

## **2.3 Masalah perhitungan yang berulang-ulang (redundan)**

Kita telah menetapkan dua fakta penting:
1.  Model menghasilkan teks secara autoregresif, yaitu dengan mengirimkan teks hasil keluarannya sendiri kembali ke dalam sebagai masukan baru.
2.  Untuk menebak kata berikutnya, model tersebut hanya memerlukan hasil matematika dari kata yang posisinya paling *terakhir* di dalam kalimat tersebut.

Jika model terus-menerus memproses ulang kalimat yang semakin panjang hanya untuk mengambil data dari kata terakhir saja, pemborosan tenaga yang terjadi sangatlah mengejutkan.

### **2.3.1 Intuisi Logika: Apakah kita sedang menghitung hal yang sama berulang kali?**

Bayangkan proses ini seperti Anda sedang membaca buku.
*   **Langkah 1:** Untuk memahami Bab 1, Anda membaca Bab 1.
*   **Langkah 2:** Untuk memahami Bab 2, Anda membaca ulang Bab 1 dari awal, lalu membaca Bab 2.
*   **Langkah 3:** Untuk memahami Bab 3, Anda membaca ulang Bab 1, membaca ulang Bab 2, lalu membaca Bab 3.

Jika membaca satu bab membutuhkan waktu yang pasti (tetap), maka membaca sampai bab ke-$n$ akan membutuhkan tenaga sebanyak $1 + 2 + 3 ... + n$ unit pekerjaan. Secara matematika, hal ini berkembang secara kuadrat, yang disimbolkan sebagai $O(n^2)$.

Mari kita lihat data yang mengalir ke dalam model AI kita:
*   **Waktu T=3:** Masukannya adalah *"The next day"*. Hasil tebakannya adalah *"is"*.
*   **Waktu T=4:** Masukannya adalah *"The next day is"*. Hasil tebakannya adalah *"bright"*.
*   **Waktu T=5:** Masukannya adalah *"The next day is bright"*. Hasil tebakannya adalah *"and"*.

Pada Langkah 4, kita memproses ulang kata *"The"*, *"next"*, dan *"day"* dari nol. Pada Langkah 5, kita memproses ulang semuanya lagi dari nol. Setiap kata baru yang dibuat akan memaksa komputer (GPU) untuk memproses data berlipat ganda secara eksponensial. Ini membuat pembuatan kalimat yang panjang menjadi sangat lambat dan menyiksa. Mari kita buktikan hal ini secara matematis.

### **2.3.2 Bukti matematis: Melihat perhitungan yang diulang**

Mari kita lihat lebih dekat dua langkah yang berurutan di dalam mekanisme *attention* untuk membuktikan bahwa kita sedang melakukan perhitungan matematika yang sama persis dua kali.

**Langkah A: Penggunaan di Waktu T=4 (Masukan: "The next day is")**

![Gambar 2.14 Perhitungan attention secara utuh untuk kalimat masukan "The next day is."](https://drek4537l1klr.cloudfront.net/dandekar/v-2/Figures/02__image014.png)

Lacak aliran pada Gambar 2.14: Matriks masukan **X** (4, 8) diproyeksikan menjadi **Q, K, V** berukuran (4, 4). Kita menghitung Skor Perhatian $4 \times 4$, menerapkan topeng batas dan *softmax*, lalu mengalikannya dengan matriks **V** untuk mendapatkan Matriks Konteks berukuran (4, 4) kita. Kita menggunakan baris terakhirnya saja untuk menebak kata *"bright"*.

**Langkah B: Penggunaan di Waktu T=5 (Masukan: "The next day is bright")**

Kita menambahkan kata *"bright"* ke belakang, dan mengirim kalimat 5 kata yang baru ini kembali ke dalam.

![Gambar 2.15 Perhitungan attention secara utuh untuk masukan baru yang terdiri dari 5 kata ini.](https://drek4537l1klr.cloudfront.net/dandekar/v-2/Figures/02__image015.png)

Lihat baik-baik pada Gambar 2.15. Empat baris pertama dari matriks masukan **X** (5, 8) kita yang baru ini ternyata *sama persis* dengan seluruh matriks masukan dari langkah sebelumnya. Karena matriks bobot ($W_q, W_k, W_v$) miliki AI selalu dikunci secara kaku selama tahap penggunaan, maka **empat baris pertama dari matriks Query, Key, dan Value kita yang baru ini hasilnya sama persis dengan matriks yang baru saja kita hitung sepersekian milidetik yang lalu.**

Pengulangan sia-sia ini mengalir terus secara beruntun sampai ke matriks Skor Perhatian. Skor pada posisi baris ke-$j$ dan kolom ke-$i$ adalah hasil kali dari pertanyaan (Query) $j$ dan kunci (Key) $i$. Karena empat pertanyaan dan kunci pertama belum berubah sama sekali, maka seluruh kotak berukuran $4 \times 4$ di bagian kiri atas dari matriks Skor Perhatian berukuran $5 \times 5$ kita yang baru ini adalah hasil menghitung ulang angka-angka yang sama persis.

Kita sedang sibuk menghitung ulang seluruh sejarah masa lalu hanya untuk menghitung satu baris baru di bagian bawahnya, dan kemudian kita membuang semua baris masa lalu yang sudah dihitung tersebut.

### **2.3.3 Dampak terhadap performa komputer: Dari rumit kuadrat menjadi ringan lurus (linier)**

Tanpa ada pengoptimalan, mekanisme perhitungan inti *attention* ini bersifat kuadratik: disimbolkan $O(n^2)$. Semakin panjang teks ($n$), jumlah perhitungannya melonjak secara kuadrat.

*   Untuk 4 kata: Kita harus menghitung kotak matriks $4 \times 4$ (16 skor).
*   Untuk 5 kata: Kita harus menghitung kotak matriks $5 \times 5$ (25 skor).
*   Untuk 100.000 kata: Kita harus menghitung kotak matriks $100.000 \times 100.000$ (10 miliar skor).

Seiring panjang kalimat ($n$) bertambah, jumlah pekerjaan komputer meledak. Setiap kata baru butuh waktu yang semakin lama untuk dibuat oleh AI.

![Gambar 2.16 Sebuah grafik yang membandingkan pertumbuhan perhitungan komputer kuadratik yang meledak (O(n²)) pada sistem AI autoregresif tanpa cache (tanpa memori) dibandingkan dengan pertumbuhan linier yang ideal (O(n)).](https://drek4537l1klr.cloudfront.net/dandekar/v-2/Figures/02__image016.png)

Tujuan para insinyur yang merancang AI adalah untuk mengubah proses $O(n^2)$ yang meledak ini menjadi proses $O(n)$ yang linier (lurus bersahabat). Jika kita bisa mencapai perhitungan linier, maka kalau panjang teks berlipat ganda, pekerjaan komputer hanya bertambah dua kali lipat saja, sehingga mencegah kelambatan yang parah.

Hal inilah yang bisa dicapai dengan metode *caching* (membuat memori simpanan). Jika kita menyimpan hasil rumus matematika dari kata-kata masa lalu, maka satu-satunya perhitungan baru yang diperlukan oleh komputer hanyalah untuk menghitung kata yang paling baru. Dengan begitu, menghasilkan kata yang ke-1.000 akan memakan waktu yang kira-kira sama cepatnya dengan menghasilkan kata yang ke-10.

---

## **2.4 Solusi: Caching (Memori Simpanan) untuk efisiensi**

Jika kita menyadari bahwa kita berulang kali menghitung nilai yang sama persis untuk kata-kata di masa lalu, seharusnya kita menghitungnya cukup sekali saja lalu menyimpannya di memori komputer. Inilah yang dinamakan **Key-Value (KV) Cache**. Dengan memutus siklus kerja kuadratik yang melelahkan tersebut, teknologi ini membuat LLM modern akhirnya bisa digunakan oleh publik.

### **2.4.1 Apa saja yang harus disimpan ke memori? Penjelasan langkah demi langkah**

Untuk memahami secara pasti data apa saja yang perlu dimasukkan ke dalam memori *cache*, kita perlu mundur selangkah dari tujuan akhir kita: kita hanya menginginkan vektor konteks akhir untuk satu kata terbaru kita, yaitu *"bright."*

![Gambar 2.17 Dari keseluruhan matriks konteks yang ada, hanya baris paling bawah, yang sesuai dengan kata terbaru, yang diperlukan untuk menebak kata berikutnya dalam kalimat.](https://drek4537l1klr.cloudfront.net/dandekar/v-2/Figures/02__image017.png)

Untuk mendapatkan vektor konteks khusus untuk *"bright"*, kita harus mengalikan barisan bobot perhatian (*attention weights*) milik *"bright"* dengan keseluruhan matriks *Value* (Nilai).

![Gambar 2.18 Vektor konteks untuk kata "bright" dihitung dengan mengalikan bobot perhatian milik kata "bright" dengan matriks Value yang utuh.](https://drek4537l1klr.cloudfront.net/dandekar/v-2/Figures/02__image018.png)

Oleh karena itu, benda-benda yang kita butuhkan adalah:
1.  **Baris Bobot Perhatian khusus untuk "bright"** (sebuah vektor berukuran $1 \times 5$).
2.  **Matriks Value utuh (V)** yang berisi isi/makna untuk kelima kata dalam kalimat.

Lalu bagaimana cara kita mendapatkan Bobot Perhatian untuk *"bright"*? Kita menghitungnya dari perkalian vektor matriks Query milik *"bright"* dengan kebalikan dari matriks Key yang utuh.

![Gambar 2.19 Bobot perhatian untuk kata "bright" didapatkan dari vektor Query miliknya yang diadu dengan vektor Key dari semua kata dalam kalimat tersebut.](https://drek4537l1klr.cloudfront.net/dandekar/v-2/Figures/02__image019.png)

Jadi, hal minimal mutlak yang harus kita miliki untuk menebak kata selanjutnya adalah:
*   Vektor **Query** khusus untuk kata *"bright"*.
*   Matriks **Key (K)** secara utuh untuk *semua* kata di kalimat itu.
*   Matriks **Value (V)** secara utuh untuk *semua* kata di kalimat itu.

Ketika kata *"bright"* yang baru ini masuk, kita segera mengubah angkanya (memproyeksikannya) menjadi vektor Query, Key, dan Value-nya sendiri.

![Gambar 2.20 Tiga perhitungan penting untuk sebuah kata baru. Matriks masukan untuk kata "bright" diproyeksikan untuk membuat vektor Query, Key, dan Value miliknya sendiri.](https://drek4537l1klr.cloudfront.net/dandekar/v-2/Figures/02__image020.png)

Sekarang pertanyaannya, bagaimana dengan Key dan Value untuk semua kata-kata yang lama (*"The next day is"*?) Pada cara awal yang bodoh tadi, kita menghitung ulangnya dari awal. Pada cara yang sudah dioptimalkan, **kita cukup mengambil data masa lalu itu dari dalam KV Cache (Memori).**

Kita *tidak* menyimpan Query ke dalam memori. Kenapa? Karena vektor Query hanya digunakan sesekali untuk mengajukan pertanyaan mewakili kata yang *saat ini* sedang aktif. Begitu kata *"bright"* selesai menemukan konteksnya, vektor Query miliknya tidak akan pernah dibutuhkan lagi untuk menebak kata-kata di masa depan. Sebaliknya, Key (identitas) dan Value (isi) bertindak sebagai perpustakaan sejarah bagi model AI, sehingga mereka harus disimpan di memori selama mungkin.

### **2.4.2 Putaran penggunaan baru yang lebih cepat dengan KV caching**

Putaran pembuatan kata (autoregresif) sekarang menjadi sangat dioptimalkan:

1.  **Menerima Kata:** Model komputer menerima satu buah kata (*token*) baru.
2.  **Menghitung Perubahan:** Komputer melakukan perhitungan matriks untuk mendapatkan vektor Q, K, dan V *hanya khusus untuk satu kata baru ini saja*.
3.  **Mengambil Data Lama:** Komputer memuat/mengambil matriks K dan V masa lalu dari *KV Cache*.
4.  **Menyambung Data:** Komputer menggabungkan vektor K dan V yang baru dengan matriks yang baru saja diambil dari memori.
5.  **Menghitung Perhatian (Attention):** Komputer mengalikan vektor Q *baru* dengan matriks K yang *sudah digabungkan*.
6.  **Menghitung Konteks:** Komputer mengalikan hasil bobot perhatian tadi dengan matriks V yang *sudah digabungkan*.
7.  **Menebak:** Komputer menghasilkan skor (logits) untuk menebak kata berikutnya.
8.  **Memperbarui Memori:** Komputer menyimpan matriks K dan V yang sudah ditambahkan kata baru ini kembali ke dalam memori kartu grafis (GPU).

Perhitungan perkalian angka matriks yang berat untuk kata-kata masa lalu kini hanya dilakukan tepat satu kali saja, tidak diulang lagi.

### **2.4.3 Mendemonstrasikan peningkatan kecepatan dari KV caching**

Kita bisa membuktikan peningkatan kecepatan ini menggunakan kode pemrograman `transformers` dari Hugging Face, yang memungkinkan kita menyalakan atau mematikan tombol sistem memori `use_cache`.

##### Kode Pemrograman 2.2 Mendemonstrasikan Peningkatan Kecepatan KV Caching

```python
prompt = "The next day is bright"
inputs = tokenizer(prompt, return_tensors="pt")
input_ids = inputs.input_ids
attention_mask = inputs.attention_mask
 
# Menghitung waktu TANPA KV cache (memaksa komputer berhitung 
# dari awal secara O(n^2))
start_time_without_cache = time.time()
output_without_cache = model.generate(input_ids,
    max_new_tokens=100,
    use_cache=False,
    attention_mask=attention_mask)
duration_without_cache = time.time() - start_time_without_cache
print(f"Waktu tanpa KV Cache: {duration_without_cache:.4f} detik")
 
# Menghitung waktu DENGAN KV cache (beroperasi secara cepat dan 
# ringan secara linier O(n))
start_time_with_cache = time.time()
output_with_cache = model.generate(input_ids,
    max_new_tokens=100,
    use_cache=True,
    attention_mask=attention_mask)
duration_with_cache = time.time() - start_time_with_cache
print(f"Waktu dengan KV Cache: {duration_with_cache:.4f} detik")
 
speedup = duration_without_cache / duration_with_cache
print(f"\nPeningkatan Kecepatan KV Cache: {speedup:.2f}x")
```

Menjalankan kode ini pada perangkat komputer standar akan memberikan hasil sebagai berikut:

```text
Waktu tanpa KV Cache: 30.9818 detik
Waktu dengan KV Cache: 6.1630 detik
Peningkatan Kecepatan KV Cache: 5.03x
```

Dengan menghilangkan perhitungan berulang yang sia-sia, kita bisa membuat komputer bekerja 5 kali lebih cepat hanya untuk menghasilkan 100 kata. Untuk dokumen teks yang sangat panjang, peningkatan kecepatan ini bisa mencapai ratusan kali lebih cepat. Akan tetapi, mukjizat kecepatan ini datang dengan hukuman yang sangat parah.

---

## **2.5 Sisi gelap dari KV cache: Biaya memori komputer**

Kita telah berhasil memecahkan masalah kemacetan beban kerja prosesor, tetapi sebagai gantinya, kita menciptakan **kemacetan memori** yang parah.

Untuk menggunakan *KV cache*, matriks Key dan Value yang ukurannya sangat raksasa ini harus disimpan secara fisik di dalam Memori Berkecepatan Tinggi (HBM) pada kartu grafis (GPU), dan memori ini harus terus-menerus dipindahkan bolak-balik ke inti komputernya. Penggunaan LLM dengan cepat berubah dari sebelumnya *terbatas oleh seberapa kuat otak komputernya*, menjadi *terbatas oleh seberapa besar kapasitas memorinya*. Kita pada dasarnya sedang menukar ruang VRAM fisik komputer demi menghemat waktu proses.

### **2.5.1 Rumus KV cache: Membedah ukuran memori**

Kita bisa menghitung secara matematika ukuran VRAM (memori kartu grafis) yang pasti untuk menampung *KV cache* ini.

![Gambar 2.21 Rumus untuk menghitung ukuran KV cache dan penerapannya pada model-model AI yang terkenal.](https://drek4537l1klr.cloudfront.net/dandekar/v-2/Figures/02__image021.png)

Rumusnya adalah:

$$ \text{Ukuran KV Cache (Byte)} = 2 \times 2 \times l \times u \times n \times b \times z $$

*   **l (layers / lapisan):** Jumlah total blok Transformer.
*   **u (batch size / pengguna):** Jumlah tugas/kalimat yang diproses dalam waktu bersamaan.
*   **n (heads / kepala):** Jumlah kepala perhatian (*attention heads*) per lapisan.
*   **b (head size / ukuran kepala):** Ukuran/dimensi dari vektor K dan V.
*   **z (sequence length / panjang teks):** Jumlah kata/token yang sedang dibaca.
*   **Angka 2 pertama:** Karena kita menyimpan *dua* jenis matriks (Key dan Value).
*   **Angka 2 kedua:** Untuk presisi angka. Angka desimal 16-bit (FP16/BF16) memakan 2 byte memori.

Seiring berjalannya waktu, model AI dibuat semakin canggih dengan lapisan ($l$) dan kepala ($n$) yang semakin banyak. Selain itu, pengguna internet juga meminta AI mampu membaca teks cerita yang sangat panjang ($z$). Akibatnya, hasil hitungan dari rumus ini meledak tak terkendali.

### **2.5.2 Masalah ukuran di dunia nyata**

![Gambar 2.22 Perbandingan jumlah jaringan otak (parameter) model versus ukuran KV cache-nya untuk berbagai varian model GPT-3.](https://drek4537l1klr.cloudfront.net/dandekar/v-2/Figures/02__image023.png)

Coba bayangkan sebuah arsitektur AI modern seperti DeepSeek-V3 atau V4, yang memiliki 61 lapisan, 128 kepala (dengan ukuran per kepalanya 128), dan kemampuan membaca teks panjang hingga **1.000.000 kata (token)**.

Jika kita menerapkan sistem *caching Multi-Head Attention* standar ke urutan 1 juta kata untuk hanya melayani satu orang pengguna saja, maka *KV cache* ini sendirian akan menghabiskan memori VRAM lebih dari **3,8 Terabyte**. Padahal, satu keping kartu grafis kelas dunia yaitu NVIDIA H100 GPU yang sangat mahal hanya memiliki memori 80GB. Untuk melayani satu pertanyaan dari pengguna saja, Anda akan membutuhkan satu rak server berisi 48 kartu grafis tersebut hanya untuk menampung memori cache-nya saja.

Inilah fakta pahit dalam pemasangan AI modern. Tekanan pada memori membuat teknik *attention* standar menjadi mustahil secara fisik untuk digunakan pada teks yang sangat panjang. Krisis kelangsungan hidup inilah yang memaksa perusahaan pembuat AI untuk menciptakan jalan pintas pada rancangan AI mereka, yang melahirkan: *Multi-Query Attention* dan *Grouped-Query Attention*.

---

## **2.6 Pendekatan yang mengutamakan hemat memori: Multi-Query Attention (MQA)**

Apa cara paling ekstrem (brutal) untuk memotong rumus *KV Cache* tadi? Jawabannya adalah *Multi-Query Attention* (MQA). MQA mengusulkan perubahan desain AI yang sangat radikal: bagaimana jika *semua* kepala ahli (attention heads) yang berbeda-beda tadi dipaksa untuk berbagi satu buah matriks Key dan Value yang sama persis?

### **2.6.1 Ide intinya: Berbagi satu Kunci dan Nilai**

Pada sistem standar *Multi-Head Attention* (MHA), setiap kepala bertindak sebagai seorang ahli yang mandiri dengan otak matriks $W_k$ dan $W_v$ miliknya sendiri-sendiri.

![Gambar 2.23 Sistem standar Multi-Head Attention (MHA). Masing-masing dari empat kepala perhatian ini memiliki matriks bobot Key dan Value tersendiri yang berbeda-beda.](https://drek4537l1klr.cloudfront.net/dandekar/v-2/Figures/02__image024.png)

MQA menyederhanakan hal ini secara paksa. Meskipun setiap kepala masih boleh menyimpan proyeksi Query-nya masing-masing (sehingga setiap kepala masih bisa mengajukan pertanyaan yang berbeda), **semua kepala tersebut dipaksa untuk meminjam dan memakai satu set matriks Key dan Value yang sama secara bersama-sama.**

![Gambar 2.24 Multi-Query Attention (MQA). Keempat kepala masih memiliki proyeksi Query yang unik, tetapi sekarang mereka meminjam satu proyeksi Key dan Value yang sama.](https://drek4537l1klr.cloudfront.net/dandekar/v-2/Figures/02__image025.png)

Pada Gambar 2.24, $K_1, K_2, K_3,$ dan $K_4$ sebenarnya adalah salinan matematik dari satu hal yang sama persis. Dampak dari hal ini sangatlah besar: alih-alih kita menyimpan salinan matriks Key/Value sebanyak $n$ (jumlah kepala) ke memori, sekarang kita hanya perlu menyimpan **satu buah** saja.

### **2.6.2 Dampaknya terhadap rumus KV cache**

Dengan memaksa semua kepala berbagi satu pasangan K/V tunggal, variabel $n$ (jumlah kepala) di rumus kita yang sebelumnya besar, secara efektif langsung berubah menjadi angka 1.

$$ \text{Ukuran KV Cache MQA} = 2 \times 2 \times l \times u \times \mathbf{1} \times b \times z $$

Bagi sebuah model AI yang memiliki 128 kepala, **MQA mengurangi ukuran KV Cache hingga 128 kali lipat.** Ukuran memori komputer yang secara teori butuh 3,8 Terabyte tadi, kini menyusut drastis menjadi sekitar 30GB saja—sehingga muat dengan lega di dalam satu buah kartu grafis (GPU) standar.

### **2.6.3 Harga yang harus dibayar: Hilangnya kecerdasan bahasa**

Penghematan memori besar-besaran ini harus dibayar dengan harga yang mahal: kecerdasan dan kemampuan AI memahami bahasa yang rumit jadi menurun drastis. Pertimbangkan kalimat ambigu (bermakna ganda) ini: *"The artist painted the portrait of a woman with a brush."* (Seniman itu melukis potret seorang wanita dengan kuas).

*   Makna A: Seniman menggunakan kuas untuk melukis.
*   Makna B: Wanita di dalam lukisan tersebut sedang memegang kuas.

Sistem MHA standar dapat menangani makna ganda ini dengan sangat indah. Kepala 1 (si ahli tata bahasa) menggunakan matriks K/V unik miliknya untuk menghubungkan kata "painted" (melukis) dan "brush" (kuas). Kepala 2 (si ahli makna kalimat) menggunakan matriks K/V unik miliknya yang berbeda untuk menghubungkan kata "woman" (wanita) dan "brush" (kuas). Kedua makna tersebut bisa dipahami oleh AI secara bersamaan tanpa tertukar.

Sedangkan pada MQA, karena AI hanya memiliki satu matriks Key tunggal untuk semua kepala, matriks itu harus bertindak sebagai alat serba guna yang generik. Ketika Kepala 1 dan Kepala 2 mengirimkan pertanyaan (Query), mereka hanya bisa melihat ke jawaban (Key) yang sama dan generik tentang kata "brush". Akibatnya, model AI ini akan kesulitan untuk menyandikan kedua makna halus tersebut secara bersamaan. Kemampuan menalar dan pemahaman bahasanya akan menurun. MQA murni merupakan solusi kompromi yang hanya memikirkan "memori yang penting hemat".

### **2.6.4 Membangun lapisan MQA dari nol melalui kode**

Menulis kode MQA di bahasa pemrograman PyTorch cukup mudah. Kita cukup membuat satu perhitungan untuk K dan V, lalu menggunakan perintah `.repeat()` (ulang) untuk membagikannya ke semua kepala tanpa harus menggandakan datanya di memori komputer.

##### Kode Pemrograman 2.3 Membangun lapisan MQA dari nol

```python
import torch
import torch.nn as nn
 
class MultiQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.0):
        super().__init__()
        assert d_model % num_heads == 0, \
            "d_model harus bisa dibagi rata dengan num_heads"
 
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads # Catatan: Pembagian bulat 
                                           # untuk PyTorch
 
        self.W_q = nn.Linear(d_model, d_model)
        # Lapisan Key dan Value sekarang hanya dibuat satu kali (tunggal). 
        # Inilah perubahan rancangan utama yang sangat menghemat memori 
        # KV cache.
        self.W_k = nn.Linear(d_model, self.d_head)
        self.W_v = nn.Linear(d_model, self.d_head)
        self.W_o = nn.Linear(d_model, d_model)
 
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(
        torch.ones(1, 1, 1024, 1024), diagonal=1))
 
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
 
        q = self.W_q(x).view(batch_size, seq_len, self.num_heads, 
        self.d_head).transpose(1, 2)
 
        k = self.W_k(x).view(batch_size, seq_len, 1, 
        self.d_head).transpose(1, 2)
        v = self.W_v(x).view(batch_size, seq_len, 1, 
        self.d_head).transpose(1, 2)
 
        # Tensor K dan V tunggal tadi "diulang" agar jumlahnya cocok  
        # dengan kepala Query. 
        # Ini menciptakan ilusi salinan tanpa harus membuang memori 
        # untuk menggandakan data secara sungguhan.
        k = k.repeat(1, self.num_heads, 1, 1)
        v = v.repeat(1, self.num_heads, 1, 1)
 
        attn_scores = (q @ k.transpose(-2, -1)) / (self.d_head ** 0.5)
 
        attn_scores = attn_scores.masked_fill(
        self.mask[:,:,:seq_len,:seq_len] == 0, float('-inf'))
 
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
 
        context_vector = (attn_weights @ v).transpose(1, 2) \
        .contiguous().view(batch_size, seq_len, self.d_model)
 
        output = self.W_o(context_vector)
        return output
```

---

## **2.7 Mencari jalan tengah: Grouped-Query Attention (GQA)**

Untuk mengurangi kebodohan bahasa yang terjadi pada MQA, para peneliti mengembangkan *Grouped-Query Attention* (GQA). GQA memberikan kompromi yang masuk akal dan bisa diatur di antara kedua sisi. Ia berusaha mencari keseimbangan antara AI yang sangat pintar pada MHA, dan AI yang super hemat memori pada MQA.

### **2.7.1 Ide intinya: Berbagi kunci dan nilai di dalam kelompok kecil**

Alih-alih memaksa *semua* kepala untuk miskin dan berbagi satu pasangan K/V, GQA menciptakan kelompok-kelompok kecil.

![Gambar 2.25 Grouped-Query Attention (GQA). Keempat kepala perhatian ini dibagi ke dalam dua kelompok terpisah.](https://drek4537l1klr.cloudfront.net/dandekar/v-2/Figures/02__image028.png)

Pada Gambar 2.25, Kepala 1 dan 2 (Kelompok 1) berbagi satu pasangan K/V. Sementara itu, Kepala 3 dan 4 (Kelompok 2) berbagi pasangan K/V lain yang beda rumus matematikanya. Sistem pembagian kelompok ini memungkinkan kelompok kepala tertentu untuk memfokuskan keahliannya di bidang tertentu (mengembalikan sebagian kecerdasan AI yang hilang), namun tetap mampu menghemat memori karena jumlah matriks yang disimpan jauh lebih sedikit daripada MHA asli.

### **2.7.2 Tombol pengatur: Menyeimbangkan memori dan kecerdasan**

GQA memperkenalkan variabel baru yaitu $g$ (jumlah kelompok / grup) ke dalam rumus *cache* kita:

$$ Size_{GQA} = 2 \times 2 \times l \times u \times \mathbf{g} \times b \times z $$

*   Jika angka $g$ sama besarnya dengan jumlah seluruh kepala ($n$), maka GQA akan menjadi MHA asli (Kecerdasan AI maksimal, tapi memori boros maksimal).
*   Jika angka $g$ diubah menjadi angka 1, maka GQA berubah kembali menjadi MQA (Kecerdasan AI jatuh minimal, tapi memori sangat hemat).
*   Dengan mengatur $g$ ke angka pecahan tengah dari $n$ (contohnya, model AI bernama Llama 3 menggunakan 8 kelompok untuk 32 kepalanya), kita berhasil mencapai keseimbangan pengurangan memori yang pas.

### **2.7.3 Membangun lapisan GQA dari nol melalui kode**

Di bahasa pemrograman PyTorch, kita menggunakan perintah `repeat_interleave` untuk membagikan jatah K/V kelompok kepada anggota kepala Query yang ditugaskan ke kelompok tersebut.

##### Kode Pemrograman 2.4 Membangun lapisan GQA dari nol

```python
import torch
import torch.nn as nn
 
class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_groups, 
    dropout=0.0, max_seq_len: int = 0):
        super().__init__()
        assert d_model % num_heads == 0, \
            "d_model harus bisa dibagi rata oleh num_heads"
        assert num_heads % num_groups == 0, \
             "num_heads harus bisa dibagi rata oleh num_groups"
 
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.d_head = d_model // num_heads
 
        self.W_q = nn.Linear(d_model, d_model)
        # Kita membuat sejumlah 'num_groups' (kelompok). 
        # Ini adalah tombol putar yang bisa kita sesuaikan.
        self.W_k = nn.Linear(d_model, self.num_groups * self.d_head)
        self.W_v = nn.Linear(d_model, self.num_groups * self.d_head)
        self.W_o = nn.Linear(d_model, d_model)
 
        self.dropout = nn.Dropout(dropout)
        self._register_mask_buffer(max_seq_len)
 
    def forward(self, x):
        B, T, _ = x.shape
        
        q = self.W_q(x).view(B, T, self.num_heads, 
        self.d_head).transpose(1, 2)

        # Diproyeksikan ke sejumlah grup Key dan Value yang berbeda-beda 
        # sesuai jumlah num_groups.
        k = self.W_k(x).view(B, T, self.num_groups, 
        self.d_head).transpose(1, 2)
        v = self.W_v(x).view(B, T, self.num_groups, 
        self.d_head).transpose(1, 2)
 
        heads_per_group = self.num_heads // self.num_groups
        
        # perintah repeat_interleave akan menyiarkan jatah grup K/V 
        # ke kepala-kepala anggota di bawahnya. 
        k = k.repeat_interleave(heads_per_group, dim=1)
        v = v.repeat_interleave(heads_per_group, dim=1)
 
        attn_scores = (q @ k.transpose(-2, -1)) * (self.d_head**-0.5)
        causal_mask = self._get_causal_mask(T, x.device)
        attn_scores = attn_scores.masked_fill(causal_mask, float("-inf"))
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = (attn_weights @ v).transpose(1, 2).contiguous() \
        .view(B, T, self.d_model)
        
        return self.W_o(context)
 
    def _register_mask_buffer(self, max_seq_len):
        if max_seq_len > 0:
            mask = torch.triu(torch.ones(1, 1, max_seq_len, max_seq_len,
            dtype=torch.bool), diagonal=1)
            self.register_buffer("causal_mask", mask, persistent=False)
        else:
            self.causal_mask = None
 
    def _get_causal_mask(self, seq_len, device):
        if self.causal_mask is not None and \
        self.causal_mask.size(-1) >= seq_len:
            return self.causal_mask[:, :, :seq_len, :seq_len]
        return torch.triu(torch.ones(1, 1, seq_len, seq_len, 
        dtype=torch.bool, device=device), diagonal=1)
```

---

## **2.8 Pengorbanan antara kecerdasan dan memori komputer**

Kita telah menjelajahi solusi generasi pertama terhadap masalah krisis penuhnya *KV Cache*. Akan tetapi, harus disadari bahwa baik MQA maupun GQA sebenarnya berjalan di atas satu bentuk kompromi yang sama: **kedua solusi ini menghemat memori komputer dengan cara menghapus data Key dan Value yang unik, dan sebagai gantinya mereka mengorbankan kecerdasan model AI tersebut.**

Ketegangan yang belum terselesaikan inilah yang membuat arsitektur DeepSeek menjadi sangat penting secara historis di dunia AI. Insinyur DeepSeek mengajukan pertanyaan yang sama sekali berbeda: daripada kita menghapus kepala AI dan menghilangkan kecerdasannya, bisakah kita memampatkan (meng-kompresi) informasi matematika di dalam *KV Cache* itu sendiri secara mendasar tanpa ada yang dibuang?

Perubahan pola pikir dari "menghapus kepala AI" menjadi "memampatkan data" ini mengarah langsung pada penemuan **Multi-Head Latent Attention (MLA)**, dan pada akhirnya menciptakan teknologi yang sangat luar biasa hemat bernama **DeepSeek Sparse Attention (DSA)** dan **Hybrid Attention** yang dapat kita lihat pada model versi V3.2 dan V4. Di Bab 3, kita akan membedah secara rinci bagaimana model DeepSeek berhasil mempertahankan kecerdasan penuh seperti MHA yang asli, tetapi di saat bersamaan berhasil melakukan penghematan memori yang membuat membaca teks 1 juta kata menjadi kenyataan.

## **2.9 Ringkasan**

*   Pola pembuatan kata demi kata (*autoregresif*) menyebabkan model AI naif memproses seluruh sejarah teks secara berulang dari awal setiap saat. Hal ini menciptakan kemacetan kerja komputer dengan rumus kuadrat $O(n^2)$.
*   *Key-Value (KV) Cache* memecahkan masalah ini dengan menyimpan matriks Key dan Value masa lalu di dalam memori, mengubah proses tebakan menjadi sangat efisien dan ringan secara linier $O(n)$.
*   Namun, *KV Cache* menciptakan masalah kemacetan baru berupa kepenuhan memori komputer, yang membengkak setiap kali panjang teks bertambah. Pada pembacaan teks yang super panjang, ukuran memori AI ini melampaui kemampuan kartu grafis (VRAM GPU) paling mahal sekalipun.
*   *Multi-Query Attention* (MQA) mengurangi ukuran memori *cache* dengan cara memaksa semua kepala AI berbagi satu pasang K/V saja. Kelemahannya, ini sangat merusak kemampuan AI dalam memahami bahasa ambigu yang rumit.
*   *Grouped-Query Attention* (GQA) menawarkan kompromi jalan tengah yang dapat diatur sesuka hati. Sayangnya, sistem ini pada dasarnya tetap mengorbankan tingkat kecerdasan demi menghemat memori.