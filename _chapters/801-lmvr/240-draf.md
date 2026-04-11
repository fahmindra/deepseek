---
slug: lmvr-24
title: lmvr-24
---
# Bab 24: Interpretabilitas dan Eksplanabilitas dalam LLM

> **"Kita harus ingat bahwa potensi sejati AI tidak hanya terletak pada kemampuannya untuk melakukan tugas, tetapi juga pada kemampuan kita untuk memahami dan mempercayai keputusannya." — Cynthia Rudin**

Bab 24 dari LMVR menawarkan eksplorasi komprehensif tentang interpretabilitas dan eksplanabilitas dalam model bahasa besar (LLM). Bab ini menggali perbedaan kritis antara interpretabilitas dan eksplanabilitas, serta pentingnya konsep-konsep ini dalam membangun sistem AI yang tepercaya dan etis. Bab ini mencakup berbagai teknik, mulai dari atribusi fitur dan visualisasi atensi hingga metode model-agnostik dan model-spesifik, memberikan alat praktis untuk membuat LLM lebih transparan dan mudah dipahami. Bab ini juga membahas implikasi etis dan regulasi dari penerapan model yang buram dan mengeksplorasi tren masa depan dalam eksplanabilitas, menekankan perlunya model yang tidak hanya kuat tetapi juga dapat diinterpretasikan dan dipertanggungjawabkan.

## 24.1. Pengantar Interpretabilitas dan Eksplanabilitas dalam LLM

Kemajuan pesat dalam model bahasa besar (LLM) telah menghadirkan alat-alat kuat yang mampu menghasilkan dan memproses bahasa mirip manusia dengan presisi yang luar biasa. Namun, seiring bertambahnya kompleksitas model-model ini, cara kerja internal mereka menjadi semakin buram. Interpretabilitas dan eksplanabilitas, dua area kritis dalam AI, berupaya mengatasi keburaman ini. Interpretabilitas mengacu pada upaya membuat proses internal model dapat dipahami, sementara eksplanabilitas berfokus pada penjelasan alasan di balik keputusan atau output spesifik suatu model. Bersama-sama, keduanya membentuk dasar untuk memastikan transparansi dan kepercayaan pada LLM, yang memungkinkan para pemangku kepentingan untuk memahami bagaimana dan mengapa suatu model mencapai kesimpulannya. Hal ini sangat penting dalam lingkungan berisiko tinggi seperti layanan kesehatan, keuangan, dan sistem hukum, di mana salah tafsir atau bias yang tidak terdeteksi dalam output model dapat menyebabkan konsekuensi serius.

![Gambar 1: Tantangan dalam interpretabilitas dan eksplanabilitas dalam LLM.](https://lmvr.rantai.dev//images/L6A504iArDGBvUq8WSeO-2mo4J3FvQON2hNYXV3uV-v1.png)
**Gambar 1:** Tantangan dalam interpretabilitas dan eksplanabilitas dalam LLM.

Teknik interpretabilitas dapat diklasifikasikan secara luas menjadi pendekatan global dan lokal. Interpretabilitas global bertujuan untuk membuat perilaku dan struktur keseluruhan model dapat dipahami, biasanya dengan memeriksa pola dalam parameter yang dipelajari atau arsitektur jaringan. Ini melibatkan identifikasi fitur dominan, pola berulang, dan karakteristik umum lainnya di berbagai input. Di sisi lain, interpretabilitas lokal berkaitan dengan output spesifik dan faktor-faktor yang memengaruhi prediksi atau respons individual. Teknik interpretabilitas lokal, seperti peta saliensi (saliency maps) dan visualisasi atensi, memungkinkan praktisi untuk menunjukkan apa yang memengaruhi output model tertentu. Dengan menggabungkan interpretabilitas global dan lokal, kita memperoleh pemahaman yang lebih lengkap tentang kemampuan dan keterbatasan LLM.

Mencapai interpretabilitas dan eksplanabilitas dalam LLM merupakan tantangan karena arsitektur model-model ini yang secara inheren kompleks dan berlapis-lapis. Banyak teknik interpretabilitas, seperti atribusi fitur dan perambatan relevansi tingkat lapisan (layer-wise relevance propagation), diadaptasi dari model pembelajaran mesin yang lebih sederhana seperti pohon keputusan dan jaringan saraf konvolusional. Menerapkan metode ini ke LLM, yang memproses ribuan token dan membentang di banyak lapisan, memperkenalkan tantangan unik. Teknik yang dikembangkan untuk model yang lebih sederhana seringkali kurang memiliki skalabilitas saat diterapkan pada LLM. Salah satu pendekatan matematis untuk interpretabilitas dalam LLM melibatkan pemeriksaan gradien dan bobot atensi. Dengan menganalisis gradien dari fungsi kerugian (loss function) model terhadap token inputnya, praktisi dapat menilai bagian mana dari input yang memiliki pengaruh paling signifikan terhadap prediksi.

Selain interpretabilitas, eksplanabilitas telah mendapatkan perhatian dalam beberapa tahun terakhir, dengan fokus pada penyediaan alasan yang dapat dipahami bagi para pemangku kepentingan atas keputusan model. Misalnya, dalam keuangan, keputusan model untuk menandai transaksi sebagai mencurigakan harus dapat dijelaskan kepada petugas kepatuhan. Persyaratan ini bukan hanya masalah kepercayaan pengguna tetapi juga seringkali diwajibkan oleh badan regulasi. Pertimbangan etis sangat krusial di sini: LLM yang beroperasi sebagai "kotak hitam" dapat memperkenalkan bias, yang menyebabkan perlakuan tidak adil atau diskriminasi, terutama dalam aplikasi yang melibatkan data sensitif.

Implementasi praktis dari interpretabilitas mungkin melibatkan pengembangan alat untuk memvisualisasikan lapisan atensi dalam transformer, struktur model yang umum pada LLM. Alat tersebut dapat menghasilkan representasi grafis dari head atensi, yang mengilustrasikan token mana yang memperhatikan token lainnya dan sejauh mana tingkat perhatiannya. Pola atensi dapat mengungkapkan bias, terutama ketika model secara konsisten memberikan atensi yang lebih tinggi pada pola token tertentu berdasarkan data pelatihan. Alat interpretabilitas semacam ini sangat berharga untuk mengidentifikasi potensi sumber bias dan memperbaikinya.

Berikut adalah versi Python untuk memvisualisasikan lapisan atensi menggunakan `torch` dan `matplotlib`.

### Contoh Python: Visualisasi Lapisan Atensi Transformer

```python
import torch
import matplotlib.pyplot as plt
import numpy as np

# Mendefinisikan struktur model transformer yang disederhanakan
class TransformerModel:
    def __init__(self):
        # Placeholder untuk bobot atensi
        self.attention_weights = []

    def forward(self, input_tensor):
        # Simulasi forward pass dan pengisian bobot atensi secara acak
        # Bentuk: [Batch, Heads, Seq_Len, Seq_Len]
        self.attention_weights = [torch.randn(1, 12, 64, 64).softmax(dim=-1)]
        return input_tensor

    def get_attention_weights(self):
        return self.attention_weights

# Memvisualisasikan bobot atensi menggunakan matplotlib
def visualize_attention(attention_tensor, layer_idx, output_path):
    # Mengambil head pertama dari batch pertama
    # attention_tensor shape: [1, 12, 64, 64]
    attn_data = attention_tensor[0, 0].detach().numpy()

    plt.figure(figsize=(8, 6))
    plt.imshow(attn_data, cmap='viridis', interpolation='nearest')
    plt.colorbar()
    plt.title(f"Attention Weights - Layer {layer_idx} Head 0")
    plt.xlabel("Key Tokens")
    plt.ylabel("Query Tokens")
    plt.savefig(output_path)
    plt.close()

def main():
    # Inisialisasi model dan buat input sampel
    model = TransformerModel()
    sample_input = torch.randn(1, 64)
    model.forward(sample_input)

    # Ambil bobot atensi dan visualisasikan
    attention_weights = model.get_attention_weights()
    for i, attn in enumerate(attention_weights):
        path = f"attention_layer_{i}.png"
        visualize_attention(attn, i, path)
        print(f"Menyimpan visualisasi atensi ke {path}")

if __name__ == "__main__":
    main()
```

![Gambar 2: Visualisasi sederhana dari lapisan atensi.](https://lmvr.rantai.dev//images/L6A504iArDGBvUq8WSeO-ScvYDZij1K5HkVIrakPT-v1.png)
**Gambar 2:** Visualisasi sederhana dari lapisan atensi.

Singkatnya, interpretabilitas dan eksplanabilitas adalah landasan untuk membuat LLM tepercaya dan etis. Penggunaan visualisasi, analisis gradien, dan model pengganti menciptakan pendekatan multifaset yang memungkinkan pengembang untuk memahami perilaku model yang kompleks.

## 24.2. Teknik untuk Meningkatkan Interpretabilitas dalam LLM

Seiring dengan berkembangnya kompleksitas LLM, kebutuhan akan teknik interpretabilitas yang kuat telah menjadi perhatian yang mendesak. Teknik-teknik ini memungkinkan peneliti dan pengembang untuk memahami bagaimana LLM membuat keputusan. Di antara metode yang paling banyak digunakan adalah atribusi fitur, visualisasi atensi, dan pemodelan suragat (surrogate modeling). Atribusi fitur bertujuan untuk mengidentifikasi input spesifik yang paling berkontribusi pada output model. Visualisasi atensi menyediakan cara untuk melihat di mana model fokus saat memproses bagian input yang berbeda. Sementara itu, model suragat berfungsi sebagai representasi sederhana dari model yang kompleks, mempertahankan perilaku esensial dari model asli tetapi dalam bentuk yang lebih mudah diinterpretasikan.

Metode atribusi fitur sangat mendasar untuk memahami bagian mana dari input yang paling memengaruhi keputusan model. Secara matematis, hal ini sering kali melibatkan penghitungan gradien output model terhadap inputnya. Gradien ini dapat menonjolkan input yang paling memengaruhi respons model, menunjukkan kata atau frasa mana yang paling "diperhatikan" model saat menghasilkan output. Tingkat analisis ini sangat berharga untuk men-debug dan meningkatkan LLM, terutama dalam aplikasi sensitif seperti diagnosis medis atau prakiraan keuangan.

Pemodelan suragat adalah teknik kuat yang mendekati LLM yang kompleks dengan model yang lebih sederhana, memberikan "perwakilan" yang mempertahankan sebagian besar perilaku model asli tetapi secara inheren lebih mudah untuk diinterpretasikan. Model suragat, yang sering dirancang sebagai pohon keputusan kecil atau model linier, menawarkan cara untuk mendekati batas keputusan dan pola penalaran LLM tanpa sepenuhnya mereplikasi kompleksitasnya. Model suragat dapat dilatih pada input dan output dari LLM, menangkap pola inti dalam data sambil tetap cukup transparan untuk dipahami oleh pengguna.

Meskipun teknik-interpretabilitas ini berharga, mereka datang dengan trade-off yang inheren antara kompleksitas dan pemahaman. Model dengan kompleksitas tinggi, secara alami, menangkap pola yang lebih rumit dalam data, tetapi mereka lebih sulit untuk diinterpretasikan secara langsung. Dengan menyederhanakan atau mendekati model-model ini melalui teknik seperti pemodelan suragat, kita memperoleh wawasan tetapi mungkin mengorbankan beberapa presisi atau nuansa dalam memahami kemampuan penuh model.

Kode di bawah ini memanfaatkan paralelisme untuk meningkatkan efisiensi visualisasi bobot atensi dari model transformer yang disederhanakan menggunakan Python.

### Contoh Python: Visualisasi Atensi Paralel

```python
import torch
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

class TransformerModel:
    def forward(self):
        # Simulasi bobot atensi: [Layer, Head, Seq, Seq]
        return torch.randn(1, 12, 64, 64).softmax(dim=-1)

def save_attention_plot(attn_matrix, head_idx, layer_idx):
    plt.figure(figsize=(5, 5))
    plt.imshow(attn_matrix, cmap='magma')
    plt.title(f"L{layer_idx} H{head_idx}")
    plt.axis('off')
    path = f"attention_L{layer_idx}_H{head_idx}.png"
    plt.savefig(path)
    plt.close()
    return path

def main():
    model = TransformerModel()
    # Asumsikan 1 lapisan dengan 12 heads
    attention_weights = model.forward()[0] 
    
    print("Memulai visualisasi paralel...")
    with ThreadPoolExecutor() as executor:
        futures = []
        for h_idx in range(attention_weights.size(0)):
            attn_head = attention_weights[h_idx].detach().numpy()
            futures.append(executor.submit(save_attention_plot, attn_head, h_idx, 0))
        
        for future in futures:
            print(f"Berhasil menyimpan: {future.result()}")

if __name__ == "__main__":
    main()
```

Dalam praktiknya, teknik-teknik ini diterapkan di berbagai industri. Di sektor keuangan, atribusi fitur digunakan untuk menganalisis prakiraan yang dihasilkan LLM, membantu para ahli memahami indikator ekonomi mana yang diprioritaskan model. Di layanan kesehatan, model suragat memberikan wawasan tentang faktor-faktor yang memengaruhi rekomendasi model untuk pilihan pengobatan.

## 24.3. Teknik Eksplanabilitas untuk Output LLM

Eksplanabilitas berfokus pada upaya membuat prediksi atau respons model secara individual dapat dimengerti. Teknik seperti SHAP (*SHapley Additive exPlanations*), LIME (*Local Interpretable Model-agnostic Explanations*), dan penjelasan kontrafaktual menawarkan cara untuk mengungkap misteri output ini. SHAP dan LIME menguantifikasi kontribusi fitur input terhadap output model, sementara penjelasan kontrafaktual mengeksplorasi bagaimana sedikit perubahan pada input dapat menyebabkan prediksi yang berbeda.

![Gambar 3: Metode umum untuk eksplanabilitas LLM (SHAP, LIME, dan Kontrafaktual).](https://lmvr.rantai.dev//images/L6A504iArDGBvUq8WSeO-z84R7mAz5gB66IpF3znO-v1.png)
**Gambar 3:** Metode umum untuk eksplanabilitas LLM (SHAP, LIME, dan Kontrafaktual).

SHAP, yang diturunkan dari teori permainan kooperatif, adalah salah satu teknik yang paling banyak digunakan. Nilai SHAP membantu mengidentifikasi token atau frasa input mana yang paling berpengaruh. LIME memberikan pendekatan lain dengan mendekati model kompleks secara lokal melalui model yang lebih sederhana (seperti regresi linier) di sekitar titik data tertentu. Penjelasan kontrafaktual, di sisi lain, menjawab pertanyaan: "Apa yang akan dikeluarkan model jika inputnya sedikit berbeda?" Ini sangat berharga untuk men-debug model, karena mengungkapkan sensitivitas output terhadap variasi input.

Berikut adalah logika semu (*pseudo-code*) untuk penganalisis berbasis SHAP dan LIME guna menginterpretasikan dampak kata dalam sebuah input.

### Pseudo-code: Penganalisis SHAP dan LIME

```text
// Mendefinisikan struktur penganalisis berbasis SHAP
PenganalisisSHAP:
    Metode jelaskan(input):
        token = pecah_teks(input)
        nilai_shap = []
        untuk setiap kata dalam token:
            input_terganggu = ganti_kata_dengan_mask(kata, "[MASK]")
            skor = model_inference(input_terganggu)
            simpan skor ke nilai_shap
        kembalikan nilai_shap

// Mendefinisikan struktur penganalisis berbasis LIME
PenganalisisLIME:
    Metode jelaskan(input):
        token = pecah_teks(input)
        nilai_lime = []
        untuk setiap kata dalam token:
            input_terganggu = ganti_kata_dengan_placeholder(kata, "...")
            skor = model_inference(input_terganggu)
            simpan (input_terganggu, skor) ke nilai_lime
        kembalikan nilai_lime

Fungsi Utama:
    input = "Mengapa langit berwarna biru?"
    print "Penjelasan SHAP:", PenganalisisSHAP.jelaskan(input)
    print "Penjelasan LIME:", PenganalisisLIME.jelaskan(input)
```

Dalam industri, teknik-teknik ini sangat penting bagi sektor yang menuntut akuntabilitas. Dalam perawatan kesehatan, alat berbasis LIME dapat membantu penyedia layanan memahami alasan di balik saran diagnostik tertentu. Penjelasan kontrafaktual sangat membantu dalam diagnosa pasien, di mana mengubah sedikit data pasien dan mengamati respons model memberikan wawasan tentang faktor yang paling relevan.

## 24.4. Interpretabilitas Model-Agnostik vs. Model-Spesifik

Teknik interpretabilitas model-agnostik dirancang untuk dapat diterapkan pada berbagai jenis model tanpa mempedulikan arsitektur dasarnya. Teknik ini memperlakukan model sebagai "kotak hitam". Sebaliknya, teknik interpretabilitas model-spesifik disesuaikan dengan arsitektur tertentu, seperti transformer, dengan memanfaatkan fitur struktural unik seperti mekanisme atensi.

![Gambar 4: Strategi umum untuk interpretabilitas dalam LLM.](https://lmvr.rantai.dev//images/L6A504iArDGBvUq8WSeO-fq3DFI8mR5So3ptpjNbD-v1.png)
**Gambar 4:** Strategi umum untuk interpretabilitas dalam LLM.

Metode model-agnostik memberikan fleksibilitas tinggi dan sangat menguntungkan dalam konteks regulasi di mana transparansi diperlukan di berbagai model. Namun, metode model-spesifik menawarkan wawasan yang lebih dalam. Dengan memahami aspek unik model, seperti mekanisme atensi berlapis dalam transformer, praktisi dapat melakukan analisis yang lebih bernuansa.

Berikut adalah logika semu untuk pipeline pengujian guna menerapkan teknik SHAP dan LIME ke beberapa model LLM untuk menguji konsistensi.

### Pseudo-code: Pipeline Pengujian Konsistensi

```text
PipelinePengujian:
    Metode jalankan_tes(daftar_model, teks_input):
        hasil = {}
        untuk setiap model dalam daftar_model:
            skor_shap = PenganalisisSHAP(model).jelaskan(teks_input)
            skor_lime = PenganalisisLIME(model).jelaskan(teks_input)
            hasil[model] = {"SHAP": skor_shap, "LIME": skor_lime}
        kembalikan hasil

    Metode nilai_konsistensi(hasil):
        untuk setiap pasangan model:
            bandingkan pentingnya fitur antara model A dan model B
            jika perbedaan > ambang_batas: log ketidakkonsistenan
```

## 24.5. Mengevaluasi Efektivitas Interpretabilitas dan Eksplanabilitas

Mengevaluasi efektivitas teknik ini sering kali berkisar pada metrik yang mengukur seberapa baik sebuah teknik membantu pengguna memahami dan mempercayai model. Metrik ini meliputi kejelasan (*clarity*), kelengkapan (*completeness*), dan relevansi. Hal ini juga sering melibatkan tindakan subjektif seperti kepuasan pengguna.

![Gambar 5: Proses mengevaluasi metrik interpretabilitas LLM.](https://lmvr.rantai.dev//images/L6A504iArDGBvUq8WSeO-k72h3ZsqyDNbEsv3OPB8-v1.png)
**Gambar 5:** Proses mengevaluasi metrik interpretabilitas LLM.

Salah satu tantangan utama dalam pengukuran ini adalah menyeimbangkan tingkat detail penjelasan dengan risiko membebani pengguna secara kognitif. Hal ini sering dikuantifikasi melalui skor pemahaman, yang mengukur seberapa baik pengguna memahami perilaku model berdasarkan penjelasan yang diberikan.

### Pseudo-code: Kerangka Kerja Evaluasi Konsistensi

```text
KerangkaKerjaInterpretabilitas:
    Metode cek_konsistensi(model_A, model_B, input):
        penjelasan_A = PenganalisisSHAP(model_A).jelaskan(input)
        penjelasan_B = PenganalisisSHAP(model_B).jelaskan(input)
        
        varians = hitung_varians(penjelasan_A, penjelasan_B)
        jika varians <= ambang_batas:
            kembalikan "konsisten"
        lainnya:
            kembalikan "tidak konsisten"
```

## 24.6. Pertimbangan Etis dan Regulasi

Menerapkan model "kotak hitam" tanpa mekanisme transparansi dapat menyebabkan konsekuensi parah, termasuk hilangnya kepercayaan dan bias yang tidak terkendali. Badan regulasi di seluruh dunia semakin sadar akan risiko ini, yang mendorong pengembangan undang-undang yang mewajibkan transparansi dan akuntabilitas dalam sistem AI.

![Gambar 6: Kekhawatiran etika dan kepatuhan pada LLM.](https://lmvr.rantai.dev//images/L6A504iArDGBvUq8WSeO-kPj2pl0fn4nMXPdHNgwF-v1.png)
**Gambar 6:** Kekhawatiran etika dan kepatuhan pada LLM.

Di Uni Eropa, GDPR menyertakan ketentuan "hak atas penjelasan", di mana pengguna yang terkena dampak keputusan otomatis memiliki hak untuk mengetahui alasan di balik keputusan tersebut. Akuntabilitas membutuhkan dokumentasi yang jelas dan proses yang dapat diaudit. Framework logging yang kuat dapat merekam status perantara dari teknik interpretabilitas secara waktu nyata untuk tujuan audit di masa depan.

## 24.7. Arah Masa Depan dalam Interpretabilitas dan Eksplanabilitas

Tren saat ini menunjukkan pergerakan kuat menuju pendekatan *explainability-by-design*. Dalam paradigma ini, interpretabilitas tertanam langsung ke dalam arsitektur model dan proses pelatihan, alih-alih hanya mengandalkan penjelasan pasca-kejadian (*post-hoc*).

![Gambar 7: Model yang diusulkan untuk memajukan interpretabilitas LLM.](https://lmvr.rantai.dev//images/L6A504iArDGBvUq8WSeO-0WMYFQ4a7TKje9IiMZbU-v1.png)
**Gambar 7:** Model yang diusulkan untuk memajukan interpretabilitas LLM.

Arah masa depan lainnya melibatkan persimpangan antara interpretabilitas dengan ilmu kognitif, etika, dan hukum. Ilmu kognitif dapat menawarkan wawasan tentang bagaimana manusia menafsirkan informasi secara alami, memberikan kerangka kerja untuk merancang penjelasan yang secara intuitif mudah dipahami. Teknik seperti *feature sparsity* juga dieksplorasi untuk memaksa model menggunakan lebih sedikit fitur dalam pengambilan keputusannya, sehingga outputnya lebih mudah dipahami.

## 24.8. Kesimpulan

Bab 24 menekankan peran vital interpretabilitas dan eksplanabilitas dalam memastikan bahwa model bahasa besar tidak hanya kuat tetapi juga tepercaya. Dengan memanfaatkan teknik dan alat yang telah dibahas, pengembang dapat menciptakan LLM yang transparan dan selaras dengan standar etika, membuka jalan bagi aplikasi AI yang bertanggung jawab.

### 24.8.1. Pembelajaran Lebih Lanjut dengan GenAI

- Deskripsikan perbedaan utama antara interpretabilitas dan eksplanabilitas pada LLM.
- Diskusikan tantangan dalam mencapai transparansi pada model dengan miliaran parameter.
- Jelaskan peran teknik atribusi fitur dalam memahami pengambilan keputusan model.
- Analisis trade-off antara kompleksitas model dan kemudahan interpretasi.
- Jelajahi penggunaan visualisasi atensi untuk mendeteksi pola komunikasi antar token.
- Bagaimana SHAP dan LIME diimplementasikan pada model bahasa?
- Diskusikan implikasi etis dari keputusan model yang tidak dapat dijelaskan di bidang medis.
- Analisis persyaratan regulasi seperti "hak atas penjelasan" dalam GDPR.

### 24.8.2. Latihan Praktis

#### Latihan Mandiri 24.1: Implementasi Atribusi Fitur
Tujuan: Menggunakan pustaka atribusi (seperti `captum` di Python) untuk mengidentifikasi kata-kata yang paling berpengaruh dalam klasifikasi sentimen suatu kalimat.

#### Latihan Mandiri 24.2: Visualisasi Atensi Transformer
Tujuan: Mengembangkan skrip untuk mengekstrak bobot atensi dari model BERT dan membuat heatmap interaktif menggunakan `seaborn`.

#### Latihan Mandiri 24.3: Menghasilkan Penjelasan Kontrafaktual
Tujuan: Merancang eksperimen di mana Anda mengubah satu kata dalam input dan mengamati bagaimana prediksi model berubah secara drastis.

#### Latihan Mandiri 24.4: Membangun Model Suragat Sederhana
Tujuan: Melatih pohon keputusan (decision tree) sederhana untuk meniru perilaku model klasifikasi teks yang lebih besar dan membandingkan akurasinya.

#### Latihan Mandiri 24.5: Studi Pengguna tentang Eksplanabilitas
Tujuan: Merancang survei sederhana untuk menilai apakah penjelasan yang dihasilkan model membantu orang non-teknis memahami hasil prediksi AI.