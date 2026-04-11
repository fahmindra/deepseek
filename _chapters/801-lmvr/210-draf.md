---
slug: lmvr-21
title: lmvr-21
---
# Bab 21: Prompting Few-Shot dan Zero-Shot dengan LLM

> "Pembelajaran few-shot dan zero-shot mewakili batas kemampuan AI untuk menggeneralisasi dari data minimal, mengubah cara kita berinteraksi dengan dan memanfaatkan model bahasa besar." - Geoffrey Hinton

> **Catatan Sukses:**
> Bab 21 dari LMVR mengeksplorasi implementasi dan aplikasi teknik prompting few-shot dan zero-shot. Bab ini dimulai dengan mendefinisikan metode-metode prompting ini dan membandingkannya dengan pembelajaran terawasi tradisional, menyoroti pentingnya mereka dalam tugas-tugas NLP. Bab ini kemudian masuk ke implementasi praktis, berfokus pada perancangan prompt yang efektif dan memanfaatkan alat untuk menjalankan tugas-tugas few-shot dan zero-shot. Bab ini mencakup aspek konseptual dan praktis, termasuk rekayasa prompt, pemanfaatan konteks, dan evaluasi kinerja. Bab ini bertujuan untuk memberikan panduan komprehensif untuk menerapkan teknik-teknik prompting canggih ini dalam aplikasi LLM.

## 21.1. Pengantar Prompting Zero-Shot dan Few-Shot

Prompting zero-shot dan few-shot mewakili kemajuan signifikan dalam pemrosesan bahasa alami, memberdayakan model bahasa besar (LLM) untuk menggeneralisasi di berbagai tugas dengan masukan khusus tugas yang minimal. Dengan memanfaatkan pra-pelatihan yang luas pada sumber data yang beragam, teknik prompting ini memungkinkan model untuk beradaptasi dengan tugas-tugas baru tanpa memerlukan kumpulan data berlabel yang ekstensif. Kemampuan ini menandai penyimpangan dari pembelajaran terawasi tradisional, yang sangat bergantung pada contoh berlabel untuk setiap tugas unik. Melalui penggunaan prompt terstruktur yang efektif, prompting zero-shot dan few-shot memberikan fleksibilitas dalam penanganan tugas, memungkinkan LLM beroperasi secara mulus di berbagai aplikasi seperti analisis sentimen, peringkasan, dan terjemahan dengan panduan tambahan minimal.

![Gambar 1: Ilustrasi pada prompting Zero-shot vs Few-shot.](https://lmvr.rantai.dev//images/L6A504iArDGBvUq8WSeO-0XD4udN1IcemJ3sa9S0b-v1.png)
**Gambar 1:** Ilustrasi pada prompting Zero-shot vs Few-shot.

Dalam prompting zero-shot, model beroperasi semata-mata pada deskripsi tugas yang jelas tanpa bantuan contoh khusus tugas. Misalnya, sebuah prompt mungkin mengarahkan model untuk "mengklasifikasikan sentimen dari pernyataan ini" atau "meringkas teks berikut dalam satu kalimat." Terlepas dari tidak adanya contoh, model sering kali dapat memenuhi tugas-tugas ini secara akurat dengan memanfaatkan pemahaman bahasa, konteks, dan makna yang telah dipelajari sebelumnya. Kerangka kerja autoregresif model mendasari kemampuan ini, karena ia memprediksi setiap token dalam urutan $X = (x_1, x_2, \dots, x_n)$ dengan probabilitas yang didefinisikan oleh rumus:

$$P(X) = \prod_{i=1}^n P(x_i | x_{1:i-1}),$$

di mana setiap istilah $P(x_i | x_{1:i-1})$ mewakili kemungkinan sebuah token diberikan konteks sebelumnya. Formulasi probabilistik ini memungkinkan model untuk menghasilkan output yang koheren dengan menafsirkan setiap token secara berurutan, menyelaraskan responsnya dengan deskripsi tugas yang tertanam dalam prompt. Akibatnya, prompting zero-shot dapat secara efektif memandu model melalui tugas-tugas yang memerlukan instruksi langsung, meskipun mungkin menghadapi tantangan dengan penalaran kompleks atau pengetahuan khusus yang dapat memperoleh manfaat dari contoh.

Prompting few-shot meningkatkan proses ini dengan memasukkan serangkaian contoh khusus tugas yang terbatas di dalam prompt, yang berfungsi sebagai demonstrasi dalam konteks. Misalnya, dalam tugas terjemahan bahasa, prompt mungkin menyajikan beberapa terjemahan sebelum meminta terjemahan baru dari model. Contoh-contoh ini memungkinkan model untuk mengenali dan menggeneralisasi pola yang relevan dengan tugas, mendorong proses pembelajaran dalam konteks di mana model menyesuaikan output-nya tanpa mengubah parameter internalnya. Contoh-contoh dalam prompt few-shot bertindak sebagai agen pengkondisi, membantu model memahami dan mereplikasi struktur yang diinginkan dalam responsnya. Ini sangat efektif dalam skenario di mana prompting zero-shot mungkin kurang spesifik, karena contoh few-shot memberikan konteks langsung yang memperjelas persyaratan tugas.

Kekuatan prompting few-shot terletak pada kemampuannya untuk menangani tugas-tugas yang menuntut pemahaman kontekstual atau struktur kompleks dengan menyempurnakan pemahaman model tentang pola khusus tugas. Dengan mengamati format yang konsisten dan isyarat struktural dari contoh yang diberikan, model dapat menavigasi persyaratan yang bernuansa dan menghasilkan respons yang lebih akurat. Dengan demikian, prompting few-shot menyeimbangkan singkatnya dengan kejelasan, memungkinkan model untuk menangani kueri yang kompleks secara efektif dalam batasan token. Pendekatan ini memitigasi potensi ambiguitas dalam skenario zero-shot, membuatnya sangat berguna untuk tugas-tugas yang mendapat manfaat dari contoh yang didemonstrasikan, seperti pelengkapan teks, pembuatan analogi, dan klasifikasi sentimen.

Kontras antara prompting zero-shot dan few-shot menggambarkan adaptabilitas LLM dan peran transformatif dari prompt terstruktur dalam meningkatkan kinerja model. Sementara prompting zero-shot memberikan solusi minimalis untuk tugas-tugas yang lebih sederhana, dengan mengandalkan instruksi yang jelas murni, prompting few-shot menawarkan kapasitas yang diperluas untuk menggeneralisasi tugas-tugas yang lebih canggih dengan menyertakan contoh. Kedua teknik tersebut bertumpu pada arsitektur autoregresif yang sama, memanfaatkan distribusi probabilitas untuk menghasilkan output yang koheren. Namun, konteks tambahan dalam prompt few-shot sering kali mengurangi ambiguitas, meningkatkan akurasi dalam tugas-tugas di mana pendekatan zero-shot mungkin kurang memadai.

Seiring pertumbuhan skala model yang terus berlanjut, kapasitas untuk prompting zero-shot dan few-shot telah meluas, memungkinkan LLM untuk menangani aplikasi yang semakin kompleks. Metode prompting ini mengurangi kebutuhan akan kumpulan data berlabel yang ekstensif, mempromosikan adaptasi tugas yang cepat dan menandai kemajuan kritis dalam skalabilitas dan efisiensi NLP. Dengan penyempurnaan yang berkelanjutan, teknik prompting zero-shot dan few-shot menjanjikan untuk lebih memperluas fleksibilitas LLM, memposisikan mereka sebagai alat yang fleksibel dan mudah diakses untuk beragam aplikasi dunia nyata, mulai dari dukungan pelanggan hingga pembuatan konten dan lingkungan pembelajaran interaktif.

Berikut adalah kode Python yang mendemonstrasikan cara melakukan prompting zero-shot dan few-shot menggunakan pustaka `openai`.

### Contoh Python: Prompting Zero-Shot dan Few-Shot

```python
import openai
import asyncio

# Fungsi untuk simulasi prompting zero-shot
async def run_zero_shot(client, text):
    prompt = f"Summarize the following text in one sentence.\n\nText: {text}"
    response = await client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# Fungsi untuk simulasi prompting few-shot
async def run_few_shot(client, text):
    prompt = f"""Below are some examples of text summarization:

Text: The Rust programming language offers safety and speed.
Summary: Rust provides both safety and efficiency.

Text: Large language models are transforming NLP.
Summary: LLMs revolutionize natural language processing.

Text: {text}
Summary:"""
    
    response = await client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

async def main():
    # Inisialisasi klien OpenAI (pastikan API key sudah diatur)
    client = openai.AsyncOpenAI(api_key="YOUR_API_KEY")
    
    text_to_summarize = "This book introduces readers to advanced machine learning topics."

    print("--- Menjalankan Zero-Shot ---")
    zero_result = await run_zero_shot(client, text_to_summarize)
    print(f"Hasil Zero-Shot: {zero_result}")

    print("\n--- Menjalankan Few-Shot ---")
    few_result = await run_few_shot(client, text_to_summarize)
    print(f"Hasil Few-Shot: {few_result}")

if __name__ == "__main__":
    # asyncio.run(main())
    pass
```

Prompting few-shot dan zero-shot juga menghadapi keterbatasan, terutama ketika mencoba melakukan tugas yang memerlukan pengetahuan domain yang luas atau penalaran khusus. Tantangan utamanya adalah metode ini sangat bergantung pada pengetahuan pra-terlatih yang dikodekan dalam model, yang mungkin tidak sepenuhnya mencakup tugas-tugas khusus. Prompting few-shot mungkin tidak memadai jika contoh khusus tugas tidak mencakup nuansa yang diperlukan, sementara prompting zero-shot sepenuhnya bergantung pada kemampuan model untuk menggeneralisasi berdasarkan deskripsi tugas. Akibatnya, efektivitas teknik ini bervariasi di berbagai aplikasi.

Tolok ukur kinerja (*benchmarking*) sangat penting untuk mengevaluasi efektivitas teknik prompting ini. Metrik kuantitatif, seperti akurasi, skor F1, atau skor BLEU, dapat membantu mengukur kualitas prompt di seluruh tugas, sementara penilaian kualitatif berguna untuk mengevaluasi aspek yang lebih subjektif seperti koherensi dan relevansi. Melalui tolok ukur iteratif, pengembang dapat menyesuaikan struktur prompt, contoh, atau instruksi tugas untuk menyempurnakan respons model, memastikan bahwa pengaturan prompting few-shot dan zero-shot menghasilkan hasil yang optimal.

Dalam industri, prompting few-shot dan zero-shot semakin banyak digunakan dalam otomatisasi layanan pelanggan. Strategi prompting ini menghemat waktu dan sumber daya, karena mereka menghilangkan kebutuhan akan kurasi kumpulan data yang padat karya dan pelatihan ulang model yang mahal, memungkinkan perusahaan untuk menyebarkan sistem yang mudah beradaptasi yang merespons berbagai masukan pengguna.

## 21.2. Mengimplementasikan Prompting Zero-Shot

Prompting zero-shot mewakili pendekatan transformatif dalam pemrosesan bahasa alami, di mana model melakukan tugas tanpa contoh eksplisit atau pelatihan khusus tugas. Dalam teknik ini, LLM memanfaatkan pengetahuan pra-terlatihnya yang luas untuk menafsirkan dan mengeksekusi tugas murni berdasarkan instruksi bahasa alami yang diberikan dalam prompt. Teknik ini sangat berharga ketika data berlabel untuk tugas tertentu terbatas atau tidak ada, karena model menggeneralisasi responsnya berdasarkan pola yang telah dipelajarinya selama pra-pelatihan. Berbeda dengan prompting few-shot, prompting zero-shot bergantung sepenuhnya pada kemampuan model untuk mentransfer dan menerapkan pengetahuan sebelumnya dalam konteks yang tidak dikenal.

Prompting zero-shot didasarkan pada prinsip-prinsip pembelajaran transfer, di mana model yang dilatih pada satu set tugas mentransfer pengetahuannya ke tugas-tugas baru. Secara matematis, prompting zero-shot dapat direpresentasikan sebagai fungsi $f(P) \rightarrow O$, di mana $P$ adalah deskripsi tugas dalam prompt dan $O$ adalah output model. Model mengandalkan pemahaman bahasanya yang digeneralisasi untuk memetakan $P$ ke $O$, mencapai bentuk generalisasi implisit.

Prompting zero-shot dapat diterapkan pada tugas generatif dan prediktif. Dalam tugas generatif, seperti peringkasan, prompt berfungsi sebagai instruksi. Tugas prediktif, seperti klasifikasi teks, juga mendapat manfaat dari prompting zero-shot. Dengan membingkai prompt secara tepat, pengembang dapat memandu model untuk melakukan berbagai tugas NLP bahkan ketika data pelatihan eksplisit tidak tersedia.

Berikut adalah kode Python untuk melakukan tugas peringkasan teks zero-shot.

### Contoh Python: Peringkasan Zero-Shot

```python
import openai

def zero_shot_summary(api_key, text):
    client = openai.OpenAI(api_key=api_key)
    
    prompt = f"Summarize the following text in one sentence:\n\nText: {text}"
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

def main():
    api_key = "YOUR_API_KEY"
    text = "Rust is a systems programming language focused on safety, speed, and concurrency."
    
    result = zero_shot_summary(api_key, text)
    print(f"Hasil Ringkasan Zero-Shot: {result}")

if __name__ == "__main__":
    # main()
    pass
```

Aplikabilitas prompting zero-shot meluas ke berbagai tugas NLP, mulai dari analisis sentimen hingga menjawab pertanyaan. Menilai kinerja prompt zero-shot memerlukan campuran metode evaluasi kuantitatif dan kualitatif. Metrik kuantitatif seperti akurasi memberikan dasar untuk mengevaluasi efektivitas prompt, sementara evaluasi kualitatif membantu mengukur aspek-aspek seperti koherensi dan relevansi.

Terlepas dari keuntungannya, prompting zero-shot juga menghadapi keterbatasan. Respons model dapat bervariasi secara signifikan tergantung pada pilihan kata dalam prompt. Dalam tugas-tugas yang kompleks, prompt yang ambigu dapat menghasilkan output yang salah. Selain itu, prompting zero-shot mungkin kesulitan dalam domain khusus di mana pengetahuan pra-terlatih model terbatas.

## 21.3. Mengimplementasikan Prompting Few-Shot

Prompting few-shot adalah teknik ampuh di mana model diberikan serangkaian contoh khusus tugas yang terbatas untuk memandu perilakunya dalam menghasilkan atau mengklasifikasikan teks. Berbeda dengan pembelajaran terawasi tradisional, prompting few-shot memungkinkan model untuk menyimpulkan struktur dan persyaratan tugas hanya dengan segelintir contoh ilustratif. Dengan memberikan contoh-contoh ini di dalam prompt, model mempelajari aturan implisit tugas tersebut.

![Gambar 2: Konsep kunci dari prompting few-shot.](https://lmvr.rantai.dev//images/L6A504iArDGBvUq8WSeO-mSJ1Fpzye0EaASFmYecP-v1.jpeg)
**Gambar 2:** Konsep kunci dari prompting few-shot.

Merancang prompt yang efektif untuk pembelajaran few-shot sangat penting untuk memaksimalkan kinerja model. Prompt few-shot yang optimal menyeimbangkan konteks, kejelasan, dan singkatnya. Secara matematis, prompt few-shot dapat direpresentasikan sebagai urutan pasangan contoh $(x_1, y_1), (x_2, y_2), \ldots, (x_n, y_n)$ diikuti oleh input kueri $x_{n+1}$. Model ditugaskan untuk memprediksi $y_{n+1}$ berdasarkan pola yang diamati pada pasangan sebelumnya.

Dalam menyusun prompt few-shot, aspek kuncinya adalah rekayasa prompt, yaitu seni menyusun urutan input untuk memancing kinerja model yang optimal. Rekayasa prompt yang efektif untuk skenario few-shot melibatkan teknik-teknik seperti mengurutkan contoh untuk mencerminkan kasus yang bervariasi namun relevan, menggunakan petunjuk kontekstual dalam setiap contoh, dan memastikan batas yang jelas antara contoh dan input kueri.

Berikut adalah implementasi Python untuk pendekatan prompting few-shot guna klasifikasi sentimen.

### Contoh Python: Klasifikasi Sentimen Few-Shot

```python
import openai

def few_shot_sentiment(api_key, new_sentence):
    client = openai.OpenAI(api_key=api_key)
    
    # Memberikan contoh di dalam prompt
    prompt = f"""Classify the sentiment of each sentence as Positive or Negative:

Sentence: I love the new Rust features! Sentiment: Positive
Sentence: The program keeps crashing unexpectedly. Sentiment: Negative
Sentence: I am very satisfied with the model's performance. Sentiment: Positive
Sentence: {new_sentence} Sentiment:"""
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

def main():
    api_key = "YOUR_API_KEY"
    new_input = "The new update has several bugs."
    
    sentiment = few_shot_sentiment(api_key, new_input)
    print(f"Hasil Klasifikasi Sentimen: {sentiment}")

if __name__ == "__main__":
    # main()
    pass
```

Prompting few-shot memiliki aplikasi yang beragam, mulai dari klasifikasi teks hingga pembuatan konten. Untuk mengevaluasi efektivitasnya, beberapa metrik dapat digunakan, seperti akurasi, skor F1, dan recall. Evaluasi kualitatif juga esensial, terutama dalam tugas-tugas terbuka di mana metrik saja mungkin tidak menangkap kehalusan kualitas output.

Namun, prompting few-shot memang menghadapi keterbatasan, terutama bila diterapkan pada tugas-tugas yang memerlukan pemahaman kontekstual yang sangat luas. Ketergantungan pada contoh dapat membuat output sangat sensitif terhadap urutan dan pilihan kata dari contoh tersebut.

## 21.4. Teknik Lanjutan dan Kustomisasi

Seiring dengan matangnya rekayasa prompt, menggabungkan teknik prompting few-shot dan zero-shot telah muncul sebagai pendekatan yang kuat untuk menangani tugas-tugas yang kompleks. Prompting few-shot memberikan contoh bagi model untuk menyimpulkan pola tugas, sementara prompting zero-shot mengandalkan deskripsi instruksional. Dengan menggabungkan pendekatan ini, kita dapat memandu model secara berurutan, menyiapkan prompt zero-shot awal untuk menetapkan pemahaman tugas dan menggunakan contoh few-shot untuk memperkuat serta menyempurnakan output model.

Secara matematis, urutan tugas dapat direpresentasikan sebagai $S = (P_1, P_2, \dots, P_n)$, di mana setiap $P_i$ adalah prompt yang secara bertahap membangun tujuan tugas. Komponen few-shot berkontribusi pada pembelajaran berbasis contoh, direpresentasikan sebagai istilah probabilitas bersyarat $P(Y | X, E)$, di mana $E$ adalah contoh yang diberikan. Komponen zero-shot mengandalkan probabilitas generalisasi $P(Y | X, I)$, dengan $I$ sebagai prompt instruksional.

Kustomisasi memungkinkan pendekatan modular untuk membangun strategi prompting hibrida ini. Pengembang dapat mengatur prompt yang dimulai dengan struktur zero-shot, diikuti oleh contoh few-shot berantai untuk memperkuat akurasi. Tingkat kustomisasi ini sangat penting dalam aplikasi di mana presisi adalah yang utama, seperti dalam ringkasan medis atau analisis dokumen hukum.

Berikut adalah kode Python yang mendemonstrasikan rantai prompting hibrida untuk tugas peringkasan dua langkah.

### Contoh Python: Prompting Hibrida (Zero-Shot diikuti Few-Shot)

```python
import openai

class HybridSummarizer:
    def __init__(self, api_key):
        self.client = openai.OpenAI(api_key=api_key)

    def run_step(self, prompt):
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

    def generate_hybrid_summary(self, report_text):
        # Langkah 1: Ringkasan Zero-Shot awal
        zs_prompt = f"Summarize the following report in one concise sentence:\n\nReport: {report_text}"
        initial_summary = self.run_step(zs_prompt)
        print(f"Ringkasan Awal (Zero-Shot): {initial_summary}")

        # Langkah 2: Penyaluran hasil ke Prompt Few-Shot untuk perbaikan struktur
        fs_prompt = f"""Refine the summary to include key components like safety, performance, and concurrency:

Example Summary 1: Rust offers memory safety and performance for systems programming.
Example Summary 2: Rust's concurrency model ensures safe, parallel execution.

Initial Summary to refine: {initial_summary}
Refined Summary:"""
        
        final_summary = self.run_step(fs_prompt)
        return final_summary

def main():
    api_key = "YOUR_API_KEY"
    report = "Rust is a language that prioritizes safety, performance, and concurrency."
    
    summarizer = HybridSummarizer(api_key)
    result = summarizer.generate_hybrid_summary(report)
    print(f"Ringkasan Akhir yang Diperhalus: {result}")

if __name__ == "__main__":
    # main()
    pass
```

Strategi prompting tingkat lanjut sering melibatkan interaksi kompleks dengan model. Sifat modular dari desain prompt memudahkan implementasi alur kerja terstruktur ini, memungkinkan prompt disesuaikan secara dinamis berdasarkan respons sebelumnya. Pertimbangan etika sangat penting dalam desain teknik prompting tingkat lanjut ini, karena prompt yang kompleks secara tidak sengaja dapat memperkenalkan bias.

Tren yang muncul dalam rekayasa prompt tingkat lanjut mencakup pembuatan prompt dinamis dan prompting adaptif, di mana struktur serta isi prompt disesuaikan berdasarkan masukan pengguna atau umpan balik sistem secara waktu nyata. Dalam prompting adaptif, algoritma pembelajaran penguatan dapat digunakan untuk mengoptimalkan urutan prompt, memberikan penghargaan kepada model untuk output yang selaras dengan hasil yang diinginkan.

## 21.5. Kesimpulan

Sebagai kesimpulan, Bab 21 menawarkan eksplorasi menyeluruh tentang prompting few-shot dan zero-shot dengan LLM. Dengan menguasai teknik-teknik ini, pembaca dapat memanfaatkan kekuatan LLM untuk melakukan tugas-tugas kompleks dengan data minimal, membuka jalan bagi aplikasi AI yang lebih fleksibel dan efisien.

### 21.5.1. Pembelajaran Lebih Lanjut dengan GenAI

- Rinci perbedaan teoretis antara prompting few-shot dan zero-shot pada LLM.
- Analisis peran pembelajaran mandiri (*self-supervised learning*) dalam efektivitas teknik ini.
- Berikan gambaran umum tentang bagaimana library sistem mendukung pemrograman asinkron untuk interaksi LLM.
- Diskusikan metodologi dalam merancang prompt yang efektif untuk pembelajaran few-shot.
- Identifikasi keterbatasan utama dalam mengimplementasikan prompting few-shot.
- Periksa peran model autoregresif dalam meningkatkan kemampuan zero-shot.
- Bagaimana rekayasa prompt memengaruhi kinerja model dalam skenario tanpa data berlabel?
- Bandingkan kinerja metode few-shot dan zero-shot di berbagai tugas NLP (klasifikasi, terjemahan, dll).
- Jelajahi dampak pemanfaatan konteks dalam prompting.
- Deskripsikan metodologi untuk melakukan *benchmarking* dan mengevaluasi model few-shot dan zero-shot.

### 21.5.2. Latihan Praktis

#### Latihan Mandiri 21.1: Analisis Komparatif Metode Prompting
Tujuan: Memahami dan membedakan kinerja prompting few-shot versus zero-shot dalam berbagai tugas NLP menggunakan alat AI generatif.

#### Latihan Mandiri 21.2: Mengimplementasikan dan Mengoptimalkan Prompt
Tujuan: Mengembangkan prompt few-shot yang dioptimalkan untuk satu tugas NLP tertentu dan mengevaluasi dampaknya terhadap akurasi model.

#### Latihan Mandiri 21.3: Mengatasi Tantangan dalam Prompting Few-Shot
Tujuan: Mengidentifikasi masalah seperti konteks yang tidak memadai dan merancang strategi mitigasi untuk memperbaikinya.

#### Latihan Mandiri 21.4: Evaluasi Model Autoregresif
Tujuan: Memeriksa peran model autoregresif dalam prompting zero-shot dan menilai kemampuannya dalam menghasilkan teks yang koheren.

#### Latihan Mandiri 21.5: Benchmarking Model Few-Shot dan Zero-Shot
Tujuan: Melakukan pengujian sistematis untuk mengumpulkan data kinerja dan memahami trade-off antara kedua metode tersebut.