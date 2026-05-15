---
slug: ch-06-07
title: "Proyek Akhir"
layout: chapter
---

# Submission
**Proyek Akhir: Fine-tuned Chatbot Tim Legal berbasis RAG**

---

## Pengantar

Mari kita mulai dengan membayangkan sebuah skenario. Anda baru saja dipromosikan menjadi Lead AI Engineer di sebuah perusahaan teknologi yang sedang berkembang pesat. Sore menuju malam ini, Anda diundang ke sebuah rapat penting bersama jajaran direksi.

Suasana rapat terasa cukup tegang. Sang CEO membuka presentasinya dengan sebuah pengamatan yang sangat terang.

*"Kita semua pasti sudah sangat familier dengan layanan seperti ChatGPT atau Google Gemini,"* ujarnya sambil menatap anggota rapat. *"Pengalaman yang kita rasakan saat menggunakannya hampir terasa seperti berbicara dengan seorang ahli. Kita mengetik sebuah pertanyaan, menekan enter, lalu beberapa detik kemudian sebuah jawaban yang komprehensif muncul. Sangat mulus."*

![dos-e5c2162725d85a3b0cbbd67fad29e2db20260328171841.png](https://assets.cdn.dicoding.com/original/academy/dos-e5c2162725d85a3b0cbbd67fad29e2db20260328171841.png)

Ia menghentikan presentasinya sejenak. *"Dari sisi pengguna, proses itu terasa sangat sederhana. Tetapi saya tahu, di balik interaksi tersebut, ada sistem kompleks yang bekerja. Dan itulah persisnya teknologi yang kita butuhkan sekarang."*

Project Manager Anda kemudian mengambil alih pembicaraan dan memaparkan masalah sesungguhnya. Perusahaan saat ini tidak kekurangan informasi dan justru sebaliknya. Tim legal harus berhadapan dengan tumpukan dokumen hukum yang sangat besar, mulai dari Undang-Undang, Peraturan Pemerintah, hingga dokumen kepatuhan internal. Sebagian besar tersimpan dalam bentuk PDF dengan struktur panjang dan kompleks.

*"Kita tidak bisa membuang waktu lagi,"* kata Project Manager Anda. *“Kita juga tidak bisa mengandalkan AI publik. Selain masalah kerahasiaan dokumen hukum internal, model tersebut tidak memiliki pemahaman terhadap regulasi yang kita gunakan. Mereka bisa saja memberikan jawaban yang terdengar meyakinkan, tetapi sebenarnya tidak akurat atau bahkan menyesatkan.”*

Di sinilah tatapan mereka berdua tertuju pada Anda.

*“Kami membutuhkan sebuah solusi,”* kata sang CEO. *“Sebuah asisten AI internal yang mampu menjawab pertanyaan hukum secara cepat, tetapi tetap akurat dan dapat dipertanggungjawabkan. Sistem yang tidak berspekulasi, melainkan menjawab berdasarkan dokumen resmi yang kita miliki.”*

*Nah*, inilah waktu yang tepat untuk show-off seluruh pengetahuan mengenai LLM yang sudah Anda dapatkan sebelumnya dan menghasilkan sebuah produk nyata.

Alih-alih hanya memanggil model yang sudah tersedia secara *as is*, Anda akan bertindak layaknya Lead AI Engineer di cerita tadi. Anda akan membangun sistem tersebut dari nol dengan dua pilar utama:

*   **Menyiapkan Otak Sistem (Fine-Tuning SLM):** *Yup*, Anda akan menggunakan Small Language Model (SLM) yang akan Anda fine-tuning sendiri sebagai inti dari sistem Chatbot ini. Proses ini akan memastikan model Anda dapat melakukan tanya-jawab selayaknya asisten sungguhan.
*   **Menyuntikkan Pengetahuan (RAG):** Sekuat apa pun modelnya, ia tidak akan berguna jika tidak tahu konteks spesifik dari dokumen "perusahaan". Oleh karena itu, Anda akan menambahkan mekanisme Retrieval-Augmented Generation (RAG) dengan dokumen PDF yang disediakan.

---

## Kriteria Utama

Submission ini mengharuskan Anda untuk melalui tiga tahapan pengembangan aplikasi generative AI berbasis LLM sederhana, yaitu fine-tuning, membangun sistem RAG, dan pembuatan interface. Oleh karena itu, penting untuk memperhatikan beberapa hal berikut ini sebelum memulai pengerjaan proyek.

*   Pastikan notebook dapat dijalankan sepenuhnya tanpa *error* sebelum dikirimkan, agar seluruh proses berjalan dengan baik dan hasil dapat diverifikasi.
*   Penuhi terlebih dahulu kriteria **Basic Submission** sebelum melanjutkan ke level **Skilled** dan **Advanced**.
*   Jika mengalami keterbatasan komputasi, Anda sangat dianjurkan untuk memanfaatkan GPU *free tier* yang tersedia di Google Colab atau Kaggle.
*   Dataset yang digunakan untuk proses fine-tuning dan GRPO (jika Anda melakukannya) **WAJIB** berupa **dataset instruction (tanya–jawab)** dalam format Alpaca berbahasa Indonesia. Dataset tersebut dapat diunduh melalui tautan yang telah disediakan di Hugging Face berikut ini.
    *   [Ichsan2895/alpaca-gpt4-indonesian](https://huggingface.co/datasets/Ichsan2895/alpaca-gpt4-indonesian)
        ![dos-6fb94eff4847bd11cf7a401e5936c9a520260329024042.png](https://assets.cdn.dicoding.com/original/academy/dos-6fb94eff4847bd11cf7a401e5936c9a520260329024042.png)
*   Dokumen yang digunakan merupakan **empat file PDF berisi Undang-Undang (UU)** yang **WAJIB** digunakan seluruhnya oleh model untuk menjawab pertanyaan dari Legal Team. Anda dapat mengunduh dokumennya melalui tautan Google Drive berikut ini.
    *   [Dokumen Knowledge RAG](https://drive.google.com/drive/folders/1LHZ1IncPmmUN5kytFu3i7MoaafFrKDql?usp=sharing)
*   Model awal (pre-trained model) yang digunakan harus dipastikan dirancang untuk task **Text Generation**. Tujuan dari task ini adalah menguji kemampuan Anda dalam menyusun pipeline pelatihan.
*   Arsitektur model yang dipilih harus dipastikan didukung oleh **Unsloth**, seperti keluarga **Llama**, **Mistral**, **Qwen**, **Gemma**, atau **Phi**.
*   Model yang digunakan **wajib dari penyedia terpercaya** (contoh: Meta Llama, Qwen, Mistral) atau versi kuantisasi/optimasi resmi (contoh: model dari Unsloth).
*   Model hasil dari fine-tuning dan GRPO (jika melakukan) yang di-*push* ke repositori Hugging Face wajib disertakan tautannya pada sebuah file .txt (link_huggingface.txt). 

### Kriteria 1: Fine-tuning LLM Anda Sendiri

*   Menggunakan fungsi *mapping* dari Hugging Face datasets untuk mengubah dataset mentah mereka ke dalam standar Chat Template yang didukung Unsloth. Misalnya, menggunakan format Llama-3 atau ChatML. Pastikan Anda menampilkan *output print* dari satu baris dataset yang sudah terformat lengkap dengan token spesial. 
    *   Contoh dataset yang BELUM di-*mapping*.
        ![dos-64191272a2808e8c1cab16f1bdf05f9020260312172222.jpeg](https://assets.cdn.dicoding.com/original/academy/dos-64191272a2808e8c1cab16f1bdf05f9020260312172222.jpeg) 
    *   Contoh dataset yang SUDAH di-*mapping*.
        ![dos-9ab785c38dd60d99e3bef3ac3a1d148a20260312172233.jpeg](https://assets.cdn.dicoding.com/original/academy/dos-9ab785c38dd60d99e3bef3ac3a1d148a20260312172233.jpeg)
*   Memanfaatkan teknik QLoRA saat memuat model dengan mengaktifkan *double quantization* dalam format 4-bit. Siswa wajib mendefinisikan Adapter LoRA pada setidaknya salah satu komponen komputasi utama model.
    *   Multi-Head Attention
    *   Feed Forward Network
*   Menjalankan SFTTrainer pada model text generation, minimal selama 800 steps hingga selesai tanpa terjadi error VRAM (OOM).
*   Mengunggah model hasil Fine-tuning Anda ke repositori Hugging Face agar dapat dipanggil kembali pada tahap inferensi RAG. Gunakan perintah `model.push_to_hub` dengan metode `merged_16bit`.

**Reject (0 pts)**

*   Tidak melakukan mapping pada dataset menggunakan Hugging Face datasets.
*   Tidak menggunakan teknik QLoRA untuk mengefisiensi proses fine-tuning dan justru melakukan full fine-tuning. Selain itu, tidak menempatkan LoRA pada setidaknya keseluruhan satu komponen komutasi utama.  
*   Tidak menjalankan SFTTrainer atau SFTTrainer dijalankan kurang dari 800 steps atau terjadi eror VRAM (OOM)
*   Tidak mengunggah model yang telah dilatih sebelumnya ke repositori Hugging Face.

**Basic (2 pts)**

*   Menggunakan fungsi *mapping* dari Hugging Face datasets untuk mengubah dataset mentah mereka ke dalam standar Chat Template yang didukung Unsloth. Misalnya menggunakan format Llama-3 atau ChatML. Pastikan Anda menampilkan output print dari satu baris dataset yang sudah terformat lengkap dengan token spesial. 
    *   Contoh dataset yang BELUM di-*mapping*.
        ![dos-f576c0df9bff58e8e7577b5675704ccf20260312172313.jpeg](https://assets.cdn.dicoding.com/original/academy/dos-f576c0df9bff58e8e7577b5675704ccf20260312172313.jpeg)
    *   Contoh dataset yang SUDAH di-*mapping*.
        ![dos-f5fc86b4e29b504d78e12a529c49589520260312172326.jpeg](https://assets.cdn.dicoding.com/original/academy/dos-f5fc86b4e29b504d78e12a529c49589520260312172326.jpeg)
*   Memanfaatkan teknik QLoRA saat memuat model dengan mengaktifkan *double quantization* dalam format 4-bit. Siswa wajib mendefinisikan Adapter LoRA pada setidaknya salah satu komponen komputasi utama model.
    *   Multi-Head Attention
    *   Feed Forward Network
*   Menjalankan SFTTrainer pada model text generation, minimal selama 800 steps hingga selesai tanpa terjadi error VRAM (OOM).
*   Mengunggah model hasil Fine-tuning Anda ke repositori Hugging Face agar dapat dipanggil kembali pada tahap inferensi RAG. Gunakan perintah `model.push_to_hub` dengan metode `merged_16bit`.

**Skilled (3 pts)**

*   Semua ketentuan Basic terpenuhi.
*   Membagi dataset menjadi train dan validation dan memodifikasi TrainingArguments pada Unsloth untuk memasukkan eval_dataset, mengaktifkan `evaluation_strategy = "steps"`, dan mengatur logging.
*   Melakukan setidaknya dua kali percobaan atau eksperimen training dengan kombinasi hyperparameter yang berbeda untuk mencari tahu kombinasi hyperparameter yang menghasilkan kurva loss terbaik tanpa overfitting.

**Advanced (4 pts)**

*   Semua ketentuan Skilled terpenuhi.
*   Anda akan melakukan GRPO menggunakan GRPOTrainer dari TRL dan Unsloth pada model yang telah di fine-tuning sebelumnya. Oleh karena itu, Anda perlu memuat kembali model instruct buatan versi Anda.
*   Saat melakukan GRPO, Anda haru membuat beberapa Reward Model dengan deskripsi berikut.
    *   `format_reward_func`: Menggunakan strategi Reward Shaping untuk memberikan poin bertahap (maksimal **+1.0**) guna memudahkan model belajar format yang benar.
        *   Jika model berhasil membuka responsnya dengan tag `<think>`: Poin **+0.2**
        *   Jika model berhasil menutup pemikirannya dengan tag `</think>`: Poin **+0.3**
        *   Jika formatnya sempurna (berada di awal kalimat, ditutup dengan benar, dan diikuti oleh jawaban akhir): Poin **+1.0**
        *   **Penalti:** Memberi pengurangan Poin **-0.5** jika model berhalusinasi dengan memunculkan tag `<think>` atau `</think>` lebih dari satu kali.
    *   `reasoning_length_reward`: Anda wajib membuat fungsi yang memberikan poin proporsional berdasarkan jumlah karakter teks yang berada di dalam tag `<think> ... </think>`. Ini bertujuan memaksa model untuk benar-benar menjabarkan proses berpikirnya, bukan sekadar menulis tag kosong. Fungsi ini juga harus toleran jika proses berpikir model terpotong oleh batas maksimal token. Berikut ketentuan Poinnya:
        *   Jika **TIDAK** ada tag `<think>`, atau tag ada tetapi isinya kosong/hanya spasi: Poin **+0.0**
        *   Jika panjang isi teks `<think>` **kurang dari 50** karakter (terlalu singkat): Poin **+0.2**
        *   Jika panjang isi teks `<think>` antara **50 hingga 199** karakter: Poin **+0.5**
        *   Jika panjang isi teks `<think>` **200** karakter **atau lebih** (penalaran detail): Poin **+1.0**
    *   `correctness_reward`: Memberi poin **+1.0** jika output akhir model (setelah proses berpikir) mengandung ground truth dari dataset Anda, atau memiliki tingkat kemiripan teks (misalnya menggunakan metrik ROUGE/BLEU) dengan jawaban pada kolom Output.
    *   `language_reward_func`: Memberi penalti (poin **-0.5**) jika model tiba-tiba menjawab menggunakan bahasa Inggris, dan memberi poin **+1.0** jika output akhirnya murni menggunakan tata bahasa Indonesia.
*   Anda sangat diarahkan untuk mengatur parameter seperti `num_generations` dan `max_completion_length` untuk memitigasi terjadinya *Out of Memory* (OOM).
*   Menguji model hasil GRPO ke dalam pipeline RAG. Model harus menunjukkan proses berpikirnya sebelum memberikan jawaban final berdasarkan dokumen yang diambil. Test Case Wajib:
    *   Prompt: *"Saya staf admin, kemarin lembur 3 jam untuk beresin laporan. Apakah saya berhak dapat uang lembur?"*
    *   Output model harus menampilkan proses reasoning yang ditandai dengan token spesial `<think>`. Contohnya: *“<think> User adalah staf admin (pekerja non-manajerial). Berdasarkan PP 35/2021, pekerja yang bekerja melebihi waktu kerja standar wajib dibayar upah lembur. 3 jam lembur harus dibayar.</think> Ya, Anda berhak. Berdasarkan PP No. 35 Tahun 2021, perusahaan wajib membayar upah lembur untuk staf admin yang bekerja melebihi waktu kerja normal (8 jam sehari).”*

### Kriteria 2: Membangun Sistem RAG

*   Memuat dokumen PDF dan melakukan pemotongan teks menggunakan text splitter dengan ukuran chunk dan overlap yang ditentukan secara eksplisit.
*   Menggunakan model embedding open-source untuk mengubah chunk teks menjadi vektor, dan menyimpannya ke dalam Vector Database lokal (ChromaDB atau FAISS).
*   Memuat model hasil fine-tuning, merangkai prompt yang berisi `{context}` dan `{question}`, lalu melakukan inference atau generation.
*   Membungkus pipeline RAG mereka ke dalam sebuah *interface* sederhana. Anda bisa memilih salah satu dari dua cara termudah berikut ini.
    *   Menggunakan Interactive Python Loop dengan fungsi `input()` dan mencetak hasilnya menggunakan `IPython.display.Markdown` agar rapi.
    *   Menggunakan Gradio `gr.Interface` dasar (hanya satu kotak teks input dan satu kotak teks output).

**Reject (0 pts)**

*   Tidak memuat dokumen PDF dan menggunakan ukuran *chunk* yang sangat besar (>5000) tanpa overlap yang ditentukan secara eksplisit.
*   Menggunakan model embedding proprietary misal dari OpenAI untuk mengubah chunk teks menjadi vektor dan menyimpannya ke dalam Vector Database lokal (ChromaDB atau FAISS).
*   Memuat model baru dari Hugging Face atau menggunakan model Proprietary (bukan model hasil dari kriteria 1) untuk melakukan inference atau generation.

**Basic (2 pts)**

*   Memuat dokumen PDF dan melakukan pemotongan teks menggunakan text splitter dengan ukuran chunk dan overlap yang ditentukan secara eksplisit.
*   Menggunakan model embedding open-source untuk mengubah *chunk* teks menjadi vektor, dan menyimpannya ke dalam Vector Database lokal (ChromaDB atau FAISS).
*   Memuat model hasil fine-tuning, merangkai prompt yang berisi `{context}` dan `{question}`, lalu melakukan inference atau generation.
*   membungkus pipeline RAG mereka ke dalam sebuah antarmuka sederhana. Anda bisa memilih salah satu dari dua cara termudah berikut ini.
    *   Menggunakan Interactive Python Loop dengan fungsi `input()` dan mencetak hasilnya menggunakan `IPython.display.Markdown` agar rapi.
    *   Menggunakan Gradio `gr.Interface` dasar (hanya satu kotak teks input dan satu kotak teks output).
    
**Skilled (3 pts)**

*   Semua ketentuan Basic terpenuhi.
*   Menambahkan **metadata enrichment** terhadap dokumen yang digunakan.
*   Membuat **metadata filtering** serta memberikan sitasi terhadap jawaban yang dihasilkan oleh AI.
*   Membangun **Ensemble Retriever** yang menggabungkan pencarian berbasis keyword (BM25) dengan pencarian semantik (FAISS/ChromaDB). Anda juga perlu menentukan bobot untuk setiap retriever yang ingin digabungkan. Ambil atau retrieve setidaknya 5 document untuk tahap selanjutnya.
*   Membangun **Parent-Child Retriever** dengan memisahkan dokumen menjadi Child Chunks (potongan kecil untuk pencarian vektor) dan Parent Chunks (potongan besar/halaman utuh untuk konteks LLM).

**Advanced (4 pts)**

*   Semua ketentuan Skilled terpenuhi.
*   Mengimplementasikan **HyDE (Hypothetical Document Embeddings)** untuk melakukan transformasi pada Query dengan jawaban-jawaban halusinasi awal yang dibuat oleh LLM.
*   Menggunakan model **Reranker** untuk mengurutkan ulang dan hanya mengambil Top-K (misal 3 chunk) yang paling relevan.
*   **Mengekstrak Relevance Score** dari hasil output model Cross-Encoder (Reranker) pada dokumen urutan pertama (Top-1). Siswa kemudian membuat aturan kondisional (if-else). Jika skor Reranker Top-1 berada di bawah threshold yang ditentukan, sistem mengabaikan dokumen lokal dan beralih memanggil fungsi **DuckDuckGo Search** untuk mencari informasi dari internet.

---

## Ketentuan Penilaian

**Perhitungan Nilai**

Nilai akhir yang Anda dapatkan diperoleh melalui perhitungan formula berikut.

| Nilai Akhir = Total Points / Jumlah Kriteria |
| :--- |

**Catatan**:
Perhitungan nilai akhir di atas digunakan apabila setiap kriteria mendapatkan nilai 2 pts atau tidak ada kriteria yang ditolak.

**Tabel Penilaian:** 
Adapun untuk penilaian submission dapat dilihat pada tabel berikut.

| **Ketentuan Penilaian** | | | | | |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **Nilai Akhir** | **Nilai Dicoding** | **Nilai Huruf** | **Level of Mastery** | **Makna Nilai** | **Keterangan** |
| < 1 | Rejected | E | - | Tidak Lulus | Anda sudah mencoba, tetapi belum memenuhi kompetensi minimal. |
| 1 - < 2 | Bintang 2 | D | Bellow Basic | Kurang | Anda sudah memenuhi semua kompetensi minimal, tetapi terdapat area yang masih bisa ditingkatkan. |
| 2 - < 3 | Bintang 3 | C | Basic | Cukup | Anda sudah memenuhi semua kompetensi minimal dari learning objective. |
| 3 - < 4 | Bintang 4 | B | Skilled | Mahir | Anda sudah memenuhi semua kompetensi dengan baik atau mahir. |
| 4 | Bintang 5 | A | Advanced | Tingkat Lanjut | Anda sudah memenuhi semua kompetensi dengan sangat baik atau tingkat lanjut. |

---

## Ketentuan Berkas

Beberapa poin ini perlu diperhatikan ketika mengirimkan berkas submission salah satunya menggunakan bahasa pemrograman Python. Lalu dokumen yang dikumpulkan adalah sebagai berikut.

*   Kumpulkan seluruh pekerjaan Anda dalam 1 folder dan kirim dalam bentuk yang telah di-zip.
*   File .ipynb yang dikirim telah dijalankan terlebih dahulu sehingga output ada tanpa reviewer perlu menjalankan ulang notebook.
*   Salin tautan repositori Hugging Face dari model hasil fine-tuning Anda, kemudian simpan dalam sebuah file berformat `.txt`.

    ```text
    PGABL_Nama-siswa.zip
    ├── Fine-tuning_submission_PGABL_Nama-siswa.ipynb
    ├── GRPO_submission_PGABL_Nama-siswa.ipynb (Opsional)
    ├── RAG_submission_PGABL_Nama-siswa.ipynb
    ├── link_huggingface.txt
    ├── requirements.txt
    ```

---

## Tips & Trick

1.  Bagi yang belum mengetahui cara mendapatkan token WandB untuk proses logger
    *   Kunjungi tautan Weight & Biases.
        ![dos-2cff2d0e8d04a1f491342a9d1ef092fd20260312173058.jpeg](https://assets.cdn.dicoding.com/original/academy/dos-2cff2d0e8d04a1f491342a9d1ef092fd20260312173058.jpeg)
    *   Setelah itu, lakukan *sign up* atau *sign in* dengan akun Google milik Anda atau dengan akun lainnya sesuai dengan preferensi.
        ![dos-d4aa8e805e540e0f49750eec8738606d20260312173114.jpeg](https://assets.cdn.dicoding.com/original/academy/dos-d4aa8e805e540e0f49750eec8738606d20260312173114.jpeg) 
    *   Setelah semua tahapan pendaftaran selesai, Anda tinggal pergi ke menu API Keys seperti berikut ini.
        ![dos-8d1f40178be95b11f27169b447573b4d20260312172047.jpeg](https://assets.cdn.dicoding.com/original/academy/dos-8d1f40178be95b11f27169b447573b4d20260312172047.jpeg)
    *   *Nah*, setelah masuk, Anda akan diarahkan ke tampilan berikut ini.
        ![dos-6d55833b2d9a3223f268c3be171a6fd920260312173141.jpeg](https://assets.cdn.dicoding.com/original/academy/dos-6d55833b2d9a3223f268c3be171a6fd920260312173141.jpeg) 
    *   Di sana, Anda dapat membuat token atau API keys sendiri.
2.  Hindari memuat ulang model berulang kali saat melakukan eksperimen pada satu Notebook yang sama. Jika terpaksa untuk mengganti model bahasa yang digunakan, sangat disarankan untuk me-restart runtime agar memory tidak terbuang sia-sia karena menyimpan model yang lalu.
3.  Untuk membuat file requirements.txt terdapat beberapa cara salah satunya menggunakan pip freeze atau pipreqs. Berikut cara penggunaan dan perbedaannya.
    *   **pip freeze**
        pip freeze menghasilkan daftar semua library Python yang diinstal di lingkungan saat ini beserta versinya.
        ```bash
        pip freeze > requirements.txt
        ```
    *   **pipreqs**
        pipreqs menghasilkan file requirements.txt yang hanya mencantumkan library yang digunakan dalam proyek berdasarkan impor yang ada dalam file kode.
        ```bash
        pipreqs /path/to/your/project
        ```
        Tentunya kedua cara tersebut memiliki kelebihan dan kekurangan. Untuk mengetahui lebih lengkap terkait freeze dan pipreqs, Anda dapat membaca di tautan berikut: [<u>Ternyata Mengelola Dependensi Proyek Python Semudah Ini, lo!.</u>](https://www.dicoding.com/blog/ternyata-mengelola-dependensi-proyek-python-semudah-ini-lo/)
4.  Untuk export project yang Anda kerjakan di Colaboratory sebagai berkas ipynb, klik tombol file yang berada di pojok kiri atas Colaboratory dan pilih download .ipynb.
    ![dos-aa370b7c0891ef52727044f3e7ebd15520260312173327.jpeg](https://assets.cdn.dicoding.com/original/academy/dos-aa370b7c0891ef52727044f3e7ebd15520260312173327.jpeg)

---

## Lainnya

### Submission yang Tidak Sesuai Kriteria

Jika **tidak sesuai dengan kriteria**, submission Anda akan **ditolak** oleh reviewer. Berikut poin-poinnya.

1.  **Tidak melampirkan file** yang diminta pada ketentuan berkas submission.
2.  Dokumen notebook **tidak dijalankan terlebih dahulu**, sehingga output sel kode tidak terekam.
3.  Tidak mengimplementasikan fitur wajib sesuai level, mulai dari Basic, lalu Skilled, hingga ke Advanced.
4.  Menggunakan **file dokumen lain** atau hanya menggunakan **beberapa dokumen** yang disediakan sebagai knowledge RAG.
5.  Menggunakan **dataset selain yang telah disediakan** untuk melakukan fine-tuning dan reinforcement learning GRPO.
6.  Terdeteksi menggunakan model instruct/chat hasil *fine-tuning* **pihak ketiga**, atau **siswa lain** di Hugging Face yang digunakan untuk submission ini.
7.  Tautan (*link*) **model hasil** ***fine-tuning*** yang diunggah ke Hugging Face **bersifat** ***Private***, salah ketik, atau tidak disertakan.
8.  **Tidak menyembunyikan** atau **tidak menggunakan** ***environment variables*** untuk menyimpan API Key seperti Hugging Face Token.

### Forum Diskusi

Jika mengalami kesulitan, Anda bisa bertanya langsung ke [<u>forum diskusi</u>](https://www.dicoding.com/academies/818/discussions/459602).

### Ketentuan Proses Review

Beberapa hal yang perlu Anda ketahui mengenai proses review.

*   Tim penilai akan mengulas submission Anda dalam waktu selambatnya 3 hari kerja, tidak termasuk hari Sabtu, Minggu, dan libur nasional.
*   Tidak disarankan untuk melakukan submit berkali-kali karena akan memperlama proses penilaian yang dilakukan tim penilai.
*   Anda akan mendapat notifikasi hasil pengumpulan submission via email atau dapat mengecek status submission pada akun Dicoding.