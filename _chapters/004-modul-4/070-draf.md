---
slug: modul-4-7
title: modul-4-7
---
### Studi Kasus: CodeLLaMA
---

#### **Problem**
LLaMA 2 dilatih menggunakan dataset yang sebagian besar terdiri dari teks biasa, walaupun sebagiannya berupa kode. Meta ingin melatih LLaMA2 agar memiliki kemampuan *coding* yang lebih baik.

#### **Pendekatan**
CodeLLaMA dilatih dengan cara melakukan *Continued Pretraining* (CPT) dengan LLaMA 2 sebagai *base model* menggunakan task *autoregressive* dan *code infilling* (task yang kurang lebih mirip dengan autocomplete kode) secara *self-supervised* pada dataset kode yang berisi 500B-1T token, tergantung model yang dilatih.

Tahap ini penting karena terbukti pada studi lanjutan mereka bahwa secara langsung melakukan SFT untuk domain Coding tanpa CPT memberikan hasil yang lebih buruk dibanding melakukan CPT dulu sebelum SFT, meskipun penelitiannya tidak menyebutkan besar perbedaannya.

Dataset untuk SFT dikumpulkan dengan gabungan data yang dikumpulkan secara manual dan pendekatan yang mereka sebut *self-instruct*, yaitu meminta LLM membuat instruksi untuk tugas *coding* tertentu, menghilangkan duplikat, lalu menggunakan model hasil CPT yaitu Code LLaMA-7B untuk men-generate *unit test* dan beberapa sampel kode yang sesuai dengan instruksi. Kode yang berhasil lolos *unit test* ini dijadikan pasangan sampel bersama dengan instruksinya. Selanjutnya LLM hasil CPT dilatih menggunakan SFT seperti pada umumnya. Pendekatan ini menghasilkan performa yang kompetitif dengan model lainnya yang memiliki ukuran berkali lipat.

#### **Hasil**
Model yang dilatih yaitu Code LLaMA, CodeLLaMA Instruct dan CodeLLaMA Python menunjukkan performa yang baik bila dibandingkan dengan model lain yang jumlah parameternya lebih besar. Tabel berikut menyajikan perbandingan tersebut. Perhatikan bahwa yang digunakan untuk semua model yang dilatih adalah model terkecilnya.

**Tabel 10. Perbandingan varian terkecil dari setiap jenis CodeLLaMA dengan LLaMA2-70B (base model)**

| Model | Jumlah parameter | HumanEval@pass1 | MBPP@pass1 |
| :--- | :---: | :---: | :---: |
| LLaMA2 | 70B | 30.5% | 45.4% |
| CodeLLaMA | 7B | 33.5% | 41.4% |
| CodeLLaMA - Instruct | 7B | 34.8% | 44.4% |
| CodeLLaMA - Python | 7B | 38.4% | 47.6% |
| Unnatural CodeLLaMA | 34B | 62.2% | 61.2% |

Pass@1 artinya model diminta menghasilkan 1 jawaban saja, yang langsung dinilai kebenarannya. Selain itu, CodeLLaMA juga dibandingkan dengan berbagai model lain. Untuk perbandingan ini, digunakan 2 varian tertinggi dari setiap model. Tabel berikut menyajikan perbandingan tersebut:

**Tabel 11. Perbandingan performa CodeLLaMA dengan model lainnya**

| Model | Jumlah parameter | HumanEval@pass1 | MBPP@pass1 |
| :--- | :---: | :---: | :---: |
| GPT-3.5 | - | 48.1% | 52.2% |
| GPT-4 | - | **67%** | - |
| PaLM- Coder | 540B | 33.5% | 41.4% |
| CodeLLaMA | 34B | 48.8% | 55.0% |
| CodeLLaMA | 70B | 53.0% | 62.4% |
| CodeLLaMA - Instruct | 34B | 41.5% | 57.0% |
| CodeLLaMA - Instruct | 70B | **67.8%** | 62.2% |
| CodeLLaMA - Python | 34B | 53.7% | 56.2% |
| CodeLLaMA - Python | 70B | 57.3% | **65.6%** |
| Unnatural CodeLLaMA | 34B | 62.2% | 61.2% |

Dapat dilihat bahwa performanya bersaing bahkan dengan model closed source seperti GPT-3.5 dan GPT-4 yang bisa diasumsikan memiliki ukuran parameter yang jauh berkali lipat.

> **Pelajaran yang bisa diambil**
> 
> Untuk domain spesifik, apalagi yang memiliki kosakata dan istilah serta struktur tertentu, CPT menjadi pilihan yang lebih tepat.
> Penelitian ini juga menyebutkan bahwa model yang mereka latih menggunakan SFT tidak mampu mencapai ukuran performa yang sama, walaupun tidak disebutkan figur eksaknya.

---

### Lanjutan Perencanaan Proyek LLM
---

**Judul Praktikum:** Lanjutan perencanaan proyek LLM

**Tujuan Praktikum:** Siswa mampu mengaplikasikan pengetahuan dari modul 4 untuk melanjutkan perencanaan proyek LLM yang telah dibuat pada unjuk kerja modul 1.

**Deskripsi Singkat Aktivitas:** Siswa menentukan berbagai macam hal penting seperti bagaimana cara membuat kriteria dan tujuan yang jelas dari suatu proyek dan menggunakannya untuk membuat keputusan kebutuhan data dan kebutuhan penyediaan LLM apakah bisa menggunakan yang sudah ada, perlu fine tuning, atau CPT.

#### **Langkah-Langkah Praktikum**

Berdasarkan unjuk kerja modul 1 dan apa yang kamu telah pelajari pada modul 4, detilkan perencanaan lebih lanjut untuk proyek LLM yang direncanakan.

Pertama, tentukan kebutuhan dan kriteria yang harus dipenuhi seperti:
*   Kemampuan apa yang dibutuhkan dari LLM? (menghasilkan JSON, menjawab pertanyaan, membaca dokumen legal, dsb)
*   Kriteria seperti apa yang diinginkan? (gaya bahasa, panjang respon, format output, dsb)

Melalui kriteria ini, coba tentukan:
*   Instruksi dan konteksnya kira kira perlu apa saja?
*   Kapan diputuskan kalau mungkin instruksi dan konteks tidak cukup? Apa yang perlu diukur dari LLM yang digunakan untuk memastikan dia bisa memenuhi tujuan dan kriterianya?
*   Kalau perlu fine tuning, perlu menyediakan data seperti apa? Apa justifikasi diperlukannya fine tuning?
*   Kalau perlu CPT, bagaimana cara mengumpulkan datanya? Apa justifikasi diperlukannya CPT?

Selain itu, tentunya pasti kamu akan membutuhkan data, minimal untuk melakukan validasi / evaluasi. Tentukan dan jabarkan strategimu mengumpulkan data. Apakah crawling dari internet? Apakah bisa dikumpulkan secara langsung dari interaksi dengan user? Apakah bisa digenerate oleh AI?

#### **Tools & Resource yang Dibutuhkan**
*   Text Editor untuk menulis laporan (Notepad, VS Code, Google Docs).
*   Rencana proyek LLM dari unjuk kerja modul 1.

#### **Contoh Output atau Hasil Akhir**
*   Output berupa satu dokumen berisi laporan perencanaan lanjutan dari proyek LLM yang ingin dibangun.

#### **Kriteria Keberhasilan/ Penilaian Mandiri**
*   Siswa mampu mengaplikasikan pemahaman mengenai kebutuhan data, dan berbagai macam cara mengadaptasikan perilaku LLM dalam merencanakan suatu proyek LLM.

#### **Petunjuk Troubleshooting**
*   Cobalah mencari referensi proyek yang sejenis pada literatur seperti paper ilmiah maupun blog post atau sejenisnya.
*   Coba pahami dan analisis apa yang dilakukan orang lain lalu aplikasikan ke proyek yang kamu ingin bangun.

**Durasi Praktik Disarankan:** 60 menit.