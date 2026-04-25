---
title: Proyek Akhir
slug: matematika-7
abstract: Siswa diminta untuk menyelesaikan permasalahan probabilitas dan statistika yang mencakup implementasi algoritma naive bayes dan uji hipotesis melalui program berbasis bahasa pemrograman Python.
---

## Pengantar

Selamat! Anda telah menyelesaikan kelas Belajar Matematika untuk Data Science yang membahas tuntas tentang materi-materi berikut.

*   Data Univariat dan Multivariat
*   Dasar-Dasar Probabilitas
*   Distribusi Statistik
*   Statistik Inferensial
*   Studi Kasus A/B Testing dengan Python
*   Matematika pada Algoritma AI

Tak hanya itu, Anda pun sudah belajar uji hipotesis melalui skenario pengujian A/B dalam modul khusus. Setelah mempelajari seluruh materi dengan baik, harapannya Anda mampu menerapkan prinsip-prinsip atau pengetahuan terkait probabilitas dan statistika dalam alur kerja data science. 

Untuk mengasah sekaligus memvalidasi kemampuan, pada submission ini Anda akan diminta mengerjakan beberapa soal terkait probabilitas dan statistika menggunakan Python sebagai alatnya.

## Kriteria

Setiap kriteria dapat bernilai 0 sampai 4 _points_ (pts). Untuk lulus dari submission ini, Anda **harus mendapatkan 2 points dari setiap kriteria**. Submission akan **ditolak** jika masih ada kriteria dengan 0 points. 

> **Catatan:** Mohon periksa tab “Lainnya” untuk informasi lebih lanjut terkait Ketentuan Pengiriman Berkas, Ketentuan Submission Ditolak, dan Ketentuan Proses Review.

**Perhatian!**

Harap diperhatikan bahwa submission Anda akan dinilai oleh sistem otomatis. Sistem akan langsung **menolak submission** jika menemukan fungsi pertama yang gagal, tanpa melanjutkan ke pengujian berikutnya.

Anda dapat mengirimkan ulang submission **berkali-kali hingga berhasil**. Karena jawaban review tidak spesifik, silakan manfaatkan beberapa hal berikut untuk membantu Anda mengevaluasi jawaban submission:

*   Petunjuk pada Google Colab
    
*   Forum Diskusi
    

### Kriteria 1: Mengimplementasikan Algoritma Naive Bayes dan Uji Hipotesis

Anda ditugaskan untuk menyelesaikan tiga kasus utama probabilitas dan statistika. Semua soal tersebut dapat diakses pada Google Colab berikut.

*   Soal 1: [soal-1-distribusi-dan-algoritma-naive-bayes.ipynb](https://colab.research.google.com/drive/14eb8I6jtaEpOTRuy9biewSm22b3jjZZW?usp=sharing)
*   Soal 2: [soal-2-sistem-rekomendasi-ab.ipynb](https://colab.research.google.com/drive/1qrcwnUCxvpQOZ_PrE6pHh9XBxjoSFBnW?usp=sharing)
*   Soal 3: [soal-3-software-dev-ab.ipynb](https://colab.research.google.com/drive/1Ja3pXn-ga9t8HIMyGnajj36rI8CGM0Kg?usp=sharing)

Ada beberapa hal yang perlu diperhatikan saat Anda mengerjakan soal-soal di atas.

*   Tidak diperkenankan menggunakan library siap pakai, seperti scikit-learn untuk regresi linear.
*   Setiap soal yang Anda submit akan melalui tahapan pengujian otomatis dari platform Dicoding, jika terdapat kegagalan, submission ditolak.

> **Catatan:** Anda dapat meninjau kriteria submission yang akan ditolak pada poin ketentuan kriteria utama di bawah.

*   Perhatikan petunjuk berikut untuk mengerjakan submission.
    *   Perhatikan setiap petunjuk yang diberikan, guna memudahkan Anda menjawab sesuai yang diharapkan oleh sistem.
    *   Anda diwajibkan untuk menulis kode di antara komentar \`# MULAI KODE DI SINI\` dan \`# AKHIRI KODE DI SINI\`.
    *   Setiap fungsi memiliki cell pengujian sederhana, Anda dapat menggunakannya untuk memastikan output sesuai dengan yang diharapkan. 
    *   Tidak diperkenankan menggunakan library tambahan. 
    *   **TIDAK DIPERBOLEHKAN MENGGUNAKAN AI** untuk menjawab soal-soal tersebut. 

Setiap soal dirancang untuk menguji kemampuan analisis Anda, khususnya dalam menerapkan prinsip-prinsip probabilitas dan statistika. Oleh karena itu, fokus utama dari soal-soal ini adalah mengasah cara berpikir statistik Anda.

Berikut adalah ketentuan dari kriteria pertama.

*   Reject (0 pts)
    *   **Memodifikasi setiap kode dan rumus** dari soal-soal di luar ketentuan yang diberikan.
    *   **Mengubah nama berkas** yang sudah ditentukan dalam soal.
    *   Mengubah _seeding_yang ditentukan dalam soal.
        *   Beberapa soal menerapkan _seeding_ untuk menjaga konsistensi output. 
    *   Menggunakan library lain di luar yang ditentukan. 
        *   Untuk kasus z-test dan t-test, tidak diperkenankan langsung menggunakan method dari \`statsmodels.stats.proportion.proportions\_ztest\` dan \`scipy.stats.ttest\_rel\` atau sejenisnya. 
    *   Mengubah nama fungsi yang ditentukan dalam soal.
        *   Semua soal sudah diatur untuk bisa diselesaikan menggunakan library yang sudah ditentukan. 
    *   Menggunakan AI untuk melengkapi kode dari soal-soal yang diberikan.
        *   Gunakan AI atau Forum Diskusi sebagai teman belajar Anda saja.
    *   Output setiap fungsi yang didefinisikan tidak sesuai dengan yang diharapkan oleh sistem.
    *   Terdapat kegagalan pada kriteria basic.

**Catatan:** Di luar ketentuan yang telah ditetapkan, sistem Dicoding akan menguji berbagai skenario untuk mendeteksi potensi kecurangan atau output tidak sesuai. Jika terindikasi adanya pelanggaran atau hasil yang tidak valid, sistem secara otomatis akan menolak submission Anda.

*   Basic (2 pts)
    *   Mengirimkan soal pertama dengan nama file \`**soal-1-distribusi-dan-algoritma-naive-bayes.ipynb\`**.
        *   Perhatikan penggunaan dash (-).
    *   Menyelesaikan semua tugas pada soal.
    *   Setiap fungsi menghasilkan output **sesuai dengan yang diharapkan**.
    *   Tidak ada kegagalan pada proses pengujian sistem.
*   Skilled (3 pts)
    *   **Memenuhi semua ketentuan Basic.**
    *   Mengirimkan satu soal tambahan dari yang ditentukan.
        *   \`**soal-2-sistem-rekomendasi-ab.ipynb**_**\`**_ atau \`**soal-3-software-dev-ab.ipynb**_**\`**_.
            *   Perhatikan penggunaan dash (-)
    *   Menyelesaikan semua tugas pada soal tambahan.
    *   Output setiap fungsi yang didefinisikan sesuai dengan yang diharapkan.
    *   Tidak ada kegagalan dalam pengujian.
        *   Jika mengerjakan **Basic** dan **Skilled**, tapi soal **Skilled gagal**, skor tetap _**2**_ **(Basic)**.
*   Advanced (4 pts)
    *   **Memenuhi semua ketentuan “Basic” dan “Skilled”.**
    *   Menyelesaikan dan mengirimkan tiga soal yang ditentukan (perhatikan penggunaan dash \`-\`).
        *   _**\`**_soal-1-distribusi-dan-algoritma-naive-bayes.ipynb\`
        *   \`soal-2-sistem-rekomendasi-ab.ipynb\` 
        *   \`soal-3-software-dev-ab.ipynb\`
    *   Menyelesaikan semua tugas pada soal-soal yang disebutkan.
    *   Output setiap fungsi yang didefinisikan sesuai dengan yang diharapkan.
    *   Tidak ada kegagalan dalam pengujian untuk semua soal.

**Catatan:**   
Kegagalan dari sistem dapat disebabkan beberapa faktor sebagai berikut.

*   Fungsi tidak mengembalikan nilai sesuai dengan output yang diharapkan.
*   Fungsi terindikasi melakukan kecurangan, seperti melakukan _brute force_ pada nilai yang dikembalikan.
*   Fungsi terindikasi menggunakan library yang dilarang, seperti \`statsmodels.stats.proportion.proportions\_ztest\` alih-alih mengikuti rumus yang diberikan. 
*   Perbedaan hingga 0.00xx pada fungsi x dapat berpengaruh pada output di fungsi y. Sehingga adanya perbedaan output di salah satu fungsi akan ditolak kendatipun perbedaannya dalam skala decimal. 

Selain itu, perhatikan penamaan berkas. Jika Anda mengerjakan level Skilled atau Advanced tetapi mengubah nama berkas, sistem akan menganggap Anda hanya mengerjakan level Basic**.**

### Kriteria 2: Knowledge Check

Dalam kriteria kedua, Anda diminta untuk mendalami teori-teori yang melandasi soal-soal pada kriteria pertama. Seluruh soal disajikan dalam bentuk pilihan ganda dan merujuk pada teori yang telah dibahas dalam kriteria pertama. 

Silakan jawab seluruh soal dan simpan jawaban Anda dalam spreadsheet berikut.

*   [Kriteria 2 - Knowledge Check Submissions](https://docs.google.com/spreadsheets/d/1FlmUhgRUnkIbSu_0XmWGdmNI88hkbOhCxjwfYc3oCk4/copy)

> **Catatan:** Buka template di atas, lalu salin spreadsheet tersebut menjadi berkas milik Anda. 

Untuk memenuhi kriteria ini, Anda perlu mencapai skor minimal sesuai dengan ketentuan berikut.

*   Reject (0 pts)
    *   **Tidak menggunakan** template spreadsheet yang sudah ditentukan.
    *   Mengubah nama sheet, subsheet, dan soal pada template spreadsheet yang ditentukan dengan cara mengubah/menghapusnya.
    *   Menambahkan penjelasan dalam setiap opsi jawaban yang diberikan.
        *   Cukup menulis A, B, C, atau D saja.
    *   Terdeteksi melakukan tindakan kecurangan.
    *   Mendapatkan skor di bawah 80% pada tingkat Basic.
        *   Jika **memperoleh skor di bawah** 80% pada soal tingkat Skilled dan Advanced, Anda tetap dinyatakan lulus, tapi belum memenuhi kriteria untuk tingkatan tersebut. 
*   Basic (2 pts)
    *   Menjawab soal pada kategori “Soal 1- Distribusi dan Algoritma Naive Bayes” yang dapat diunduh pada link di bawah ini.
        *   [**\[Klik tautan ini untuk mengunduh soal\]**](https://drive.google.com/file/d/1XTYeIKTxH6GCEjLuN4L3xBbRSMdgUnFu/view?usp=sharing).
    *   Jawaban soal disimpan pada subsheet bernama **“soal-1-distribution-naive-bayes”****di template yang diberikan**.
        *   Ingat untuk menggunakan subsheet yang ada pada template dan **bukan membuat spreadsheet baru**.
    *   Setiap soal yang diberikan dan **mendapatkan skor minimal 80%** (Berhasil menjawab 4 dari 5 soal).
*   Skilled (3 pts)
    *   **Memenuhi kriteria Basic.**
    *   Menjawab **salah satu soal tambahan** dengan kategori sesuai yang ditentukan.
        *   [Soal 2 - Sistem Rekomendasi - A\_B Testing](https://drive.google.com/file/d/1DU9RAlYnQ59ZGacWfCW6XiFpUZHs9LGG/view?usp=sharing), atau
        *   [Soal 3 - Software Development - A\_B Testing](https://drive.google.com/file/d/1a2yOv3EnB-n_Uv4I8CdwoISvRcvRboJZ/view?usp=sharing).
    *   Jawaban soal disimpan pada subsheet sesuai nomor soal yang dikerjakan.
        *   Soal 2 disimpan pada subsheet bernama “soal-2-sistem-rekomendasi-ab”.
        *   Soal 3 disimpan pada subsheet bernama “soal-3-software-dev-ab”.
    *   **Mengerjakan** setiap soal yang diberikan dan **mendapatkan****skor minimal 80%**(Berhasil menjawab 4 dari 5 soal).
        *   Jika satu soal mendapatkan skor di bawah 80%, mendapatkan poin 2 (Basic).
*   Advanced (4 pts)
    *   **Ketentuan Basic dan Skilled terpenuhi.**
    *   Menjawab dan seluruh soal yang diberikan pada **subsheet** yang telah ditentukan.
        *   \`soal-1-distribution-naive-bayes.xlsx\`.
        *   \`soal-2-sistem-rekomendasi-ab.xlsx\`.
        *   \`soal-3-software-dev-ab.xlsx\`.
    *   Setiap soal yang dikerjakan mendapatkan skor minimal 80% (Berhasil menjawab 4 dari 5 soal).
        *   Jika satu soal mendapatkan skor di bawah 80%, mendapatkan poin 3 (Skilled).
        *   Jika dua soal mendapatkan skor di bawah 80%, mendapatkan poin 2 (Basic).

**Catatan:**  
Harap berhati-hati dalam memberikan jawaban pada baris yang ditentukan. Jika Anda memberikan jawaban seperti “B. Ini adalah opsi benar”, otomatis sistem akan menganggap Anda memberikan jawaban yang keliru. **Mohon untuk hanya memberikan jawaban “A”, “B”, “C”, atau “D”.** 

Selain itu, perhatikan penamaan berkas. Jika Anda mengerjakan level **Skilled** atau **Advanced**, tetapi mengubah nama berkas, sistem akan menganggap Anda hanya mengerjakan level **Basic.**

## Ketentuan Penilaian

### Perhitungan Nilai

Nilai akhir yang Anda dapatkan diperoleh melalui perhitungan formula berikut.

![dos-c3569692c09a07093ab584c32311dd1520250903164935.png](https://assets.cdn.dicoding.com/original/academy/dos-c3569692c09a07093ab584c32311dd1520250903164935.png)

**Catatan**

Perhitungan nilai akhir di atas digunakan apabila setiap kriteria mendapatkan nilai 2 pts atau tidak ada kriteria yang ditolak.

### Tabel Penilaian

Adapun untuk penilaian submission dapat dilihat pada tabel berikut.

|     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- |
| **Ketentuan Penilaian** |     |     |     |     |     |
| **Nilai Akhir** | **Nilai Dicoding** | **Nilai Huruf** | **Level of Mastery** | **Makna Nilai** | **Keterangan** |
| <1  | Rejected | E   | \-  | Tidak Lulus | Anda sudah mencoba, tetapi belum memenuhi kompetensi minimal. |
| 1 – <2 | Bintang 2 | D   | Below Basic | Kurang | Anda sudah memenuhi semua kompetensi minimal, tetapi terdapat area yang masih bisa ditingkatkan. |
| 2 – <3 | Bintang 3 | C   | Basic | Cukup | Anda sudah memenuhi semua kompetensi minimal dari _learning objective_. |
| 3 – <4 | Bintang 4 | B   | Skilled | Mahir | Anda sudah memenuhi semua kompetensi dengan baik atau mahir. |
| 4   | Bintang 5 | A   | Advanced | Tingkat Lanjut | Anda sudah memenuhi semua kompetensi dengan sangat baik atau tingkat lanjut. |


## Tips dan Trik

Ketika mengerjakan submission, mungkin Anda mengalami kendala. Oleh karena itu, kami mengumpulkan beberapa kendala yang sering ditemui siswa-siswa lain. Agar kendala tersebut tidak terulang kepada Anda, kami telah mengumpulkan tips untuk menanggulanginya.

Berikut adalah beberapa tips yang dapat Anda gunakan dalam pembuatan submission.

*   Topik yang dibahas dalam submission meliputi materi-materi yang dibahas pada modul **Distribusi Probabilitas, Statistik Inferensial, dan Studi Kasus A/B Testing dengan Python**. Anda dapat meninjau kembali materi-materi pada modul tersebut sembari mengerjakan submisison.
*   Kami telah menyediakan petunjuk untuk setiap soal, termasuk referensi penggunaan **library** yang relevan. Pastikan Anda memperhatikan setiap petunjuk tersebut dengan saksama.
*   Pastikan semua fungsi dan bagian kode yang **DILARANG DIUBAH**, tidak berubah sedikit pun. 
*   Gunakan **cell pengujian** sebelum masuk ke tahapan berikutnya, guna memastikan output sesuai dengan yang diharapkan.
*   Jangan upload submission jika Anda merasa ada output yang salah. Sistem akan menganggap Anda memberikan jawaban salah secara langsung.
    
*   Manfaatkan forum diskusi untuk berdiskusi terlebih dahulu menentukan jawaban.  
      
## Lainnya

### Ketentuan Pengiriman Berkas Submission

*   **Mengirimkan semua berkas** dan soal dalam .ZIP yang berisi soal-soal yang sudah dikerjakan dalam **dua format**, yakni **.ipynb** dan **.py.**
    *   Anda dapat mengunduhnya kedua format dalam Google Colab/Notebook dengan cara berikut.  
        ![dos-2b7e6d8c401f8b982d5f024ac9c28ce120250903165108.jpeg](https://assets.cdn.dicoding.com/original/academy/dos-2b7e6d8c401f8b982d5f024ac9c28ce120250903165108.jpeg)
*   **Jangan ubah nama file .py dan .ipynb** untuk memastikan sistem dapat menilai dengan benar.
*   Berikut adalah struktur berkas yang direkomendasikan.
    
    ├── soal-1-distribusi-dan-algoritma-naive-bayes.ipynb
    ├── soal-1-distribusi-dan-algoritma-naive-bayes.py
    ├── soal-2-sistem-rekomendasi-ab.ipynb (jika mengerjakan)
    ├── soal-2-sistem-rekomendasi-ab.py (jika mengerjakan)
    ├── soal-3-software-dev-ab.ipynb (jika mengerjakan)
    ├── soal-3-software-dev-ab.py (jika mengerjakan)
    └── kriteria-2-knowledge-check-submissions.xlsx
    

### Ketentuan Submission Ditolak

Submission Anda akan ditolak bila

*   setiap kriteria submission tidak terpenuhi,
*   ketentuan berkas submission tidak terpenuhi,
*   melakukan kecurangan, seperti tindakan plagiasi.

### Ketentuan Proses Review

Beberapa hal yang perlu Anda ketahui mengenai proses _review_.

*   Tim Reviewer akan mengulas submission Anda dalam waktu **selambatnya 3 (tiga) hari kerja** (_tidak termasuk Sabtu, Minggu,_ dan _hari libur nasional_).
*   **Tidak disarankan** untuk melakukan _**submit**_ **berkali-kali** karena akan memperlama proses penilaian.
*   Anda akan mendapatkan notifikasi hasil _review submission_ via email. Status _submission_ juga bisa dilihat dengan mengecek halaman [submission](https://www.dicoding.com/academysubmissions/my).

```
This file is located at: {{ page.path }}
```
---



