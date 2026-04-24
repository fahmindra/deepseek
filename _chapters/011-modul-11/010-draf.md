---
slug: modul-11-1
title: modul-11-1
---
Berikut adalah isi konten yang telah dirapikan ke dalam format Markdown sesuai dengan hirarki aslinya:

# 7.1 Mengenal Konsep Agent dan Pengaplikasian LLM untuk Membangun Sebuah Agent

---

### 7.1.1 Definisi Agent

#### Apa yang dimaksud sebagai sebuah *Agent*?

Menurut Russel dan Norvig dalam buku mereka *Artificial Intelligence: A Modern Approach* (AIMA), sebuah **_agent_ adalah sistem yang melakukan pengamatan dan melakukan aksi pada suatu lingkungan tertentu untuk mencapai suatu tujuan**. Mekanisme yang digunakan sebuah *Agent* untuk mengamati lingkungannya disebut sebagai **_sensor_** dan mekanisme yang digunakan untuk berinteraksi atau melakukan aksi pada lingkungannya disebut sebagai **_actuator_**.

Sebuah **lingkungan** (atau *environment* dalam terminologi aslinya) merupakan **dunia luar** yang menjadi objek interaksi agent. Aksi yang dilakukan oleh *agent* dapat mengubah kondisi pada lingkungan tersebut yang selanjutnya diamati lagi oleh *agent* untuk menentukan aksi selanjutnya. Secara pragmatis, sebuah “lingkungan” memiliki lingkup dan ruang aksi yang terbatas. Contohnya:

*   Pada *web-browsing agent* seperti [Manus](https://manus.im/features/manus-browser-operator), lingkungannya merupakan sebuah browser. *Agent* bisa memilih suatu aksi seperti mengunjungi suatu halaman, mengisi form, menekan tombol, dan bisa mengamati perubahan *state* dari lingkungannya, misalnya ada halaman baru terbuka, muncul notifikasi pemberitahuan kalau proses pemesanan tiket berhasil, dan lain sebagainya.
*   Pada *coding agent* seperti [Claude Code](https://www.claude.com/product/claude-code) dan [Cursor](https://cursor.com/), lingkungannya berisi sebuah *repository*, sebuah *compiler* dan *runtime* untuk menjalankan kode, serta interaksinya dengan operator manusia. *Agent* melakukan aksi seperti menulis kode, melakukan pengetesan kode yang tertulis, lalu membuat observasi seperti apakah kodenya berjalan baik serta mendapatkan *feedback* mengenai gaya penulisan maupun logika dari kode yang dibuat baik dari operator manusia maupun sumber lainnya.

Selain lingkungan, sebuah *agent* juga memiliki sebuah **_state_** internal, yaitu semacam **_memory_** yang menyimpan beberapa hal yang berkaitan dengan proses penyelesaian masalah yang juga digunakan untuk membuat keputusan, termasuk:

*   Tujuan akhir apa yang ingin dicapai
*   Kondisi terakhir dari sebuah lingkungan
*   Aksi apa saja yang sebelumnya dilakukan oleh *agent*
*   Apakah tujuannya sudah tercapai atau belum
*   Seberapa jauh *state* sekarang dari tujuan akhir (*progress*)

Dari berbagai macam struktur sebuah *Agent* yang dideskripsikan dalam buku tersebut, ada satu hal yang konstan: **terdapat sebuah logika yang menentukan aksi yang tepat berdasarkan hasil observasi dan *state internal* yang disimpan oleh *agent* tersebut**.

Pola amati-berpikir-berinteraksi ini berlanjut terus-menerus hingga diputuskan bahwa masalahnya sudah terpecahkan atau ketika batasan iterasi tertentu sudah tercapai. Interaksi berikut dapat digambarkan dalam ilustrasi berikut:

![Sebuah Agent mengamati lingkungannya](https://lms.sdmdigital.id/pluginfile.php/635491/mod_page/content/22/image2.png)
**Gambar 1. Sebuah *Agent* mengamati lingkungannya untuk menentukan aksi selanjutnya dalam menyelesaikan suatu masalah.**

### Kemampuan apa saja yang perlu dimiliki sebuah *agent*?

Sebuah *agent* perlu dapat bertindak secara otonom, reaktif terhadap perubahan pada lingkungannya, proaktif dalam menentukan langkah yang harus diambil, serta dalam kasus *multi-agent*, mampu bekerjasama dengan *agent* lainnya untuk mencapai suatu tujuan.

Sifat tersebut dapat dicapai melalui kemampuan-kemampuan berikut:

1.  **Kemampuan untuk memahami suatu tujuan**, termasuk menentukan apakah tujuan tersebut sudah tercapai atau belum.
2.  **Kemampuan untuk merencanakan cara mencapai suatu tujuan secara mandiri**, termasuk menyesuaikan rencana yang sudah dibuat ketika ada informasi baru selama berjalannya proses penyelesaian masalah.
3.  **Kemampuan untuk belajar dari lingkungan maupun interaksinya dengan lingkungan tersebut**, termasuk juga menyimpan sebuah *state* maupun ingatan jangka panjang untuk “mengingat” informasi yang penting selama berjalannya sebuah *agent*.
4.  **Kemampuan untuk mengamati dan berinteraksi dengan lingkungannya**. Sebuah agen perlu berinteraksi dengan lingkungannya untuk memenuhi tujuannya.
5.  **Kemampuan sosial**: Berbagi tugas, berdiskusi, serta bentuk komunikasi lainnya yang efektif untuk bekerja sama dengan *Agent* lain dalam sistem *multi-agent*.

### 7.1.2 Kenapa Perlu Agent

Berbeda dengan otomasi berbentuk *workflow* statik dimana sudah terlebih dahulu diketahui apa saja langkah yang perlu dilakukan, beberapa masalah sifatnya lebih dinamis, dimana:

*   Setiap *task* memerlukan perencanaan langkah-langkah yang berbeda untuk menyelesaikannya
*   Walaupun perencanaan sudah ditentukan, situasi tertentu memerlukan penyesuaian rencana secara dinamis.

Walaupun bisa diantisipasi berbagai kasus dengan mengimplementasi logika kondisional yang kompleks, pada titik tertentu metode ini bisa semakin tidak *maintainable* terlebih pada masalah-masalah yang memang sifatnya sangat dinamis, sehingga sulit untuk mengantisipasi berbagai macam skenario yang bisa terjadi.

Dibandingkan langkah-langkah dengan fungsi tertentu yang disusun secara statis, untuk menyelesaikan masalah yang dinamis, sebuah *Agent* memiliki keunggulan, karena:

*   Sebuah *Agent* tidak terbatas pada satu atau dua fungsional saja, tapi diberikan akses ke berbagai macam aksi yang bisa ia lakukan, dan
*   *Agent* mampu melakukan perencanaan kapan dan bagaimana aksi-aksi ini perlu dilakukan, termasuk secara dinamis mengatur ulang rencananya ketika ada informasi baru yang diamati dari lingkungannya.

![Perbedaan antara workflow statik dengan Agent](https://lms.sdmdigital.id/pluginfile.php/635491/mod_page/content/22/image16.png)
**Gambar 2. Perbedaan antara *workflow* statik dengan pendekatan berbasis *Agent* untuk menyelesaikan masalah.**

### 7.1.3 Bagaimana LLM dapat digunakan untuk implementasi sebuah *agent*?

LLM memiliki kemampuan yang baik untuk memahami instruksi serta informasi dalam bahasa natural. LLM juga memiliki pengetahuan *internal* mengenai keterhubungan konsep dalam berbagai bidang yang membantunya memahami konteks dari sebuah masalah. Akan tetapi, LLM kekurangan beberapa hal yang memungkinkannya bisa menjadi *agent*:

*   LLM tidak mampu secara langsung berinteraksi dengan lingkungannya. Baik input dan outputnya berbentuk teks.
*   LLM bersifat *stateless*, manajemen konteks perlu dilakukan secara eksternal.

Untuk itu, terdapat 3 kemampuan yang perlu diberikan kepada sebuah LLM agar bisa digunakan untuk implementasi sebuah *Agent*:

1.  Penggunaan **_Tool_**. Sebuah *tool* dapat berperan sebagai **_sensor_** maupun **_aktuator_**. Keduanya diperlukan agar LLM mampu mengamati dan berinteraksi dengan lingkungannya untuk membuat keputusan. Penggunaan *Tool* menjadi “Aksi” yang bisa dilakukan oleh *Agent* berbasis LLM.
2.  Kemampuan **_reasoning_**. *Agent* perlu bisa menentukan *tool* apa yang perlu dipanggil, kapan dipanggil, dan urutan memanggilnya ketika perlu lebih dari satu pemanggilan. Perlu perencanaan melalui kemampuan *reasoning* secara implisit (LLM yang dilatih untuk memiliki kemampuan *reasoning*) maupun secara eksplisit (melalui instruksi seperti *Chain-of-thoughts*) yang bisa digunakan untuk merencanakan langkah-langkah penyelesaian sebuah *task*.
3.  Manajemen **_memory_**. Tujuan dari suatu *task*, langkah-langkah yang telah dilakukan sejauh ini, informasi apa saja yang telah dikumpulkan serta perencanaan penyelesaian *task* perlu disimpan dalam sebuah *state* maupun *memory* jangka panjang agar bisa menjadi konteks bagi LLM selama menjalani proses penyelesaian sebuah *task*. *Memory* ini juga bisa menjadi media komunikasi antar *Agent*.

Satu hal yang perlu dipahami adalah ketika mengimplementasi sebuah *Agent* menggunakan LLM, peran LLM berubah dari **melakukan suatu *task* berdasarkan suatu instruksi** menjadi **perencana dan orkestrator** pelaksanaan sebuah *task*, walaupun pada dasarnya tetap perlu diberikan instruksi pada level yang lebih tinggi dari pelaksanaan setiap proses secara individual.

Submodul selanjutnya akan membahas secara detail bagaimana ketiga kemampuan di atas digunakan dalam implementasi sebuah *agent*. Tapi sebelum dilanjutkan, ada masalah terminologi yang penting untuk lebih memahami penggunaan istilah *Agent* di industri.

### 7.1.4 Menyatukan terminologi industri dan akademik mengenai *agent*

Dalam industri, Agent merupakan istilah yang seringkali digunakan secara kurang sesuai dengan definisi asalnya. Istilah Agent digunakan untuk menyebut sekedar LLM yang diberikan *tool*, sistem yang berbasis *workflow*, hingga sistem yang “lebih mirip *Agent*” yang mampu menyelesaikan masalah secara dinamis, tapi masih memerlukan intervensi dan instruksi dari manusia. Akan tetapi, ada suatu kesamaan: semua jenis sistem ini setidaknya menunjukkan beberapa “kemampuan” dasar dari sebuah *Agent*. Istilah *Agentic* biasanya digunakan untuk mendeskripsikan sistem seperti ini.

Daripada berfokus memenuhi secara sempurna definisi *Agent* yang ideal, pendekatan yang lebih pragmatis adalah memandang kemampuan sebuah *Agent* sebagai sebuah **spektrum**. Sebuah sistem tidak perlu benar-benar memenuhi semua kemampuan *Agent* ideal agar bisa efektif, karena penerapan sebagian kemampuan *Agent* seperti membuat rencana dan adaptasi (*Reasoning*) dan kemampuan persepsi dan berinteraksi dengan lingkungannya (Penggunaan *tool* dan manajemen *state*) walaupun dalam bentuk yang terbatas tetap bisa meningkatkan kapabilitas sistem tersebut.

Contohnya, walaupun sebuah *Agent* hanya bisa menggunakan *tool* tertentu atau masih memerlukan pengawasan dan intervensi saat membuat keputusan, kemampuan tersebut masih sangat membantu dalam penyelesaian masalah dibandingkan hanya bergantung pada merancang *prompt* yang kompleks saja.

![Spektrum Kemampuan Agent](https://lms.sdmdigital.id/pluginfile.php/635491/mod_page/content/22/image29.jpg?time=1768896694357)
**Gambar 3. Dibanding menginginkan sistem memiliki semua kemampuan Agent yang ideal, setiap kemampuan dapat ditambahkan secara pragmatis dalam bentuk terbatas**

Dalam modul ini, penyebutan *Agent* akan mengikuti istilah yang lebih longgar ini, selama suatu sistem memiliki sifat *Agentic*, yaitu:

*   Melibatkan penggunaan *tool*,
*   Bisa membuat keputusan *tool* apa yang perlu dipanggil, kapan, dan merencanakan urutan pemanggilan *tool*, dan
*   Menyimpan *state* untuk *tracking* keberjalanan suatu *task*, walaupun hanya sesimpel mengingat *tool* apa yang ingin dipanggil atau mengingat apa tujuan akhir yang ingin dicapai.

Istilah *Agent* akan tetap digunakan untuk menyebut sistem dengan karakteristik-karakteristik tersebut. Dalam modul ini, akan dijelaskan bahwa walaupun *Agent* dikontraskan dengan *workflow*, pada praktiknya sebuah *Agent* bisa diimplementasi sebagai *workflow* tanpa kehilangan kemampuan dinamisnya dalam menyelesaikan masalah.

### Mini Quiz
*(Konten Interaktif H5P: Multiple Choice)*

> **Refleksi Singkat**
>
> *Dalam sistem dinamis seperti web-browsing agent atau coding agent, kemampuan untuk mengamati perubahan lingkungan dan menyesuaikan rencana secara real-time menjadi sangat penting. Banyak workflow tradisional gagal karena terlalu statis dan tidak mampu menangani skenario tak terduga.*
>
> *Jika kamu merancang sebuah sistem agentic untuk menyelesaikan tugas yang melibatkan banyak ketidakpastian (misalnya troubleshooting server atau otomatisasi analisis dokumen), kemampuan agent mana yang akan kamu prioritaskan? mengapa? Jelaskan bagaimana prioritas itu akan mempengaruhi efektivitas sistem secara keseluruhan.*