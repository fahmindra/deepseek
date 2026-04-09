---
slug: modul-2-4
title: modul-2-4
---
### Studi Kasus: Aplikasi Berbasis LLM
---

Bayangkan jika kita diminta untuk membuat sebuah aplikasi berbasis LLM. Misalnya chatbot kalkulator yang bisa membantu menghitung pengeluaran berdasarkan beberapa kategori baik dari input manual maupun dari foto struk, bukti transaksi, dan sejenisnya.

Nah, secara garis besar, apa saja langkah yang perlu dilakukan?

Pertama, kita perlu memahami terlebih dahulu use case-nya. Karena aplikasi ini harus memproses teks dan gambar sekaligus, berarti kita perlu memilih LLM multimodal. Selain itu, karena aplikasi akan berurusan dengan foto struk, penting untuk memilih model yang memiliki kemampuan yang baik dalam membaca dan mengenali isi struk secara akurat.

Selanjutnya, tentukan bentuk output yang diinginkan. Untuk aplikasi pelacak pengeluaran, biasanya kita butuh informasi seperti:
*   Tanggal
*   Jumlah pengeluaran, dan
*   Kategori pengeluaran.

Artinya, hasil keluaran dari LLM sebaiknya punya format yang mudah dibaca dan diproses oleh program lain. Format JSON bisa jadi pilihan yang pas untuk kebutuhan ini. Selanjutnya, JSON tersebut dapat diproses lebih lanjut untuk kebutuhan berikutnya, misalnya untuk perhitungan lanjutan atau dimasukkan ke dalam *database*.

Selain itu, kita juga perlu mengantisipasi kalau pengguna mengunggah gambar yang salah, misalnya bukan struk, tapi foto lain atau *screenshot* yang tidak relevan. Dalam kasus seperti ini, output dari model perlu menyertakan penanda (*flag*) yang menunjukkan apakah gambar tersebut valid atau tidak.

Struktur JSON yang diinginkan dapat digambarkan sebagai berikut:

```json
{
    "is_valid": boolean,
    "date": datetime,
    "spending_amount": float,
    "category": enum[... tentukan kategorinya apa saja..]
}
```

Akan terasa agak merepotkan kalau setiap kali pengguna mengirim gambar atau deskripsi teks pengeluaran, mereka juga harus menulis instruksi. Jadi, lebih efisien kalau input dari pengguna cukup berupa gambar atau deskripsi teks saja. Nantinya, sistem yang akan menggabungkan input tersebut dengan instruksi untuk membentuk prompt yang dikirim ke LLM.

Selain itu, karena aplikasi ini digunakan untuk keperluan pribadi dan bukan untuk pengguna umum, serta belum memerlukan pemrosesan input yang kompleks, maka input dapat langsung diteruskan ke LLM tanpa langkah tambahan pada tahap awal.

Dengan demikian, sistem yang dikembangkan perlu memenuhi beberapa kebutuhan berikut:
*   Menggunakan LLM multimodal yang memiliki performa bagus dalam memproses gambar struk, bukti transaksi, dan sejenisnya,
*   Menerima input gambar atau teks dari pengguna, lalu menyusun prompt yang akan dikirim ke LLM,
*   Mengubah output teks dari LLM menjadi format JSON dan menambahkan penanganan jika terjadi kesalahan format, kategori yang tidak dikenal, atau hal lainnya, serta
*   Memproses hasil JSON, memeriksa apakah hasilnya valid, lalu jika valid menyimpan data ke *database* dan menyiapkan fungsi untuk menampilkan rekap, misalnya per bulan atau per jenis pengeluaran.

Setelah sistem selesai dibuat dan diuji, *AI Engineer* menyiapkan sistem untuk dioperasionalkan di lingkungan production, misalnya dengan membuat *Docker container* dan menyiapkan infrastrukturnya. Setelah semuanya berjalan lancar, kita berhasil mengembangkan sistem yang diinginkan.

Namun setelah digunakan, *AI Engineer* kemudian menyadari ada kebutuhan baru: pengguna ingin bisa menambahkan kategori pengeluaran sendiri. Berarti selain menerima input dari pengguna, sistem juga perlu menambahkan informasi tambahan atau konteks baru ke dalam *prompt*, yaitu daftar kategori yang digunakan untuk mengelompokkan transaksi.

Agar pengguna tidak perlu repot memasukkan daftar kategori secara manual setiap kali, sistem bisa menambahkan kategori tersebut ke *prompt* secara otomatis. Cukup dengan membuat fitur *add category*, di mana kategori baru disimpan dalam *database*, lalu dikirim ke LLM bersamaan dengan instruksi yang sudah diperbarui agar model tahu kategori apa saja yang tersedia.

Sekarang sistem sudah bisa menangani kategori tambahan. Tapi ternyata hasil klasifikasinya sering kurang akurat karena model belum benar-benar memahami arti setiap kategori. Solusinya, setiap kategori sebaiknya dilengkapi dengan deskripsi singkat atau kriteria dalam bahasa natural.

Jadi, ketika pengguna menambahkan kategori baru, mereka juga menuliskan sedikit penjelasan atau contoh untuk membantu LLM memahami maksud kategori tersebut. Sistem kemudian perlu diperbarui agar bisa menyusun prompt yang menyertakan informasi baru ini. Dengan begitu, hasil yang diberikan LLM bisa lebih konsisten dan relevan.

Sekarang sistem sudah berjalan sesuai kebutuhan. Jika di kemudian hari masih terdapat kekurangan atau kesalahan, *AI Engineer* dapat menelusuri penyebabnya dan memperbarui sistem agar dapat menangani atau bahkan mencegah masalah serupa di masa mendatang.

Studi kasus ini sebenarnya sangat disederhanakan, tapi kurang lebih beginilah proses yang akan dilalui seorang *AI Engineer* saat membangun sebuah sistem berbasis LLM.