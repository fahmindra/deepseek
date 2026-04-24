---
slug: modul-11-3
title: modul-11-3
---
Berikut adalah isi konten yang telah dirapikan ke dalam format Markdown sesuai dengan hirarki aslinya:

# 7.3 Mengenali Orkestrasi Workflow Berbasis Graf

---

Setelah melihat bagaimana sebuah *Agent* menggabungkan penggunaan *tool*, *reasoning* serta *state* untuk melakukan suatu *task* secara mandiri, bisa diamati sesuatu yang menarik: **sebuah *agent* bisa dipandang sebagai sebuah workflow**.

Ketika sebuah *Agent* bekerja, misalnya yang berbasis ReAct, ada pola tertentu yang diikuti:

1.  Memikirkan bagaimana cara menyelesaikan suatu *task*
2.  Memilih sebuah aksi
3.  Eksekusi *tool*
4.  Amati hasilnya
5.  Kembali ke (1) atau berikan hasil jika sudah selesai

Pola ini merupakan sebuah *workflow*, yang membedakan adalah sebagian tahap dari *workflow* ini memiliki lingkup yang lebih besar dan abstrak, bukan hanya sebuah *task* spesifik. Sebuah *agent* dapat dipandang sebagai *workflow* yang memiliki struktur transisi yang sangat dinamis.

Sebuah ***workflow*** merupakan susunan **tahap-tahap berurut yang mengikuti sebuah *alur*** yang sudah ditentukan. Setiap tahap memproses sebuah input, menghasilkan output, lalu dilanjutkan ke tahap selanjutnya berdasarkan suatu aturan **transisi** tertentu. Ilustrasi di bawah menunjukkan representasi *Agent* sebagai sebuah *workflow*.

![Representasi Agent sebagai workflow](https://lms.sdmdigital.id/pluginfile.php/635493/mod_page/content/21/image33.jpg)
**Gambar 8. Representasi cara kerja sebuah *Agent* sebagai *workflow* dengan percabangan dan *looping***

Walaupun struktur ini (*loop* ReAct) dapat digunakan untuk penyelesaian masalah secara dinamis, coba bayangkan apa yang terjadi ketika masalahnya cukup kompleks, misalnya:

*   Sebagian aksi dilakukan secara bertahap dalam urutan yang ditentukan.
    **Contoh**: Saat *melakukan checkout*, ada tahapan **hitung total tagihan → *checkout* → pilih opsi pengiriman → pembayaran**
*   Transisi *state* yang memerlukan logika kompleks
*   Logika kondisional rumit yang bisa mengubah ruang aksi yang perlu dilakukan
*   Melakukan pemanggilan *tool* secara paralel
*   *Human-in-the-loop*, interupsi operator manusia saat sebuah *Agent* sedang bekerja untuk melakukan *review* atau memberikan *feedback*

Kemampuan **membuat keputusan** sebuah ***Agent*** bisa digabungkan dengan **struktur dan transisi** yang jelas dari sebuah ***workflow*** untuk mendapatkan keuntungan dari masing-masing pendekatan. Adanya struktur yang eksplisit memberikan beberapa keuntungan:

*   Pengecekan **eksplisit** terhadap ketercapaian tujuan maupun validitas output setiap langkah.
*   Transisi dari satu *state* ke yang lainnya memiliki aturan yang jelas dan bisa ditelusuri. Selain meminimalisir kesalahan, hal ini juga memudahkan proses *debugging*.
*   Bisa menjamin tahap-tahap yang **wajib** dilakukan sebelum atau setelah suatu proses dilakukan secara reliabel dengan cara “memaksa” sebuah *Agent* melalui tahap-tahap tersebut dalam bentuk aturan transisi *workflow*.

Walaupun *Agent* memiliki kelebihan dibandingkan *workflow* yang statis berupa kemampuan mengambil keputusan secara dinamis serta kemampuan untuk menyesuaikan perencanaannya pada situasi tertentu, **sebuah *Agent* tetap memerlukan adanya struktur dan transisi yang jelas**. Selanjutnya mari pelajari bagaimana caranya struktur ini dapat didefinisikan dalam bentuk *workflow* berbasis graf.

---

## 7.3.1 Memahami Workflow sebagai Sebuah Graf (Graph-based Workflows)

Sebuah *workflow* dapat direpresentasikan sebagai sebuah *Finite State Machine* (FSM), sebuah konsep dalam Ilmu Komputer yang mendefinisikan sebuah sistem yang berisi transisi dari satu *state* ke yang lainnya. Contoh sederhananya adalah sebuah prosedur pemesanan barang dalam sebuah *online shop*:

![FSM alur pemesanan barang](https://lms.sdmdigital.id/pluginfile.php/635493/mod_page/content/21/image9.png)
**Gambar 9. FSM sederhana yang menggambarkan alur pemesanan suatu barang**

FSM memiliki dua karakteristik utama:
*   Jumlah state yang terbatas, sesuai dengan desain sistem.
*   Setiap state memiliki aturan transisi yang jelas, yang menetapkan:
    *   State lainnya (bisa juga dirinya sendiri) yang menjadi tujuan transisi.
    *   Sebuah *event* yang menjadi pemicu transisi state ke state lainnya.

Dalam contoh di atas, `pemrosesan checkout`, `pesanan diproses` serta `pesanan dikirimkan` merupakan *state*. Setiap panah menunjukkan transisi dari satu *state* ke *state* lainnya. Teks yang berada pada setiap anak panah menunjukkan *event* apa yang memicu perpindahan *state* tersebut.

Ada banyak cara untuk merepresentasikan sebuah FSM, tapi salah satu cara paling utama adalah dengan menggunakan graf. Graf merupakan struktur data yang terdiri atas dua bagian:
*   ***Node***. Sebuah *node* dapat merepresentasikan sebuah informasi maupun sebuah proses. Dalam contoh di atas, setiap *state* adalah nodenya masing-masing.
*   ***Edge***. Sebuah *edge* menghubungkan dua *node*. *Edge* menggambarkan **hubungan** antara dua informasi atau alur dari satu proses ke proses lainnya. Dalam terminologi *graf*, dikenal istilah *directed graph*, yaitu graf yang *edge*-nya memiliki arah.

Menghubungkan kedua konsep di atas, sebuah *workflow* dapat direpresentasikan sebagai sebuah *directed graph* dengan pemetaan berikut:

**Tabel 1. Pemetaan komponen dari sebuah graf dengan bagian dari workflow**

| Komponen Graf | Bagian dari *Workflow* | Penjelasan |
| :--- | :--- | :--- |
| Node | Sebuah operasi atau langkah dalam suatu *workflow* | Setiap operasi atau langkah membaca dan mengubah suatu *state* |
| Edge | Alur dari satu langkah ke langkah berikutnya | *Event* yang menentukan alur mana yang diambil dilakukan di dalam *node* |
| Arah edge | Arah dari alur | - |

Sebagai contoh, misalnya kita membuat sebuah *graf* untuk merepresentasikan workflow sederhana:
*   Hanya ada dua node, yaitu LLM dan *Tool Call*.
*   Bergantung pada jenis input, LLM tidak selalu perlu menggunakan *tool*.
*   Node *Tool Call* berisikan banyak *tool* yang akan dipanggil sesuai *request* dari LLM.
*   Node LLM memiliki pemrosesan yang berbeda ketika hanya memproses input maupun memproses pemanggilan tool.

![Workflow sederhana loop pemanggilan tool](https://lms.sdmdigital.id/pluginfile.php/635493/mod_page/content/21/image47.png)
**Gambar 10. Sebuah *workflow* sederhana berbentuk *loop* pemanggilan *tool*.**

LLM dan Tool Call merupakan node dan setiap panah menunjukkan transisi dari satu node ke node lainnya. Setiap event yang mendasari transisi tertulis pada setiap panah. Pada gambar di atas, dapat diamati bahwa ada dua transisi yang bisa dilakukan *node* LLM. Transisi pertama adalah berpindah ke *node* `Tool Call`, dan transisi kedua adalah langsung menghasilkan jawaban dan mengakhiri *workflow* tersebut. Bagaimana caranya *node* LLM menentukan *event* yang memicu transisi? Jawabannya adalah melalui ***state***. Dalam sebuah *workflow*, *state* mempengaruhi apa yang terjadi di dalam sebuah *node* termasuk transisi menuju tahap berikutnya.

Apa saja perilaku berbeda yang harus dilakukan oleh *node* LLM?
1.  Ketika LLM memutuskan untuk memanggil *tool*, transisi selanjutnya adalah menuju *node* `Tool Call`.
2.  Ketika LLM sudah mendapatkan informasi yang cukup atau sudah menyentuh batasan tertentu, LLM menghasilkan jawaban akhir lalu mengakhiri *workflow*. Sebagai ilustrasi, mari batasi pemanggilan *tool* maksimal sebanyak 3 kali sebelum LLM harus memberikan jawaban.

Node LLM juga memerlukan beberapa informasi, yaitu:
*   Jumlah total pemanggilan *tool*
*   Input dari *user*
*   Semua hasil pemanggilan *tool*

Selain itu, ada informasi yang diperlukan oleh *node* `Tool Call`, yaitu:
*   Nama *tool* yang perlu dipanggil
*   Parameter untuk memanggil *tool* tersebut.

Informasi tersebut perlu dimasukkan ke dalam *state*. Dalam sebuah *workflow*, semua tahap dapat membaca dan melakukan *update* pada *state*. *State* yang diperlukan untuk *workflow* di atas memiliki format dan nilai awal berikut:

```json
{
    "user_input": "",
    "final_output": "",
    "tool_call": null,
    "tool_results": [],
    "total_tool_calls": 0
}
```

*Pseudocode* berikut mendeskripsikan logika yang terjadi dalam *node* LLM saat memproses jawaban.

```python
def llm_node(state):
    instruction ="""
    Based on user input and information gathered so far,
    Generate an answer or call a tool.
    """
    # kasus khusus, kalau sudah melebihi batas pemanggilan tool
    # anggap saja nilai ini ditetapkan di bagian lain dalam kode
    if state["total_tool_calls"] >= BATAS_PEMANGGILAN:
        final_prompt = generate_prompt(
            "Jawab pertanyaan user berdasarkan informasi yang diberikan.",
            state["user_input"],
            state["tool_results"]
        )
        respon_llm = call_llm(final_prompt)
        state["final_output"] = respon_llm.output
        go_to_node("END")
   
    if ada_hasil_pemanggilan_tool(state):
        final_prompt = generate_prompt(instruction, state["user_input"], state["tool_results"])
    # kalau hanya ada input, belum pernah memanggil tool
    else:
        final_prompt = generate_prompt(instruction, state["user_input"])
   
    respon_llm = call_llm(final_prompt)
   
    # kasus kalau ada tool call, harus pindah ke node tool call.
    # detail pemanggilan tool disimpan pada state
    if ada_pemanggilan_tool(respon_llm):
        state["tool_call"] = respon_llm.pemanggilan_tool
        go_to_node("TOOL CALL")
    # kalau tidak ada tool call, artinya LLM sudah memberikan output.
    # langsung menyimpan nilai akhir di state dan mengakhiri workflow
    else:
        state["final_output"] = respon_llm.output
        go_to_node("END")
```

Dapat dilihat bahwa hanya sebagian nilai pada *state* saja yang digunakan untuk proses yang dilakukan dalam *node*. Selain itu, sebuah *node* juga melakukan *update* pada *state* sebelum menentukan transisi ke *node* lainnya secara dinamis menyesuaikan dengan kondisi-kondisi tertentu pada *state* yang sudah ter-*update*.

Pseudocode berikut mendeskripsikan proses pemanggilan tool. Untuk memudahkan ilustrasi, diasumsikan kalau ada error, juga dikembalikan sebagai hasil tool.

```python
def tool_node(state):
    nama_tool, parameter_tool = proses_detail_pemanggilan_tool(state["tool_call"])
    hasil_panggilan_tool = panggil_tool(nama_tool, parameter_tool)
    state["tool_results"].append(hasil_panggilan_tool)
    state["total_tool_calls"] += 1
    # kosongkan pemanggilan tool agar bisa di-set lagi oleh node llm nanti
    state["tool_call"] = null
    go_to_node("LLM")
```

Karena tujuan transisi *node* `Tool Call` hanya satu, yaitu selalu ke *node* LLM, maka tidak perlu logika apapun untuk transisinya.

Sekarang mari visualisasi apa yang terjadi pada 3 skenario:

### Skenario 1: LLM tidak memerlukan *Tool*

Skenario ini bermula dengan *state* berikut, contohnya *input* dari *user* hanya berupa kata sapaan saja.

```json
{
    "user_input": "Halo!",
    "final_output": "",
    "tool_call": null,
    "tool_results": [],
    "total_tool_calls": 0
}
```

*State* ini menjadi input ke *node* LLM yang selanjutnya melakukan pemrosesan untuk menghasilkan jawaban berdasarkan nilai-nilai pada *state* tersebut. Dalam skenario 1, LLM langsung memberikan jawaban, sehingga alur perubahan state serta transisi antar tahap menjadi seperti berikut:

![Alur transisi skenario 1](https://lms.sdmdigital.id/pluginfile.php/635493/mod_page/content/21/image48.png)
**Gambar 11. Alur transisi dan perubahan state untuk skenario 1**

### Skenario 2: LLM perlu memanggil *Tool*

Skenario ini bermula dengan *state* berikut, pertanyaan yang jawabannya perlu dicari lewat internet.

```json
{
    "user_input": "Berapa umur Zinedine Zidane pada tahun 2029?",
    "final_output": "",
    "tool_call": null,
    "tool_results": [],
    "total_tool_calls": 0
}
```

Anggap saja ada dua langkah pemanggilan *tool*, yaitu *tool* untuk pencarian internet dan *tool* untuk melakukan perhitungan. Saat pertama dipanggil, akan diperoleh transisi berikut:

![Iterasi pertama skenario 2](https://lms.sdmdigital.id/pluginfile.php/635493/mod_page/content/21/image37.png)
**Gambar 12. Ilustrasi iterasi pertama pada skenario 2**

Setelah melakukan pemanggilan *tool*, LLM kemudian menerima lagi *state* sebelumnya, lalu melanjutkan lagi siklus yang sama. Kali ini, karena sudah mendapatkan tahun kelahiran Zinedine Zidane, LLM meminta *tool* kalkulator untuk menghitung umur Zidane pada tahun 2029.

![Iterasi berikutnya skenario 2](https://lms.sdmdigital.id/pluginfile.php/635493/mod_page/content/21/image31.png)
**Gambar 13. Ilustrasi iterasi berikutnya pada skenario 2**

Terakhir, LLM sudah memiliki semua informasi yang dibutuhkan, sehingga langsung memberikan jawaban akhir.

![Transisi terakhir skenario 2](https://lms.sdmdigital.id/pluginfile.php/635493/mod_page/content/21/image39.png)
**Gambar 14. Transisi terakhir untuk menghasilkan jawaban akhir.**

### Skenario 3: Batas iterasi sudah tercapai

Anggap saja batas iterasi pemanggilan *tool* adalah sebanyak 3 kali. Walaupun ada pemanggilan *tool*, berdasarkan logika di dalam *node* LLM sesuai *pseudocode* di atas, LLM harus langsung memberikan jawaban akhir walaupun mungkin belum semua informasinya didapatkan.

Untuk mempersingkat, hanya ditunjukan transisi terakhir, yaitu setelah LLM sudah melakukan 3 kali pemanggilan tool. Misalkan ini adalah kondisi terakhir dari *state*. Anggap saja ada 5 pemanggilan *tool* yang perlu dilakukan:

1.  Mencari tahun kelahiran Zidane (search)
2.  Menghitung umur Zidane (kalkulator)
3.  Mencari kapan terakhir kali Prancis menang piala dunia (search)
4.  Menghitung total jumlah gol yang dicetak prancis selama piala dunia (API)
5.  Mengurangi umur Zidane dikurangi dengan total gol Prancis (kalkulator)

```json
{
    "user_input": "Hitung umur Zinedine Zidane pada tahun 2029, lalu kurangkan dengan jumlah gol yang dicetak prancis terakhir kali mereka memenangkan piala dunia",
    "final_output": "",
    "tool_call": null,
    "tool_results": ["Zinedine Zidane lahir pada tahun 1972", "2029 - 1972 = 57"],
    "total_tool_calls": 3
}
```

Walaupun masih perlu melakukan dua pemanggilan *tool* lagi, tapi batas pemanggilan *tool* sudah tersentuh. Sehingga *node* LLM “dipaksa” untuk langsung memberikan jawaban dan mengakhiri *workflow*.

![Batas iterasi tercapai](https://lms.sdmdigital.id/pluginfile.php/635493/mod_page/content/21/image15.png)
**Gambar 15. *Node* LLM terpaksa langsung menghasilkan jawaban.**

Bagaimana cara menjawab yang baik ketika batas ini tersentuh perlu ditentukan sendiri, misal mengaku saja gagal, atau memberikan jawaban sejauh ini, lalu melanjutkan lagi proses tersebut berbekal informasi yang sudah didapatkan dalam eksekusi *workflow* selanjutnya.

---

## 7.3.2 Pola-pola Workflow Praktikal

Setelah memahami betapa pentingnya struktur bagi sebuah *Agent* serta bagaimana mengaplikasikannya dalam bentuk graf, mari pelajari beberapa pola *workflow* yang umum digunakan.

### Pola dasar: Linear, percabangan, perulangan

Ini merupakan pola paling dasar untuk menyusun workflow. Karena sebuah *workflow* memecah sebuah *task* kompleks menjadi *subtask* yang lebih sederhana dan terfokus, perlu orkestrasi masing-masing *subtask* ini menggunakan logika percabangan, perulangan, maupun transisi linear sederhana.

Perlu diingat bahwa *node* dalam sebuah *workflow* tidak terbatas hanya melakukan pemanggilan LLM saja. Setiap *node* bisa berbentuk:
*   Pemanggilan LLM
*   Pemrosesan secara programatis
*   Eksekusi *Agent* atau *workflow* terpisah
*   Intervensi / partisipasi operator manusia.

![Pola dasar workflow](https://lms.sdmdigital.id/pluginfile.php/635493/mod_page/content/21/image32.jpg)
**Gambar 16. Pola paling sederhana: linear, percabangan, perulangan**

### Pemrosesan paralel

Pola ini melibatkan lebih dari satu *node* atau proses yang berjalan secara paralel. Untuk menggabungkan semuanya, sebuah *node* akan berperan sebagai agregator untuk melakukan agregasi output dari setiap proses yang berjalan secara paralel.

![Pola paralel](https://lms.sdmdigital.id/pluginfile.php/635493/mod_page/content/21/image17.png)
**Gambar 17. Pola pemrosesan paralel**

**Kapan dipakai?**
*   Menjalankan setiap *subtask* yang menyusun sebuah *task* secara paralel jika *subtask* tersebut independen.
*   Menjalankan suatu *task* dengan beberapa cara yang berbeda untuk agregasi hasilnya melalui *voting*.

**Contoh penggunaan**:
*   Dalam implementasi *guardrails*, proses untuk menentukan apakah output toxic dan apakah output faktual bisa dijalankan secara paralel.

### Orchestrator-workers

Pola ini memerlukan perencanaan dulu untuk menentukan apa saja *subtask* yang perlu dikerjakan secara dinamis.

![Pola orchestrator-worker](https://lms.sdmdigital.id/pluginfile.php/635493/mod_page/content/21/image42.png)
**Gambar 18. Pola orchestrator-worker**

**Kapan dipakai?**
Ketika ada masalah kompleks yang perlu dipecah menjadi *subtask*, tetapi tidak bisa ditentukan sedari awal apa saja *subtask* yang perlu dilakukan. *Orchestrator* melakukan “pembagian tugas” secara *real time*, lalu dikerjakan oleh *worker* terpisah.

**Contoh penggunaan**:
*   Sebuah *coding agent* yang perlu mencari file mana saja yang menggunakan fungsi tertentu secara dinamis, lalu meminta banyak *worker* melakukan perubahan secara paralel.

### Evaluator-optimizer

Dalam pola ini, output sebuah *generator* diberikan ke *evaluator* untuk dinilai. Jika belum “lolos”, maka *evaluator* akan memberikan *feedback* dan meminta *generator* membuat output baru.

![Pola evaluator-optimizer](https://lms.sdmdigital.id/pluginfile.php/635493/mod_page/content/21/image8.png)
**Gambar 19. Pola Evaluator-optimizer**

**Kapan dipakai?**
Ketika output perlu memenuhi kriteria tertentu dimana proses evaluasinya bisa memberikan *feedback* untuk perbaikan. Feedback bisa berasal dari manusia (*human-in-the-loop*), LLM, atau program.

**Contoh penggunaan**:
*   *Agent* penulis kode yang mengecek apakah kode bisa di-*compile*. Jika error, pesan error diberikan ke generator untuk diperbaiki.

---

### Mini Quiz
*(Konten Interaktif H5P)*

> **Refleksi Singkat**
>
> *Struktur workflow berbasis graf memungkinkan sebuah sistem Agent mengeksekusi langkah-langkah dengan transisi yang jelas, sementara Agent sendiri memungkinkan pengambilan keputusan yang fleksibel berdasarkan state terkini dari sebuah sistem. Dalam skenario nyata, kedua aspek ini jarang berdiri sendiri, biasanya justru harus dipadukan agar dapat menangani proses yang kompleks dan dinamis.*
>
> *Jika kamu diminta merancang workflow graf untuk Agent yang menangani proses audit keamanan sistem (misalnya: scanning, analisis temuan, rekomendasi tindakan, dan verifikasi ulang), bagaimana kamu menentukan batas antara bagian yang harus ditetapkan secara statis sebagai node dalam workflow dan bagian yang sebaiknya dibiarkan fleksibel melalui reasoning Agent? Jelaskan bagaimana keputusanmu mempengaruhi reliabilitas dan adaptabilitas sistem.*