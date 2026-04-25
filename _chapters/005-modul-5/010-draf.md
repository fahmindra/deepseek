---
slug: modul-5-1
title: modul-5-1
---
# 1.1 Bagaimana Cara Membuat Training Lebih Efisien?

Skala dari proses *training* baik itu data, jumlah parameter maupun *compute* menjadi kebutuhan utama yang harus dipenuhi untuk melatih sebuah LLM. Selain perlu memastikan ketersediaan mesin dengan spesifikasi yang memadai untuk bisa menjalankan proses *training*, jumlah komputasi yang diperlukan dalam proses training LLM juga perlu di-*manage* dengan tepat agar proses *training* tidak memakan waktu terlalu lama. Umumnya ada beberapa pendekatan yang bisa dilakukan untuk mengoptimalkan proses *training*:

1.  **Distributed training:** dilakukan dengan cara menyebar komputasi yang perlu dilakukan agar dilakukan banyak mesin sekaligus.
2.  **Mixed precision training:** secara langsung mengurangi penyimpanan dan beban komputasi melalui perubahan representasi bilangan.
3.  **Memaksimalkan setiap training step:** melalui konfigurasi data loader serta metode seperti packing untuk memaksimalkan context window saat training. Pendekatan ini akan dibahas pada submodul lain karena lebih berhubungan erat dengan persiapan data.

---

## 1.1.1 Distributed Training

Dalam proses training, kita mengetahui setiap training step terdiri dari beberapa langkah:

1.  *Batch* data diambil dari data loader.
2.  *Forward pass* dilakukan hingga didapatkan perhitungan loss.
3.  Gradien dihitung melalui *backpropagation*.
4.  Nilai gradien digunakan oleh *optimizer* untuk melakukan update parameter.

Keseluruhan proses ini membutuhkan:
1.  Memory yang cukup untuk menampung data dan parameter model.
2.  Memory yang cukup untuk menampung nilai activation, gradien, serta nilai-nilai intermediate ketika melakukan perhitungan.

Selain kebutuhan memory, komputasi yang dilakukan setiap bagian memerlukan waktu. Untuk data yang ukurannya besar (komputasi lebih rumit) dan banyak (komputasi dilakukan lebih banyak), proses ini bisa memakan waktu yang sangat lama. Distributed training bertujuan memecahkan kedua masalah ini, yaitu ketika ukuran model begitu besar sehingga untuk menyimpan parameter serta menjalankan proses training tidak cukup berada dalam 1 GPU saja, serta untuk mempercepat proses training, terutama ketika jumlah datanya sangat banyak.

Ada beberapa metode distributed training yang umum digunakan untuk melatih LLM:

**Tabel 1. Bentuk paralelisme yang digunakan dalam distributed training LLM**

| Jenis | Penjelasan singkat | Kapan digunakan? | Catatan |
| :--- | :--- | :--- | :--- |
| Data Parallelism (DP) | 1 Batch data disebar ke banyak GPU untuk diproses secara paralel | Titik awal distributed training untuk mempercepat proses training | Biasanya digunakan bersamaan. |
| Tensor Parallelism (TP) | Menyebar setiap perhitungan (terutama operasi matriks) ke banyak GPU secara paralel | Ketika tensor parameter terlalu besar, bagi ke beberapa GPU. | Kalau modelnya sangat besar, perlu digabung dengan metode lainnya |
| Pipeline Parallelism (PP) | Menyebar layer-layer suatu model ke banyak GPU | Modelnya memiliki layer yang sangat banyak, jadi perlu dibagi-bagi karena tidak cukup di 1 GPU. | |
| Context Parallelism (CP) | Memecah pemrosesan input yang panjang ke banyak GPU | Ketika context window model terlalu besar sampai aktivasinya tidak cukup di memory GPU. | |
| Expert Parallelism (EP) | Pada arsitektur MoE, setiap expert atau kelompok expert disebar ke banyak GPU | Jika menggunakan arsitektur MoE saja. | Hanya berlaku untuk MoE |

Dari ilustrasi di bawah bisa diamati cara kerja dari masing-masing metode paralelisasi dengan lebih jelas.

### 1. DP (Data Parallelism)

![DP Illustration](https://lms.sdmdigital.id/pluginfile.php/635409/mod_page/content/28/image58.jpg?time=1768376362011)
**Gambar 1. Ilustrasi cara kerja DP**

Setiap training step dilakukan secara paralel di masing-masing GPU. Artinya setiap GPU perlu menyimpan replika dari model dan optimizer yang sama. Ketika ada satu batch masuk, setiap batch dipecah jadi mini-batch, lalu setiap replika menghitung gradien. Di akhir proses, nilai gradien yang dihitung setiap GPU digabungkan lalu semua replika di-update menggunakan gradien tersebut.

### 2. TP (Tensor Parallelism)

![TP Illustration](https://lms.sdmdigital.id/pluginfile.php/635409/mod_page/content/28/image14.png)
**Gambar 2. Ilustrasi cara kerja TP**

TP dilakukan dengan cara membagi operasi agar dilakukan ke banyak GPU, yang umumnya dilakukan dengan perkalian matriks. Ketika ada matriks yang ukurannya terlalu besar, matriks tersebut disebar dipecah, disebar ke beberapa GPU, lalu perhitungannya dilakukan secara terpisah sebelum akhirnya digabungkan lagi.

### 3. PP (Pipeline Parallelism)

![PP Illustration](https://lms.sdmdigital.id/pluginfile.php/635409/mod_page/content/28/image24.png)
**Gambar 3. Ilustrasi cara kerja PP**

Berbeda dengan TP, kalau PP justru membagi setiap layer model untuk dijalankan di GPU terpisah. PP biasanya digunakan kalau sebuah model memiliki terlalu banyak layer sehingga perlu dipisah menjadi bagian yang bisa berjalan di masing-masing GPU.

### 4. CP (Context Parallelism)

![CP Illustration](https://lms.sdmdigital.id/pluginfile.php/635409/mod_page/content/28/image53.jpg)
**Gambar 4. Ilustrasi cara kerja CP**

CP biasanya digunakan ketika model dilatih menggunakan context window yang besar. CP penting terutama karena kebutuhan memory untuk mekanisme attention meningkat secara kuadratik seiring meningkatnya panjang setiap input. Caranya adalah dengan memisahkan input menjadi beberapa bagian, lalu dihitung di masing-masing GPU. Nantinya ketika 1 GPU membutuhkan informasi dari bagian input yang diproses GPU lain, terutama untuk menghitung attention akan ada proses komunikasi antar GPU dimana mereka akan saling bertukar informasi yang disimpan masing-masing.

### 5. EP (Expert Parallelism)

![EP Illustration](https://lms.sdmdigital.id/pluginfile.php/635409/mod_page/content/28/image52.jpg?time=1768376630666)
**Gambar 5. Ilustrasi cara kerja EP**

Untuk model dengan arsitektur MoE, setiap expert atau setiap kelompok expert bisa diletakkan di GPU yang berbeda-beda. Saat proses routing, hanya GPU tertentu saja yang berisi expert yang dibutuhkan yang berjalan.

> **Hal yang penting diingat:**
>
> Kita tidak bisa sembarangan mencampur Berbagai metode dan berharap training menjadi sangat cepat. Dalam distributed training, ada komunikasi yang perlu dilakukan setiap GPU, seperti sinkronisasi pada DP, menggabungkan hasil perhitungan untuk TP, mengirim data ke kelompok layer selanjutnya pada PP, maupun proses komunikasi antar GPU dalam CP.
>
> Komunikasi ini penting, tapi menghambat perhitungan karena GPU harus berhenti menghitung dulu untuk menunggu sampai data yang diperlukan diterima. Perlu eksperimentasi bahkan tuning manual untuk menentukan konfigurasi yang tepat.

---

## 1.1.2 Efisiensi DP melalui ZERO

DP adalah salah satu bentuk paralelisme yang paling umum digunakan, dan sebaiknya dijadikan pilihan paling awal sebelum digabungkan dengan bentuk paralelisme lainnya. Kita mengetahui setiap GPU tidak hanya harus menyimpan model, tapi juga menyimpan optimizer masing-masing, dan ketika melakukan training step, memory GPU juga perlu menyimpan baik gradien yang dihasilkan maupun memory aktivasi sepanjang proses. Setiap training step dari masing-masing replika juga diakhiri dengan sinkronisasi semua gradien sebelum diberikan ke semua replika untuk update parameter. Di sini kita bisa melihat sebuah redundansi.

**Optimizer yang sama, melakukan update untuk parameter yang sama, menggunakan gradien yang sama!**

![DP Detail](https://lms.sdmdigital.id/pluginfile.php/635409/mod_page/content/28/image63.jpg?time=1768376670101)
**Gambar 6. Detil lebih lanjut soal kasus DP, setiap bagian menyimpan optimizer untuk update parameter**

**ZeRO** *(Zero-Redundancy Optimizers)* merupakan solusi atas masalah ini. Dalam ZeRO, parameter dan internal state dari sebuah optimizer dipecah lalu setiap bagiannya disebar ke masing-masing GPU. Dalam setiap training step, ZERO melakukan langkah berikut:

1.  Setiap replika melakukan *feedforward* dan backpropagation untuk mendapatkan gradien dari micro batch.
2.  Semua gradien dari setiap replika saling disinkronkan agar semua replika mendapatkan semua gradien.
3.  Setiap replika **hanya melakukan update parameter untuk bagian model yang bagian state optimizernya disimpan**. Ini merupakan inti kerja ZeRO.
4.  Sebelum kembali ke (1), semua replika saling sharing bagian parameternya yang sudah ter-update untuk digabungkan melalui proses sinkronisasi, lalu diberikan ke setiap replika.

![ZeRO Steps](https://lms.sdmdigital.id/pluginfile.php/635409/mod_page/content/28/image56.jpg?time=1768376695594)
**Gambar 7. Ilustrasi cara ZeRO melakukan 1 training step secara distributed**

Dalam ZeRO, agak redundan jika seluruh gradien perlu disimpan kalau pada akhirnya hanya bagian tertentu saja dari kepingan model yang disimpan pada GPU tersebut yang di-update parameternya, sehingga muncul ZeRO-2 yang menyimpan hanya gradien yang akan di-update saja.

ZeRO-3 mengurangi kebutuhan memory lebih lanjut dengan cara juga menyebarkan parameter dari bagian model ke setiap replikanya. Saat melakukan perhitungan, ada tahap komunikasi dulu antar replika untuk menyusun parameter penuhnya, melakukan update parameter, lalu menghapus bagian parameter yang tidak disimpan untuk menghemat memory.

![ZeRO 1 2 3](https://lms.sdmdigital.id/pluginfile.php/635409/mod_page/content/28/image22.jpg?time=1768376737654)
**Gambar 8. Ilustrasi perbedaan nilai yang disimpan dalam 1 GPU untuk ZeRO, ZERO 2 dan ZERO 3**

Kelemahan ZeRO-3 adalah proses perhitungan penuh masih dilakukan pada masing-masing GPU. Ketika perhitungan ini memiliki nilai activation dan intermediate output yang sangat besar sehingga kebutuhan memory-nya tidak bisa dipenuhi oleh 1 GPU, kita perlu kombinasikan dengan metode-metode lainnya.

### 1.1.3 Mixed precision training

Setiap data dalam sebuah komputer direpresentasikan menggunakan barisan biner (hanya bernilai 0 dan 1). Jumlah bit yang menyusun barisan biner berhubungan langsung dengan:
1.  Total memory yang diperlukan untuk menyimpan data
2.  Beban komputasi untuk melakukan operasi pada data tersebut

Mixed precision training menggunakan representasi yang lebih compact utamanya untuk parameter-parameter suatu model agar penyimpanan dan komputasi lebih efisien.

Umumnya Deep Learning menggunakan tipe data **_float32_** yang terdiri dari 32 bit. Setiap bit ini dikelompokkan menjadi tiga bagian, yaitu ***sign, exponent,*** dan ***mantissa.***

![IEEE 754](https://lms.sdmdigital.id/pluginfile.php/635410/mod_page/content/28/image5.png)
**Gambar 9. Representasi sebuah float menggunakan 32-bit berdasarkan standar IEEE 754**

Untuk menghitung nilai float yang direpresentasikan oleh tipe tersebut, digunakan rumus **Perhitungan nilai float 32 bit** berikut:

$$ \text{Nilai} = (-1)^S \times 2^{(E-127)} \times \left( 1 + \sum_{i=1}^{23} b_i \times 2^{-i} \right) \tag{Eq. 1} $$

Dimana:
*   **S** adalah nilai bit sign, jika bitnya bernilai 1, maka S bernilai 1, dan sebaliknya. S menentukan apakah suatu bilangan float merupakan bilangan positif atau negatif.
*   **E** merupakan nilai integer dari angka binary 8 bit dalam exponent.
*   **i** merupakan indeks seluruh bilangan pada mantissa, bermulai dari 1 untuk bit paling kanan.
*   **bᵢ** merupakan nilai bit pada setiap indeks mantissa.

Untuk melakukan mixed precision training, digunakan representasi bilangan yang lebih compact (lebih sedikit jumlah bitnya):
*   ***Float16:*** 1 bit sign, 5 bit untuk exponent dan 10 bit untuk mantissa.
*   ***BFloat16:*** 1 bit sign, 8 bit untuk exponent dan 7 bit untuk mantissa, banyak digunakan dalam Deep Learning.
*   ***NVFP4:*** 1 bit sign, 2 bit untuk exponent dan 1 bit untuk mantissa, baru baru ini dikenalkan oleh NVIDIA bersamaan dengan arsitektur blackwell mereka.

Satu hal yang perlu diingat adalah representasi yang lebih compact lebih tidak presisi sehingga bisa mempengaruhi performa model. Walaupun begitu, perbedaannya dengan full precision (fp32) seringkali tidak begitu besar, dan bisa diterima mengingat keuntungan efisiensinya.

---

### 1.1.4 Training efisien menggunakan Accelerate

Implementasi distributed training itu sulit. Tapi anda tidak perlu khawatir karena sudah banyak library yang bisa langsung digunakan untuk melakukan distributed training, beberapa di antaranya bisa dilihat pada tabel berikut:

**Tabel 2. Beberapa library/framework yang digunakan untuk distributed training LLM**

| Library/Framework | Pengembang | Fitur |
| :--- | :--- | :--- |
| Megatron-LM | NVIDIA | Model Parallelism untuk skala parameter besar |
| DeepSpeed | Microsoft | ZERO, Mixed Precision, CPU/NVME offloading (meletakkan parameter atau hasil komputasi di luar GPU untuk sementara) |
| Torch Titan | Meta AI | FSDP2, Model Parallelism, Mixed Precision |
| Accelerate | HuggingFace | Orkestrasi dan integrasi library-library milik Hugging Face dengan fitur dan framework lainnya (FSDP, DeepSpeed) |

Semua library di atas kompatibel dengan pytorch. Dari sekian banyak library tersebut, salah satu yang paling mudah digunakan adalah Accelerate. Accelerate memiliki cara integrasi yang mudah, serta mendukung berbagai macam metode distributed training baik secara native maupun melalui integrasinya dengan framework lain.

Bayangkan anda memiliki kode training loop seperti ini:

```python
device = "cuda"
model.to(device)

for batch in training_dataloader:
    optimizer.zero_grad()
    inputs, targets = batch
    inputs = inputs.to(device)
    targets = targets.to(device)
    outputs = model(inputs)
    loss = loss_function(outputs, targets)
    loss.backward()
    optimizer.step()
    scheduler.step()
```

Integrasi *Accelerate* bisa dilakukan dengan mengubah hanya beberapa bagian dari kode:

```python
from accelerate import Accelerator
accelerator = Accelerator()
...
device = accelerator.device
model, optimizer, training_dataloader, scheduler = accelerator.prepare(
    model, optimizer, training_dataloader, scheduler
)

for batch in training_dataloader:
    optimizer.zero_grad()
    inputs, targets = batch
    outputs = model(inputs)
    loss = loss_function(outputs, targets)
    accelerator.backward(loss)
    optimizer.step()
    scheduler.step()
```

Perubahan yang perlu dilakukan bisa dilihat pada bagian-bagian yang ditebalkan. Anda hanya perlu:
1.  Import dan inisialisasi Accelerator
2.  Memanggil fungsi prepare dengan input seluruh kebutuhan training anda
3.  Lalu memanggil accelerator.backward alih alih loss.backward

Tapi mungkin anda bertanya-tanya, konfigurasi untuk distributed trainingnya di mana? Sebelum menjalankan kode training menggunakan accelerate, anda perlu melakukan konfigurasi dulu dengan menjalankan perintah `accelerate config` di terminal anda. Lalu anda bisa melihat program interaktif yang bisa anda ikuti untuk menentukan konfigurasi distributed training.

```text
> accelerate config
In which compute environment are you running?
> This machine
Which type of machine are you using?
Please select a choice using the arrow or number keys, and selecting with enter
    No distributed training
    multi-CPU
    multi-XPU
    multi-HPU
 * multi-GPU
    multi-NPU
    multi-MLU
    multi-SDAA
    multi-MUSA
    TPU
```

Berikut merupakan contoh konfigurasi yang didapatkan dari menjalankan konfigurasi di atas:

```yaml
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: FSDP
downcast_bf16: 'no'
dynamo_config:
  dynamo_backend: TENSORRT
  dynamo_mode: max-autotune
  dynamo_use_dynamic: true
  dynamo_use_fullgraph: true
  dynamo_use_regional_compilation: true
enable_cpu_affinity: false
fsdp_config:
  fsdp_activation_checkpointing: true
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_cpu_ram_efficient_loading: true
  fsdp_offload_params: true
  fsdp_reshard_after_forward: true
  fsdp_state_dict_type: SHARDED_STATE_DICT
  fsdp_version: 2
machine_rank: 0
main_process_ip: 192.168.1.1
main_process_port: 7878
main_training_function: main
mixed_precision: bf16
num_machines: 2
num_processes: 4
parallelism_config:
  parallelism_config_cp_comm_strategy: allgather
  parallelism_config_cp_size: 512
  parallelism_config_dp_replicate_size: 2
  parallelism_config_dp_shard_size: 2
  parallelism_config_tp_size: 2
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

Perlu diingat: anda perlu akses ke mesin yang memiliki lebih dari 1 GPU untuk bisa menggunakan *distributed training*, sehingga untuk sekarang, metode-metode di atas bisa dipahami dulu saja untuk digunakan kedepannya.

---

### 1.1.5 Seberpengaruh apa trik-trik efisiensi ini?

Mari mulai dari yang paling sederhana, yaitu DP. Bayangkan anda memiliki model dengan 3 layer, lalu dilatih menggunakan DP pada 2 GPU. Kita tahu di akhir training step, DP perlu melakukan sinkronisasi dulu, sebelum melakukan update parameter.

![Timeline DP](https://lms.sdmdigital.id/pluginfile.php/635410/mod_page/content/28/image55.jpg?time=1768376953441)
**Gambar 10. Ilustrasi timeline data parallelism untuk sebuah training step di 2 GPU**

Masalahnya, ini sangat membuang waktu. Umumnya sinkronisasi langsung dimulai ketika sudah ada setidaknya 1 gradien yang selesai dihitung untuk meminimalisir waktu tunggu. Contohnya seperti berikut:

![Overlapping Sync](https://lms.sdmdigital.id/pluginfile.php/635410/mod_page/content/28/image9.jpg?time=1768376972024)
**Gambar 11. Ilustrasi DP dengan sinkronisasi yang dilakukan begitu gradien baru tersedia**

Keuntungan melakukan distributed training berupa faktor percepatan atau disebut gain bisa dihitung dengan rumus **Faktor gain dari data parallelism** berikut:

$$ \text{Gain} = N \times \frac{T_{compute}}{T_{compute} + T_{communication} - T_{overlap}} \tag{Eq. 2} $$

Dimana:
*   **N** adalah jumlah GPU
*   **T_compute** adalah waktu setiap GPU untuk menyelesaikan seluruh training step
*   **T_communication** adalah waktu total yang diperlukan untuk komunikasi antar GPU dalam proses sinkronisasi
*   **T_overlap** adalah waktu "curi start" proses sinkronisasi

Walaupun idealnya menggunakan N GPU untuk melakukan training secara paralel bisa meningkatkan kecepatan training sebanyak N kali lipat, ada waktu yang diperlukan untuk komunikasi antar GPU, sehingga realitanya gain akan selalu lebih kecil dari N. Metode lain lebih sulit dihitung karena pola komunikasinya lebih rumit, tapi umumnya bisa diamati faktor peningkatan yang bervariasi mulai dari 1.2-4 kali lipat, tergantung ukuran model, konfigurasi distributed training, serta faktor lainnya seperti batch size yang digunakan.

Berbeda dengan metode efisiensi berbasis scaling ke banyak GPU, metode efisiensi yang berbasis menggunakan presisi yang lebih rendah untuk representasi parameter tidak begitu terpengaruh dengan overhead dari komunikasi. Dengan berpindah dari full-precision ke mixed-precision sendiri bisa memberikan percepatan tambahan hingga 3 kali lipat.

Perlu diingat bahwa menumpuk berbagai macam metode distributed training bisa berakibat buruk, sehingga perlu dipilih secara hati-hati melalui eksperimen. Secara umum, prinsip memilih metode distributed training adalah sebagai berikut:
1.  Selalu Mulai dari DP.
2.  Kalau sebelum training (baru memasukkan parameter model ke GPU) bahkan sudah OOM (out of memory), coba terapkan PP, TP atau EP kalau arsitektur modelnya menggunakan MoE.
    *   PP kalau layer model terlalu banyak
    *   TP kalau ada layer tertentu yang ukurannya terlalu besar, jadi walaupun sudah dibagi dengan PP masih tidak cukup
    *   EP kalau menggunakan MoE
3.  Kalau saat training masih tidak cukup, misal context window yang digunakan terlalu besar, coba tambahkan TP atau CP.

Proses ini perlu eksperimen berulang kali sampai didapatkan konfigurasi yang bisa menjalankan training dengan sukses untuk beberapa batch.

---

### Mini Quiz
*(Kuis interaktif tersedia di platform LMS)*

---

> **Refleksi Singkat:**
> 
> Training LLM umumnya memiliki skala yang besar, sehingga banyak framework yang dikembangkan untuk melakukan proses training secara lebih efisien yang mengimplementasi berbagai macam trik seperti distributed training dan mixed precision training. Terlihat juga kalau kompleksitasnya bisa beragam, dimana berbagai macam metode paralelisme bisa dicampur satu sama lain.
> 
> Beberapa pihak dalam dunia AI baik dari akademi maupun industri memberikan detil gambaran mengenai proses training yang mereka lakukan, misalnya:
> - [HuggingFace Smol Training Playbook](https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook)
> - [Uber Optimization](https://www.uber.com/en-ID/blog/open-source-and-in-house-how-uber-optimizes-llm-training/)
> - [Meta Llama 3 Blog](https://ai.meta.com/blog/meta-llama-3/)
> 
> Coba amati dan cari detail mengenai strategi efisiensi yang mereka gunakan untuk lebih memahami bagaimana sebenarnya LLM dilatih pada dunia nyata.