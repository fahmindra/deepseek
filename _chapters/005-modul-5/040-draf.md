---
slug: modul-5-4
title: modul-5-4
---
# 1.4 Teknis Pelaksanaan Post-training dan Evaluasinya

---

Setelah mendapatkan base model dengan performa yang memuaskan, saatnya melanjutkan dengan post training untuk mempertajam kemampuan spesifik dari sebuah model, misalnya kemampuan mengikuti instruksi dan alignment. Utamanya post training dibagi menjadi dua bagian yaitu SFT (supervised fine tuning) yang umumnya berbentuk instruction tuning dan preference alignment.

## 1.4.1 Apa yang membedakan kedua tahapan ini?

Walaupun sama-sama membutuhkan supervisi melalui pelabelan, kedua tahapan ini melibatkan “gaya” supervisi yang berbeda. Supervised Fine Tuning dilakukan untuk perilaku LLM yang bisa dideskripsikan secara langsung seperti apa perilaku yang diharapkan, terutama untuk kemampuan yang bisa dikonfirmasi langsung apakah respon maupun langkah-langkah dalam memberikan respon tersebut sudah sesuai dengan ekspektasi yang ditetapkan oleh pelabel manusia. Misalnya kita ingin model memiliki kemampuan mengikuti instruksi yang baik, maka kita berikan data berisi pasangan instruksi serta cara menjalankannya. Hal ini berlaku untuk hal-hal yang bisa dicontohkan secara langsung yang tingkat kebenarannya jelas dan objektif, sehingga bisa diekspresikan secara efektif melalui pasangan input dan output yang seharusnya.

Banyak bentuk SFT yang bisa dilakukan, mulai dari instruction tuning, melatih model untuk bercakap dan menjawab pertanyaan dengan baik, hingga kemampuan khusus seperti menggunakan tool dan reasoning yang akan lebih detail dijelaskan pada bab selanjutnya. Intinya adalah berbeda dengan pretraining, tahap ini berfokus memberikan contoh secara langsung seperti apa output yang diharapkan dari sebuah input, bukan hanya belajar “melanjutkan” teks saja.

Berbeda dengan SFT, ketika kita berbicara mengenai preference alignment, apa yang menjadi preferensi dan derajat dari kemampuan mengikuti preferensi tersebut jauh lebih susah untuk dikuantifikasi secara numerik. Tahap preference alignment memerlukan paradigma yang berbeda dalam pelabelan, dimana selain harus memastikan ada persetujuan bersama mengenai kriteria yang diharapkan, bentuk labelnya bukanlah binary contoh baik vs contoh salah, tapi benar benar memberikan label preferensi. Dari satu input, diberikan banyak output, lalu dipilih atau diurutkan berdasarkan kesesuaian dengan kriteria tersebut. Label preferensi selain memberikan informasi “mana yang paling sesuai”, juga memberikan informasi kuantifikasi secara tidak langsung dimana urutan preferensi juga memberikan gambaran “tingkat kesesuaian” dari setiap output.

Kalau SFT memberikan label dalam bentuk contoh input dan output yang harus diikuti oleh model untuk mempelajari suatu kemampuan, preference alignment memberikan label dalam bentuk kesesuaian dengan sebuah kriteria preferensi untuk “mendorong” model agar semakin mampu memberikan respon sesuai preferensi.

---

## 1.4.2 Batasan SFT dan kapan Reinforcement Learning dibutuhkan

**RL bukan hanya untuk preference alignment.**
Mungkin sudah tidak asing istilah RLHF atau Reinforcement Learning from Human Feedback yang kerap disangkutpautkan dengan preference alignment untuk tujuan seperti mengatur gaya bahasa, mengajarkan model agar tidak toxic, ramah dan lain sebagainya. Yang perlu dipahami adalah tidak selamanya RL hanya digunakan untuk melakukan preference alignment. Seperti yang akan dibahas nanti, RL merupakan “senjata” yang manjur ketika sudah semakin sulit memberikan “contoh” kepada model bagaimana seharusnya ia memberikan respon sesuai suatu kriteria kemampuan tertentu.

**Batasan SFT**
Secara intuitif, SFT secara langsung memberikan model sebuah contoh yang bisa ditiru untuk meningkatkan suatu kemampuan spesifik. Dengan memberikan contoh yang bervariasi baik itu dari segi skenario maupun domain, harapannya adalah sebuah model dapat mempelajari bagaimana caranya memiliki suatu kemampuan dengan cara meniru contoh-contoh ini dan melakukan generalisasi terhadap konsep yang berkait dengan kemampuan tersebut. Walaupun terbukti cukup efektif untuk melatih banyak kemampuan yang dibutuhkan, pendekatan ini memiliki kelemahan. Model hanya belajar meniru, dan generalisasi kemampuannya bergantung variasi contoh yang bisa diberikan. Pada kasus tertentu apalagi yang begitu spesifik, tidak mungkin semua skenario dapat dimasukkan, atau mungkin terlalu rumit dan mahal untuk dilakukan. Selain itu karena model hanya belajar meniru, arah pembelajaran yang baik untuk meningkatkan kemampuan model sulit diketahui karena tidak ada gambaran “seberapa salah” atau “seberapa sudah hampir benar” jawaban dari model.

**Perbedaan RL dengan SFT, alasan kenapa bisa lebih efektif**
Berbeda dengan SFT, Reinforcement Learning memiliki paradigma yang berbeda. Dibanding secara langsung diberikan contoh bagaimana model harus berperilaku, RL mendefinisikan pola trainingnya dalam bentuk memaksimalkan reward. Sebuah model (LLM) akan berinteraksi dengan lingkungannya yang akan memberikannya reward untuk setiap aksinya. Reward ini kemudian akan digunakan model untuk menentukan bagaimana caranya mengubah parameter yang tepat untuk memaksimalkan reward ini.

Paradigma ini memungkinkan proses training yang hanya perlu diberikan cara untuk “menilai” kualitas jawaban dari sebuah model, alih alih mencoba membuat label yang dapat menangani berbagai macam skenario. Sebuah model / cara menilai kualitas ini disebut sebagai Reward Model yang menjadi salah satu kunci utama dari RL. Cara ini bermacam, bisa berupa sesuatu yang mudah diimplementasi dan diverifikasi secara konkrit hingga sesuatu yang lebih abstrak yang membutuhkan pemodelan. Contohnya untuk menilai kode, bisa dinilai dari segi apakah kodenya bisa di-compile, apakah ada syntax error, dan apakah bisa menghasilkan jawaban yang diinginkan. Akan tetapi untuk sesuatu yang lebih rumit, misalnya menilai kualitas puisi, menilai gaya bahasa, dan kriteria lainnya yang abstrak, perlu pemodelan untuk menentukan penilaian terhadap kriteria tersebut, yang umumnya diimplementasi menggunakan reward model.

Akibatnya, kita tidak lagi terbatas pada kemampuan untuk menyiapkan contoh yang tepat dan beragam, dan model bisa melakukan iterasi sendiri untuk terus meningkatkan kemampuannya.

---

## 1.4.3 Tantangan dari melakukan RL dan kenapa sebaiknya RL menjadi pilihan terakhir

Meskipun terkesan begitu powerful untuk melatih sebuah model, RL memiliki beberapa hal yang menjadikannya sulit untuk dimanfaatkan, yakni:

1.  RL membutuhkan kompleksitas engineering yang lebih tinggi. Proses training menggunakan RL memerlukan model tambahan, infrastruktur spesifik, serta keperluan teknis lainnya untuk mengimplementasi berbagai macam kebutuhan untuk memastikan RL berjalan lancar.
2.  RL memerlukan biaya komputasi yang lebih besar disertai dengan proses yang lebih rumit. RL melibatkan model melakukan banyak iterasi untuk melakukan eksplorasi untuk mencari dan memahami bagaimana caranya terus meningkatkan reward.
3.  RL bisa tidak stabil dan bisa gagal dengan cara yang tidak terduga.
4.  RL bergantung seberapa baik desain reward model agar bisa efektif.

Berbagai macam kompleksitas tambahan ini, dan fakta bahwa RL bisa diterapkan untuk mengembangkan lebih jauh kemampuan model yang sebelumnya sudah dilatih, menjadikan RL sebaiknya dilakukan sebagai tahap terakhir untuk meningkatkan kemampuan model dalam aspek tertentu.

---

## 1.4.4 Mempersiapkan Evaluasi

Sama seperti semua tahap sebelumnya, yang harus pertama dilakukan sebelum melatih model adalah menetapkan evaluasi yang akan digunakan untuk menilai kinerja model. Selain menilai kinerja model, evaluasi juga akan digunakan untuk memahami peningkatan kemampuan model pada kemampuan-kemampuan tertentu dibandingkan dengan base model yang digunakan untuk melakukan post- training.

Pada tahap ini, baseline bisa menggunakan performa base model disertai kemampuan model lain yang sejenis atau yang sebanding, misal dari segi jumlah parameter. Evaluasi ini penting untuk memastikan dua hal:
*   Post training benar benar memberikan peningkatan performa yang berarti, dan
*   Post training tidak memiliki pengaruh yang buruk terhadap kemampuan model bila dibandingkan dengan evaluasi dari base model hasil pretraining.

Metrik performa yang sebelumnya digunakan untuk melakukan evaluasi pada pretraining bisa digunakan untuk evaluasi post training. Akan tetapi tetap diperlukan dataset khusus untuk mengukur performa-performa spesifik kemampuan yang ingin dilatih secara eksplisit. Untuk domain yang cukup niche / spesifik, tahap ini memerlukan pengumpulan data manual yang dibantu oleh domain expert. Tabel berikut menyajikan perbedaan antara evaluasi pretraining dengan post training.

**Mempersiapkan Data**

Tahap ini memiliki aturan yang sama dengan persiapan data untuk pretraining, yang membedakan adalah skala dan bentuk datanya. Tabel berikut memberikan perbedaan antara data yang diperlukan untuk pretraining dan post training:

**Tabel 9. Perbedaan keperluan data berbagai tahap training LLM**

| Aspek Data | Pretraining | Data SFT | Data Preference Alignment |
| :--- | :--- | :--- | :--- |
| **Skala** | Skala internet (web-scale), 300B-15T token pada model populer | 5-20B token, menurut rilis dari beberapa LLM komersil | 100M-1B token |
| **Tujuan** | Mengajarkan kemampuan pemodelan bahasa, termasuk pengetahuan dan pemahaman mengenai berbagai konsep | Mengajarkan LLM kemampuan spesifik (mengikuti instruksi, percakapan, menjawab pertanyaan) | Mengajarkan LLM cara merespon sesuai kriteria preferensi tertentu |
| **Bentuk data** | Teks mentah | Pasangan input dan output disertai instruksi | Pasangan output untuk input yang sama (dengan instruksi) |
| **Bentuk data** | Teks mentah | Pasangan input dan output disertai instruksi | Pasangan output untuk input yang sama (dengan instruksi) |

Karena skalanya yang lebih kecil dan seberapa lebih dekatnya dengan keperluan khusus yang langsung berhubungan dengan tujuan pengembangan model, pengumpulan data untuk post-training bisa dilakukan dengan lebih berfokus pada kualitas melalui proses yang lebih manual, walaupun juga terbukti data sintetis hasil generate LLM bisa memberikan hasil yang baik, seperti misalnya dataset Alpaca milik stanford.

**Special token untuk post-training dan chat template**
Sama seperti pretraining, post training juga memerlukan special token. Selain untuk melakukan packing yang umum dilakukan untuk SFT (tapi tidak untuk preference alignment), special tokens khusus digunakan untuk post-training, khususnya pada SFT tergantung kemampuan apa yang sedang dilatih. Salah satu bentuk special token yang umum digunakan adalah satu set token yang digunakan untuk memberikan LLM petunjuk mengenai pembagian semantik dari setiap komponen inputnya. Contohnya adalah sebagai berikut, misalnya kita sedang melatih LLM untuk berinteraksi dengan user untuk menjawab suatu pertanyaan / instruksi, kita akan menggunakan token khusus untuk memisahkan bagian-bagian teks yang akan diproses LLM, misalnya untuk system message, instruksi dari user, dan bagian khusus untuk jawaban / respon dari LLM.

```text
<|system|> Ikuti instruksi dari pengguna untuk memberikan output yang dia inginkan!
<|user|> Tolong jelaskan apa itu tranformer dalam konteks LLM
<|assistant|> Transformer adalah...
```

Setiap token ini membatasi / memberi petunjuk ke LLM bahwa setiap input ini merupakan bagian yang berbeda. Semakin kompleks interaksi yang kita inginkan, misalnya meminta LLM membaca dokumen, atau meminta LLM memperhatikan sebuah gambar, atau menulis kode, kita perlu menggunakan special token yang dipilih secara tepat untuk membentuk format input yang lebih mudah dipahami agar LLM lebih mudah melaksanakan tugasnya.

Implementasi special token ini berbeda-beda tergantung pengembang dan biasanya special token ini juga digunakan dalam proses inference untuk memformat input ke LLM agar mengikuti bentuk yang digunakan untuk melatihnya saat training. Format ini disebut sebagai chat template. Berikut adalah beberapa chat template yang umum digunakan dan banyak diadopsi di industri:

**Tabel 10. Berbagai jenis chat template pada industri**

| Template | Kemampuan | Catatan |
| :--- | :--- | :--- |
| **ChatML** | Pengaturan peran, penggunaan tools | Paling simpel dan mungkin cukup untuk penggunaan umum. |
| **Qwen3** | Pengaturan peran, penggunaan tools, reasoning | Mendukung reasoning / penalaran. |
| **LLaMA 3** | Pengaturan peran, penggunaan tools | Punya spesial token untuk memberikan kemampuan khusus, misalnya kemampuan menulis dan menjalankan sendiri kode python |

Memilih chat template yang tepat dengan pilihan special token yang baik dapat memberikan LLM efektivitas tambahan untuk mempelajari dan mengeksekusi kemampuan-kemampuan khusus.

**Contoh Penerapan Chat Template**
Kode berikut menunjukkan bagaimana chat template yang ada di sebuah tokenizer bisa digunakan.

```python
from datasets import load_dataset
from transformers import AutoTokenizer


MODEL = "Qwen/Qwen3-0.6B"   # bebas pilih saja yang mau dipakai chat templatenya


# Load Alpaca. Kalau tidak ingin download,
# ganti saja langsung sampled_from_train
# dengan list dictionary yang berisi
# key input, output, dan instruction
dataset = load_dataset("tatsu-lab/alpaca")


# ambil beberapa data random
sampled_from_train = dataset["train"].shuffle().select(range(10))


# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True)


# Format and tokenize using chat template
def format_with_chat_template(example):
    messages = [
        {"role": "system", "content": example["instruction"]},
        {"role": "user", "content": example["input"]},
        {"role": "assistant", "content": example["output"]},
    ]
    conversation = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return conversation


for sample in sampled_from_train:
  # untuk melihat bentuk teksnya yang sudah
  # dibentuk menggunakan template
  formatted_message = format_with_chat_template(sample)
  print(formatted_message)
  print("==="*30)
```

---

## 1.4.5 Framework untuk melakukan post-training

Sama seperti pre-training, banyak framework yang bisa digunakan untuk melakukan post-training yang sudah mengimplementasi metode-metode yang umum digunakan, sehingga pengembang LLM hanya perlu membuat konfigurasi saja tanpa memikirkan implementasinya. Biasanya framework yang tersedia menunjang berbagai kapabilitas, mulai dari fungsionalitas utama seperti jenis post training yang didukung, juga konfigurasi tambahan terutama untuk meningkatkan efisiensi training. Walaupun tersedia berbagai macam framework, yang membedakan hanyalah kelengkapan fitur dan cara penggunaannya. Modul ini akan menggunakan framework **TRL** milik huggingface, akan tetapi prinsip dari melakukan post-training akan sama dan mudah diadaptasikan ketika menggunakan framework lain.

---

## 1.4.6 Supervised Fine Tuning (SFT)

Setelah mempelajari apa saja yang perlu dipersiapkan untuk melakukan post training, tahap pertama yang umum dilakukan pada post training adalah SFT. Mengapa? Alasannya adalah sebagai berikut.

*   SFT membutuhkan komputasi yang lebih rendah dibanding preference alignment yang umumnya berbasis RL,
*   Tahap SFT lebih stabil dan lebih simpel untuk meningkatkan kemampuan model pada suatu bidang / task tertentu dibandingkan RL, dan yang paling utama
*   SFT sudah dapat memberikan kemampuan yang diinginkan dengan cara setup dan teknis yang lebih sederhana sebelum dilanjutkan dengan preference alignment, sehingga lebih efisien untuk mendapatkan model dengan kemampuan tertentu

Tidak banyak perbedaan antara SFT dengan pretraining, mulai dari loss yang digunakan, kebutuhan pipeline evaluasi dan cara iterasinya hingga infrastrukturnya dan teknis yang perlu dilakukan selama training. Akan tetapi masih ada beberapa hal yang berbeda yang perlu dilakukan selama SFT:
*   Melakukan response masking. Karena input SFT sudah ditentukan dan yang ingin disesuaikan adalah outputnya, pada saat menghitung gradien, kita hanya ingin menghitung gradien berdasarkan respon dari LLM.
*   SFT memerlukan learning rate yang lebih rendah dibandingkan saat pretraining. Logikanya, model sudah memiliki pemahaman yang kuat terhadap bahasa yang disimpan dalam parameternya, dan kita tidak ingin pemahaman ini terlalu dirubah. Umumnya digunakan learning rate sekitar ~10x lipat lebih kecil dibandingkan pretraining

Selain perbedaan di atas, teknis yang perlu dilakukan tidak jauh berbeda.

---

## 1.4.7 Preference Alignment

Setelah mendapatkan model dengan kemampuan spesifik domain yang memuaskan, tahap terakhir adalah memastikan model ini memiliki perilaku maupun sifat yang sesuai dengan kriteria tertentu, dari segi cara menjawab, gaya bahasa, serta jenis preferensi lainnya yang lebih berhubungan dengan “bagaimana cara model menyampaikan respon” dibandingkan “respon apa yang diberikan model”. Selain preferensi berbasis kriteria tertentu seperti kriteria etis / moral, preference alignment juga bisa digunakan untuk mengembangkan kemampuan model seperti misalnya memberikan reasoning yang lebih baik.

Teknis pengumpulan data, jenis data, serta berbagai metode implementasinya sudah sebelumnya disebutkan. Dalam modul ini kita akan mencoba membahas mengenai berbagai pilihan yang perlu dilakukan serta detil spesifik teknik pelaksanaan proses ini yang terdiri dari memilih metode optimasi dan hyperparameter.

---

## 1.4.8 Metode apa yang dipilih? Yang berbasis RL atau yang tidak?

Sebelumnya sudah disinggung dua pendekatan untuk melakukan preference alignment, yaitu metode seperti PPO yang berbasis RL dan metode seperti DPO yang bisa di-frame sebagai task SFT.

RL memiliki intrikasi yang membuatnya tidak stabil dan sulit untuk mengaturnya agar memberikan hasil yang diinginkan, serta membutuhkan lebih banyak resource dibandingkan dengan SFT. Sebagai langkah pragmatis, biasanya DPO lebih disukai.

---

## 1.4.9 Penentuan hyperparameter

Ada dua hyperparameter penting yang perlu ditentukan lagi, yaitu learning rate dan batch size. Batch size sendiri menggunakan cara yang sama yaitu melalui eksperimen untuk penentuan batch size yang optimal. Selain itu, pada tahap sebelumnya yaitu SFT, sudah didapatkan model yang memiliki kemampuan tertentu. Kita tidak menginginkan kemampuan ini berubah secara drastis sehingga perlu digunakan learning rate dengan skala yang lebih kecil. Biasanya karena SFT juga sudah menghasilkan model yang “baik” yang tidak ingin terlalu banyak diubah lagi perilakunya, digunakan learning rate yang lebih kecil dari SFT, biasanya 10-100x lebih kecil dibandingkan yang digunakan saat SFT. Pada praktiknya, sekitar 5-20x lebih kecil sudah memberikan learning rate yang optimal.

---

## 1.4.10 Contoh implementasi kode

Untuk melakukan post-training, TRL menyediakan implementasi yang mudah. Berikut adalah contoh implementasi SFTTrainer:

```python
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset


dataset = load_dataset("alpaca", "train")


def preprocess_function(example):
    # beberapa data di dataset alpaca hanya memiliki instruction, tidak ada input
    # handlingnya beragam, di sini sederhana saja. Kalau tidak ada input,
    # instruction dianggap jadi input user langsung saja
    if example["input"]:
        return {
            "messages": [
                {"role": "system", "content": example["instruction"]},
                {"role": "user", "content": example["input"]},
                {"role": "assistant", "content": example["output"]},
            ]
        }
    return {
        "messages": [
            {"role": "user", "content": example["instruction"]},
            {"role": "assistant", "content": example["output"]},
        ]
    }


dataset = dataset.map(preprocess_function, remove_columns=["instruction", "input", "output"])


trainer = SFTTrainer(
    model="Qwen/Qwen3-0.6B",
    args=SFTConfig(
        output_dir="Qwen3-0.6B-Instruct",
        chat_template_path="HuggingFaceTB/SmolLM3-3B",
    ),
    train_dataset=dataset,
)
trainer.train()
```

Hal yang sama juga bisa dilakukan untuk DPO. Berikut adalah contoh DPO menggunakan trl:

```python
# train_dpo.py
from datasets import load_dataset
from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer


model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
train_dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")


training_args = DPOConfig(output_dir="Qwen3-0.6B-DPO")
trainer = DPOTrainer(model=model, args=training_args, processing_class=tokenizer, train_dataset=train_dataset)
trainer.train()
```

---

### Mini Quiz

*(H5P Placeholder untuk Multiple Choice Quiz)*

> **Refleksi Singkat**
>
> Umumnya dilakukan terlebih dahulu tahap post-training yang supervised sebelum RL. Tapi Deepseek-R1 langsung melakukan RL untuk melatih kemampuan reasoning, tidak melewati tahap SFT terlebih dahulu. Coba pahami mengapa ini bisa bekerja dan buat analisis, apakah hal yang sama bisa dilakukan pada banyak hal lainnya, atau ini unik untuk reasoning saja?
>
> Silahkan baca [paper](https://arxiv.org/abs/2501.12948) yang mereka rilis untuk melakukan eksplorasi lebih lanjut.