---
slug: modul-5-5
title: modul-5-5
---
# 1.5 Pengembangan Kemampuan Lanjutan: Tool-use Augmentation dan Reasoning

LLM modern umumnya memiliki dua kemampuan tambahan selain mengikuti instruksi maupun berinteraksi dengan manusia, yaitu tool use dan reasoning yang sudah disinggung sebelumnya. Kapabilitas ini perlu dilatih secara eksplisit dimana detail dari proses latih ini melibatkan beberapa komponen tambahan, mulai dari pengumpulan data khusus, pertimbangan dalam membuat tokenizer, penggunaan special token hingga proses training itu sendiri. Sebelum memahami berbagai detil implementasi, mari kita pahami dulu kedua kapabilitas ini.

## 1.5.1 Tool Use

Tool use merupakan kemampuan sebuah LLM untuk menggunakan sebuah tool, atau bisa dianggap seperti sebuah fungsi yang didefinisikan di luar LLM. Contohnya adalah kemampuan untuk mencari informasi di internet, kemampuan untuk mengakses database, menulis kode dan menjalankannya, dan banyak kemampuan lainnya yang mungkin sering ditemukan dalam aplikasi LLM yang digunakan sehari-hari.

![Aplikasi LLM Tool Use](https://lms.sdmdigital.id/pluginfile.php/635419/mod_page/content/22/image15.png)
**Gambar 44. Aplikasi LLM populer memiliki kemampuan untuk menggunakan “tool”. Claude bisa melakukan searching di internet untuk membantu mencari informasi.**

**Bagaimana caranya LLM bisa menggunakan tool?**

Pada praktiknya, tool use ini bukan berarti LLM menjalankan toolnya secara langsung. Yang dilakukan LLM adalah menerima definisi tool yang berisi deskripsi mengenai fungsi serta detail parameter yang dibutuhkan untuk memanggil tool tersebut. Berikut merupakan salah satu format definisi tool yang digunakan dalam API openAI:

```json
{
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the internet about a topic defined by the query.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
            },
            "required": ["query"],
        },
    },
}
```

Sebagai bagian dari instruksi yang diberikan ke LLM, informasi mengenai sebuah tool yang terdiri dari nama, kegunaannya, serta informasi parameter yang dibutuhkan untuk inputnya akan diberikan kepada LLM. Ketika LLM memutuskan untuk menggunakan sebuah tool, LLM akan menghasilkan suatu format yang mendefinisikan tool apa yang perlu digunakan serta nilai nilai parameter input yang diperlukan untuk menggunakan tool tersebut. Biasanya istilah **tool call** digunakan untuk menyebut situasi dimana LLM memutuskan menggunakan sebuah tool.

Informasi ini akan diproses dalam sistem aplikasi AI yang dibangun untuk memanggil implementasi tool tersebut dalam sistem yang biasanya berbentuk sebuah fungsi. Bentuk format ini beragam, tapi umumnya diberikan dalam struktur data yang mudah diproses oleh program komputer, seperti JSON.

Berikut merupakan contoh output tool call dalam API OpenAI:

```json
{
  "id": "function-call-10884871146028053016",
  "function": {
    "arguments": "{\"query\":\"Penyebab jakarta tenggelam\"}",
    "name": "web_search"
  },
  "type": "function"
}
```

## 1.5.2 Reasoning

Sifat autoregressive LLM dapat dimanfaatkan untuk mengarahkan LLM agar memberikan jawaban akhir yang lebih tepat. Reasoning atau penalaran adalah kemampuan LLM untuk “berpikir”, terlebih dahulu memecah pertanyaan / masalah yang diberikan yang setiap bagiannya kemudian dipertimbangkan secara runut dan detail. Hasil dari reasoning ini berupa breakdown masalah secara detail yang akan menjadi referensi untuk LLM menghasilkan jawaban akhirnya.

![Contoh Reasoning ChatGPT](https://lms.sdmdigital.id/pluginfile.php/635419/mod_page/content/22/image61.png)
**Gambar 45. Contoh kemampuan reasoning dalam ChatGPT dengan menggunakan mode “think”**

Sebelum memberikan jawaban, LLM “berpikir” dulu bagaimana cara menjawab pertanyaanya, termasuk mungkin memutuskan informasi apa saja yang perlu didapatkan melalui tool yang tersedia. Secara intuitif, reasoning memaksa LLM untuk menghasilkan dulu output berupa alur berpikir untuk menyelesaikan suatu masalah. Karena LLM autoregressive, artinya output yang dia hasilkan akan digunakan kembali menjadi input hingga dia selesai menghasilkan jawaban, bisa dibilang kita meminta LLM memberikan “konteks” kepada dirinya sendiri.

## 1.5.3 Bagaimana cara melatih kemampuan ini?

Baik tool use maupun reasoning memerlukan beberapa persiapan, yaitu data, special token, serta cara evaluasi, serta proses trainingnya.

**Data dan cara mengumpulkannya:**

1. LLM diberikan definisi tools.
2. Ada input / pertanyaan yang membutuhkan tool call.
3. LLM menentukan tool mana yang perlu digunakan.
4. Pengguna LLM menggunakan informasi untuk menjalankan tool pada sistem.
5. LLM menerima output tool call sebelumnya, lalu memberikan jawaban.

Dataset untuk melatih kemampuan tool use menggunakan format yang terdiri dari tiga bagian:

1. Definisi tool, biasanya disajikan melalui system prompt.
2. Respon tool call berupa JSON berisi nama fungsi dan parameternya yang perlu digunakan.
3. Menghasilkan jawaban akhir menggunakan nilai yang didapatkan dari tool yang digunakan.

Dataset ini juga perlu memperhatikan beberapa skenario:
(1) Tool call gagal (ada error),
(2) Tool call tidak memberikan informasi yang dibutuhkan. Setiap skenario ini perlu dipikirkan dalam mendesain dataset tool use.

Berikut adalah salah satu contoh dataset percakapan yang memberikan contoh tool use:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "Kamu dapat memanggil tool untuk membantu menjawab pertanyaan pengguna."
    },
    {
      "role": "tool_spec",
      "content": {
        "name": "weather_api",
        "description": "Mengecek cuaca pada suatu lokasi",
        "parameters": {
          "city": {"type": "string"}
        }
      }
    },
    {
      "role": "user",
      "content": "Cuaca di Tokyo hari ini bagaimana?"
    },
    {
      "role": "assistant",
      "content": "<tool_call>{\"name\": \"weather_api\", \"arguments\": {\"city\": \"Tokyo\"}}</tool_call>"
    },
    {
      "role": "tool",
      "content": "{\"temperature\": 18, \"condition\": \"Cloudy\"}"
    },
    {
      "role": "assistant",
      "content": "Cuaca di Tokyo hari ini berawan dengan suhu sekitar 18°C."
    }
  ]
}
```

**Persiapan Data Reasoning:**

Data untuk melatih kemampuan reasoning memerlukan 3 bagian:
(1) Input berupa pertanyaan / instruksi,
(2) Contoh Reasoning yang perlu dilakukan oleh LLM,
(3) Jawaban akhirnya.

Ada dua format data yang biasa digunakan, yaitu format yang menjadikan reasoning sebagai bagian dari jawaban akhir, dan format yang menjadikan reasoning sebagai bagian tertentu yang ditandai menggunakan special token, misalnya `<think> </think>`.

```json
// Format 1: Reasoning menyatu dengan jawaban
{
    "messages": [
        {
            "role": "user",
            "content": "Hitung 45 * 22."
        },
        {
            "role": "assistant",
            "content": "Mari kita hitung secara runut. 45 * 22 = 45 * (20 + 2) = 900 + 90 = 990. Artinya jawabannya adalah 990."
        }
    ]
}

// Format 2: Reasoning dengan special token
{
    "messages": [
        {
            "role": "user",
            "content": "Hitung 45 * 22."
        },
        {
            "role": "assistant",
            "content": "<think>Mari kita hitung secara runut. 45 * 22 = 45 * (20 + 2) = 900 + 90 = 990. Artinya jawabannya adalah 990.</think><final>990</final>"
        }
    ]
}
```

## 1.5.4 Special Token dan Chat Template

Setiap LLM biasanya sudah memiliki special token masing-masing untuk keperluan ini disertai dengan chat template yang digunakan untuk memformat baik data latih maupun input dan output model pada saat inference.

**1. Tool Use**
Untuk tool use, perlu special token minimal untuk menandai bagian respon yang merupakan permintaan tool use dari model dan bagian dari instruksi lanjutan yang berisikan hasil dari tool use yang perlu digunakan model sebagai referensi jawaban berikutnya. Format ini beragam, contohnya kita menggunakan `<|tool_use|>` dan `<|tool_response|>`, bentuk datasetnya seperti ini setelah diformat menggunakan chat template:

Contoh penggunaan chat template untuk tool use:

```text
<|system|>
Tool tersedia:

search_web(query: string) -> { result: string }

<|user|>
Cari siapa presiden Indonesia tahun 2020.

<|assistant|>
<tool_use>{"name": "search_web", "arguments": {"query": "presiden Indonesia 2020"}}</tool_use>

<|tool_response|>
{"result": "Presiden Indonesia tahun 2020 adalah Joko Widodo."}

<|assistant|>
Presiden Indonesia pada tahun 2020 adalah Joko Widodo.
```

**2. Reasoning**
Untuk reasoning, special token diperlukan untuk memisahkan reasoning dengan jawaban akhir (untuk format yang memisahkan keduanya). Misalnya digunakan special token `<think></think>` untuk membungkus bagian reasoning dan `<answer></answer>` untuk memisahkan keduanya.

```text
<|user|>
Sebuah toko menjual 12 kotak kue. Setiap kotak berisi 8 kue. Berapa banyak kue semuanya?

<|assistant|>
<think>
1. Hitung total kue: 12 kotak × 8 kue.
2. 12 × 8 = 96.
</think>
<answer>
96
</answer>
```

## 1.5.5 Teknis Evaluasi Tool Use

**Tool Use**
Beberapa metrik berikut biasa digunakan untuk mengukur kemampuan tool use dari sebuah model:

**Tabel 11. Berbagai macam metrik evaluasi kemampuan tool use**

| Nama metrik | Apa yang diukur | Cara mengukur | Implementasi pengukuran |
| :--- | :--- | :--- | :--- |
| **Akurasi tool use** | Akurasi pemilihan tool yang benar | Membandingkan tool yang sebenarnya digunakan dengan yang berada di label | Rule based |
| **Akurasi pengisian parameter** | Nilai dan tipe parameter yang diberikan benar | Membandingkan parameter yang terdapat pada label dengan yang diberikan oleh model | Rule based, fuzzy matching, LLM as a judge |
| **Multi step accuracy** | Akurasi urutan pemanggilan tool ketika dibutuhkan lebih dari 1 tool secara berurutan | Mengukur akurasi setiap step dibanding total step yang ada | Rule based |
| **Final answer accuracy** | Mengukur akurasi jawaban akhir setelah memanggil tool | Perbandingan jawaban akhir dengan label | LLM as a judge, Rule based |
| **Tool use/misuse rate** | Ketepatan keputusan memanggil tool dan pilihan toolnya | Membandingkan penggunaan tool | Rule based |

**Reasoning**

**Tabel 12. Metrik evaluasi kemampuan tool use dan reasoning**

| Nama metrik | Apa yang diukur | Cara mengukur | Implementasi pengukuran |
| :--- | :--- | :--- | :--- |
| **Akurasi jawaban akhir** | Apakah jawaban akhir yang diberikan LLM sesuai dengan referensi | Membandingkan prediksi LLM dengan label | LLM as a judge, exact match |
| **Self consistency** | Seberapa konsisten reasoning dan konklusi berupa jawaban akhir | Generate banyak reasoning berbeda untuk satu input, bandingkan konsistensi dan akurasi akhir | LLM as a judge, rule based |
| **Reasoning quality** | Apakah setiap langkah reasoningnya logis dan tidak lompat lompat | Menggunakan LLM untuk menilai reasoning model melalui perbandingan / skor | LLM as a judge |
| **Faithfulness** | Mengukur apakah hasil reasoning benar-benar dipakai oleh model untuk menghasilkan jawaban akhir | Membandingkan klaim / jawaban akhir dengan poin-poin dalam reasoning | LLM as a judge |
| **Efisiensi** | Jumlah token / langkah yang digunakan untuk melakukan reasoning | Menghitung langkah / total token yang digunakan dalam bagian reasoning | LLM as a judge, Rule based |

## 1.5.6 Proses Training

Selain beberapa perbedaan spesifik kedua kapabilitas ini, proses training untuk keduanya tidak begitu berbeda.

*   **Pertama**, dilakukan SFT dengan aturan yang sama dengan melakukan SFT sebagaimana umumnya.
*   **Kedua**, ketika performa masih kurang memuaskan dan semakin sulit meningkatkan performa dengan memberikan contoh secara langsung, digunakan RL.

Prosesnya tidak begitu berbeda secara kode, karena hanya perlu mengisi bagian template berbeda yang bisa anda baca lebih lanjut pada petunjuk yang diberikan trl pada [dokumentasi berikut.](https://huggingface.co/docs/trl/main/en/dataset_formats)

### Mini Quiz

(Konten kuis interaktif H5P tersedia pada platform LMS)

> **Refleksi Singkat**
>
> Sebelumnya dijelaskan bahwa post training dapat dilakukan baik menggunakan SFT maupun RL. Kalau SFT sudah jelas, anda perlu memberikan secara eksplisit respon apa yang harus diberikan oleh LLM. Bagaimana kalau kasusnya RL untuk tool use dan reasoning? “Feedback” Seperti apa yang perlu diberikan? Bagaimana cara menilai kualitas sebuah tool use maupun reasoning?