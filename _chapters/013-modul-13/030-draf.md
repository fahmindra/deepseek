---
slug: modul-13-3
title: modul-13-3
---
# 1.3 Tracing LLM

---

Untuk menghasilkan gambaran yang lengkap dari bagaimana suatu sistem bekerja, semua catatan peristiwa yang terjadi dalam suatu sistem dalam bentuk log perlu disusun dan dikelompokkan untuk membentuk "cerita penuh" dari proses berjalannya suatu sistem. Ada dua tantangan utama dari proses ini:

1. Log disimpan dalam bentuk rentetan peristiwa. Bagaimana caranya merangkai kembali seluruh peristiwa ini menjadi alur terstruktur yang menggambarkan cara kerja sistem?
2. Suatu sistem bisa terdiri dari banyak bagian yang tersebar yang memiliki eventnya masing-masing. Bagaimana caranya event yang tersebar ini bisa disatukan untuk merangkai sebuah trace?

Sebelumnya telah dipelajari bahwa sebuah structured log juga perlu menyimpan ID dari trace dan span dari suatu sistem. ID trace ini akan digunakan untuk mengelompokkan seluruh log dalam suatu trace, kemudian ID span digunakan sebagai basis untuk menyusun kembali struktur hierarki proses yang menyusun sebuah sistem. Karena penentuan ID ini secara otomatis dilakukan oleh library yang digunakan, yang menjadi penting adalah bagaimana caranya membagi sistem menjadi span.

Kalaupun setiap log tersebar dalam bagian yang berbeda pada sebuah sistem, karena semuanya menuliskan log ke tempat yang sama, maka selama ada ID trace dan span, trace penuh dari sebuah sistem dapat disusun kembali berdasarkan informasi-informasi tersebut. Untuk itu, diperlukan cara untuk mengirimkan "konteks" berupa ID dari Trace dan Span agar bisa digunakan ketika melakukan logging di tempat lain.

---

### 1.3.1 Membagi keseluruhan proses sistem menjadi span

Sebelum melakukan instrumentasi dalam bentuk apapun, sistem terlebih dahulu dibagi menjadi unit-unit kerja yang disebut span. Sebuah span merepresentasikan satuan unit operasi dengan suatu fungsi spesifik dalam sebuah sistem yang ingin diamati dan diukur, seperti pemanggilan API, query database, atau pemrosesan data. Setiap span ini nantinya akan memiliki ID masing-masing yang ditentukan / diberikan secara otomatis oleh library yang digunakan.

Pembagian span tidak selalu mengikuti struktur fisik kode, bisa juga mengikuti alur pemrosesan (logical flow). Misalnya, satu fungsi bisa mengandung beberapa span jika di dalamnya terdapat beberapa operasi penting yang perlu diukur secara terpisah. Span tersusun secara hierarkis dengan relasi parent-child. Satu span dapat memiliki beberapa child span di dalamnya, membentuk struktur tree yang merepresentasikan seluruh alur eksekusi sebuah request.

Contohnya, misal kita memiliki kode seperti berikut yang ingin kita instrumentasi:

```python
def retrieval(query: str):
    embeddings = calculate_embeddings(query)
    retrieval_result = vector_search(query)
    reranked_result = rerank(retrieval_result)
    return reranked_result

def rag_assistant(user_input: str, conversation_id: int):
    conversation_history = retrieve_history(conversation_id)

    query = expand_query(user_input, conversation_history)

    retrieval_result = retrieval(query)

    final_prompt = RAG_FINAL_ANSWER_PROMPT.render(
        user_input, conversation_history, retrieval_result
    )

    final_answer = call_llm(final_prompt).answer

    return final_answer
```

Pertama, kita perlu menentukan dulu apa saja yang ingin diukur. Tentunya kita ingin memastikan dua hal dari sebuah sistem RAG:

1. Proses retrieval memberikan dokumen yang relevan untuk menjawab pertanyaan
2. Jawaban yang diberikan oleh sistem dapat menjawab pertanyaannya dengan baik, faktual, serta dibasiskan pada dokumen yang didapatkan dari retrieval.

Karena ada kedua proses tersebut ingin diamati secara terpisah, sistem bisa kita bagi seperti ini:

```text
Root Span: rag_assistant (2000 ms)
├── Child Span: retrieval (1500 ms)
└── Child Span: generate_answer (500 ms)
```

Dari pembagian ini, kita bisa menjawab pertanyaan seperti:
* Apakah hasil retrievalnya relevan?
* Apakah jawabannya menggunakan informasi hasil retrieval?

Tetapi informasi yang bisa kita dapatkan masih kurang detail. Ketika terjadi kesalahan, pertanyaan-pertanyaan ini perlu dijawab:
* Kalau retrieval buruk, yang salah bagian mana? apakah karena querynya jelek? Atau karena embedding yang digunakan? Atau karena vector search tidak berhasil menemukan data yang tepat?
* Kalau jawabannya salah, apakah karena dokumen yang didapatkan salah? Apakah karena LLM berhalusinasi, padahal sudah diberikan dokumen yang benar? Apakah dokumennya sudah benar-benar masuk ke prompt?

Artinya, setiap bagian di atas perlu dibagi lagi. Kali ini, setiap langkah penting dari kedua span tersebut sudah terbagi dalam span masing-masing:

```text
Root Span: rag_assistant
├── Child Span: retrieval
│   ├── Child Span: expand_query
│   └── Child Span: search_document
│       ├── Child Span: calculate_embeddings
│       ├── Child Span: vector_search
│       └── Child Span: rerank
└── Child Span: generate_answer
    ├── Child Span: render_final_prompt
    └── Child Span: call_llm
```

Dari sini semakin jelas pertanyaan yang bisa kita jawab.
* Kalau LLM memberikan jawaban yang salah, kita bisa mengamati secara terpisah apakah final promptnya dirender dengan baik, jadi bisa menentukan apakah memang informasinya belum diberikan atau ternyata LLM berhalusinasi walaupun sudah diberikan jawaban
* Ketika dokumen yang dikembalikan tidak tepat, kita bisa melihat masalahnya dimana. Misalnya vector_search ternyata sudah menemukan dokumen yang relevan, tapi malah dianggap tidak relevan ketika proses rerank. Atau ketika ternyata vector_search hasilnya kurang baik, kita bisa melihat apakah ada yang salah dengan embeddingnya, dan lain sebagainya.

Ketika membagi sebuah sistem menjadi span, anda harus berhati-hati dalam mencari keseimbangan yang tepat:
* span tidak boleh terlalu lebar, nanti kurang detail / kurang informatif untuk debugging dan analisis.
* span tidak boleh terlalu detail, karena trace nya malah menjadi noisy dan sulit dibaca karena terlalu banyak informasi.

Beberapa prinsip berikut bisa diaplikasikan sebagai best practices untuk membagi sebuah sistem menjadi span:

1. **Satu span satu kesalahan.** Artinya, keseluruhan proses yang dilingkupi sebuah span punya kondisi gagal yang jelas dan mudah dimengerti. Kalau satu span bisa memiliki lebih dari satu bentuk kesalahan yang berbeda-beda penyebabnya, coba dipecah menjadi hierarki saja. Pada praktiknya mungkin agak sulit dilakukan, karena suatu bagian kode bisa memiliki kesalahan logika, semantik, maupun error yang berasal dari sistem, sehingga setidaknya perlu diusahakan walaupun ada banyak kesalahan, bisa dengan jelas diketahui bahwa asalnya dari span tersebut.
    * *Contoh baik:* Proses rerank menjadi satu span sendiri, karena kondisi gagalnya jelas: kalau inputnya ada dokumen yang relevan, tapi outputnya tidak ada, berarti logika reranknya bermasalah.
    * *Contoh buruk:* Workflow yang berisi beberapa tahap pemanggilan tool disatukan menjadi satu span. Kalau jawabannya salah, sulit diketahui apa yang salah. Apakah tool A? Apakah tool B? Apakah kode yang memproses pemanggilan tool? Apakah hasil pemanggilan tool tidak ter-render dengan baik ke dalam prompt akhir?
2. **Tidak semua fungsi perlu menjadi span masing-masing.** Kalau banyak fungsi bisa dipandang sebagai 1 operasi besar saja yang signifikan, bisa disatukan saja menjadi 1 span.
3. **Pisahkan span pada batasan domain.** Misalnya ketika suatu fungsi perlu memanggil database, atau sebuah Agent perlu memanggil tool, pisahkan menjadi span sendiri.
4. **Secara berkala, lakukan refactoring untuk memisahkan atau menggabungkan span.** Walaupun sudah direncanakan dengan baik, bisa jadi seberapa baik granularity dari pemisahan sebuah sistem menjadi span baru diketahui langsung ketika digunakan untuk analisis / debugging.

---

### 1.3.2 Mengatur format dan pembagian span berdasarkan semantic conventions

Untuk menstandarkan pengelompokan, penamaan, dan atribut dari sebuah span, standar OTel mendefinisikan Semantic Conventions. Semantic Conventions (SemCon) menspesifikasikan antara lain nama dan jenis span, instrumen dan unit metrik, serta nama atribut, tipe, dan nilai yang valid yang memberikan konteks semantik pada data ketika proses pengumpulan, produksi, dan konsumsi data tersebut dilakukan.

Berikut contoh span yang mengikuti spesifikasi tersebut:

```json
{
  "gen_ai.operation.name": "generate_content",
  "gen_ai.tool.call.id": "call_ab12cd34ef56",
  "gen_ai.tool.description": "Fetch current weather by city",
  "gen_ai.tool.name": "WeatherAPI",
  "gen_ai.tool.type": "function",
  "gen_ai.tool.call.arguments": {
    "city": "Jakarta",
    "unit": "celsius"
  },
  "gen_ai.tool.call.result": {
    "temperature": 30,
    "condition": "Cloudy"
  }
}
```

Selain struktur yang jelas, penamaan dari setiap field diberikan namespace agar mudah dicari dan difilter, serta memberikan sistem penamaan yang jelas dan bisa diprediksi. Standar OTel tidak hanya mengatur format dan model data untuk span saja, tapi juga mengatur pembagian operasi, format untuk sebuah event, metrik, dan lain sebagainya yang bisa dipelajari lebih lanjut pada [dokumentasinya](https://opentelemetry.io/docs/specs/semconv/gen-ai/).

---

### 1.3.3 Context Propagation

**Apa yang dimaksud dengan konteks?**

Berbeda dengan istilah konteks yang digunakan pada LLM, dalam instrumentasi, konteks merupakan informasi yang berisi detail yang penting untuk menghubungkan berbagai bagian sistem yang dioper dari satu bagian sistem ke bagian lainnya. Dalam kasus observability, konteks umumnya melibatkan mengirimkan informasi seperti TraceID dan SpanID bersamaan dengan komunikasi antar sistem. Tujuannya agar ketika melakukan logging dalam suatu bagian sistem, bagian tersebut bisa mengetahui logging yang dilakukan termasuk dalam suatu trace maupun span tertentu.

**Apa yang dimaksud dengan propagation?**

Dalam sebuah arsitektur aplikasi, ada istilah "boundary", batasan / perpindahan alur pemrosesan dari satu bagian sistem ke bagian lainnya. Misalnya, Frontend mengirim data ke Backend, suatu service memanggil service lainnya, dan lain sebagainya. Untuk sebagian besar sistem, umumnya untuk berkomunikasi melewati batasan ini dilakukan dengan protokol seperti HTTP request, GRPC, dan protokol lainnya. Propagation merupakan proses yang melibatkan komunikasi antar bagian dari suatu sistem untuk mengirimkan suatu konteks. Proses ini melibatkan serialisasi dan deserialisasi serta implementasi pengiriman konteks melalui protokol komunikasi yang digunakan.

![Ilustrasi Propagasi Konteks](https://lms.sdmdigital.id/pluginfile.php/635524/mod_page/content/6/image16.jpg)
**Gambar 7. Ilustrasi propagasi konteks untuk kasus frontend memanggil backend. Propagasi dilakukan melalui HTTP request**

Dengan context propagation, detail penting atau "konteks" dari sebuah operasi atau keberjalanan pemrosesan sebuah request bisa secara konsisten diberikan ke setiap bagian sistem mengikuti alur pemrosesan, sehingga nanti bisa dikorelasi dengan mudah dan dianalisis. Dalam sebagian besar library, hal ini sudah dilakukan secara otomatis, sehingga pada sebagian besar kasus, tidak perlu ditangani manual. Pada submodul selanjutnya, kita akan melakukan eksplorasi secara hands-on salah satu cara melakukan instrumentasi sebuah sistem LLM lengkap hingga tracing menggunakan Langfuse.

---

### Mini Quiz
*(Konten Interaktif H5P)*

> **Refleksi Singkat**
> 
> Anda sudah melihat bagaimana span bisa digunakan untuk membangun seluruh alur eksekusi dari ujung ke ujung. Namun pada sistem nyata, kompleksitas tidak berhenti di situ. Bayangkan Anda diminta merancang observability untuk sistem dengan pola komunikasi yang rumit yang terdiri dari banyak komponen yang masing-masing dirancang oleh tim yang berbeda.
> 
> Jika Anda harus membuat aturan pembagian span yang konsisten untuk seluruh organisasi, bagaimana Anda akan merumuskannya agar tetap seragam tanpa menghambat kreativitas dan kecepatan pengembangan setiap tim?
> 
> Dalam merancang jawabannya, pikirkan hal-hal seperti:
> - Mana yang wajib diikuti? Mana yang anda bebaskan ke setiap tim?
> - Bagaimana caranya anda menangani operasi non-linear atau asinkron?
> - Bagaimana mencegah variasi schema dan cara pembagian span antar tim tanpa membuat proses instrumentasi menjadi birokratis?
> - Bagaimana memastikan trace tetap dapat dibaca oleh manusia, bukan hanya mesin?
> 
> Semua pertanyaan ini merupakan motivasi mengapa standar seperti OTel muncul. Ketika keputusan teknis perlu diikuti oleh berbagai individu, tim, maupun organisasi, masalahnya sudah berubah dari masalah teknis menjadi masalah sosial, atau bahkan politik.