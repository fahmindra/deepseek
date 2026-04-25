---
slug: modul-13-2
title: modul-13-2
---
# 1.2 Structured Logging dan Analisis log

---

Sebelum cara kerja suatu sistem bisa dianalisa melalui trace, setiap event yang terjadi dalam sistem tersebut perlu bisa dibaca terlebih dahulu. Logging merupakan cara untuk mengekspos event-event ini agar bisa dibaca dari luar sistem.

Dalam implementasi observability, perlu konvensi dan format tertentu dalam melakukan logging karena log tidak hanya akan ditulis begitu saja, tapi perlu dibaca dan diproses oleh sebuah sistem lain, lalu dikelompokkan ke dalam sebuah span, kemudian disatukan menjadi sebuah trace yang siap untuk analisis. Diperlukan struktur dan skema yang memudahkan pemrosesan sebuah log, yang bisa dicapai melalui structured logging.

---

### 1.2.1 Apa urgensi structured logging?

Sebuah log tidak hanya berisi artifact atau informasi inti saja, tapi juga perlu dilengkapi dengan informasi ekstra yang penting untuk pemrosesan lanjut seperti mengurutkan atau mengelompokkan log, seperti:

*   Timestamp untuk mengurutkan log secara kronologis
*   ID / nama dari bagian sistem tempat log tersebut berasal
*   ID / nama dari trace atau span yang memayungi log tersebut untuk pengelompokkan
*   Informasi ekstra lainnya, misalnya log yang berisi sebuah prompt perlu memberikan informasi seperti versi dari template yang digunakan

Informasi dalam sebuah log perlu diekstraksi dengan cara melakukan parsing agar setiap informasi tersebut bisa digunakan secara terpisah. Untuk format log yang lebih sederhana, proses parsingnya mungkin cukup simpel. Contohnya pada log dengan format CLF (Common Log Format) berikut:

```text
[INFO] 2024-08-04 12:45:23 - LLM CALLED WITH PROMPT: “# ROLE\nYou are a helpful assistant…..”
```

Setiap aplikasi bisa jadi memiliki format masing-masing dalam melakukan logging. Tapi bayangkan jika bagian lain dari sistem memiliki format log yang berbeda:

```text
[INFO] 2024-08-04 12:45:23 - LLM CALLED WITH PROMPT: “# ROLE\nYou are a helpful assistant…..”
15:27:29.0982 08-04-2024 Log Level: Debug Content: ….
```

Bayangkan juga jika struktur dari beberapa log begitu kompleks, sehingga perlu logika yang rumit untuk parsing:

```text
[2025-12-01 17:30:15.897] || Service: RAG_Worker || State: END || TaskID: 98df7-a3c4 ||
>> [CHUNK_PROCESSOR] Finished 50 documents in 4.5s. Documents had an average of 12 chunks.
    * Embedding: 70% success rate
    * API_Host: https://api.embedder.io/v3/vectors
    * UserAgent: Internal/Python-Worker
    ! WARN: 3 Docs failed embedding due to context limit (MaxChunkSize=2000). Total Tokens Sent: 32450.
```

Structured logging menyelesaikan masalah ini melalui dua prinsip utama:

1.  Ada struktur yang baku yang memudahkan pengaksesan informasi, terutama struktur data seperti JSON, dan
2.  Ada konvensi penamaan, skema, serta cara merepresentasikan data yang jelas, sehingga logika parsing konsisten untuk log manapun dari sebuah sistem.

---

### 1.2.2 Seperti apa bentuk structured logging?

Prinsip utama dari sebuah structured log umumnya dicapai dengan menggunakan struktur data yang memiliki pengaksesan nilai yang mudah, seperti JSON. Sisa dari submodul ini akan mengasumsikan penggunaan JSON untuk melakukan structured logging.

JSON memiliki beberapa sifat yang ideal:

*   Syntaxnya jelas dan mudah diproses secara programatis
*   Struktur key-value yang memudahkan pencarian dan memberikan konteks semantik yang jelas
*   Mendukung struktur / tipe yang kompleks, seperti nested objects atau list untuk mengelompokkan informasi berdasarkan hierarki atau kesamaan semantik tertentu

Menggunakan struktur data seperti JSON saja tidak cukup untuk implementasi structured logging. Selain struktur yang mendukung representasi informasi kompleks yang mudah untuk diproses, diperlukan sebuah skema data yang stabil. Skema ini meliputi:

*   Penamaan key yang konsisten
*   Value yang bisa diprediksi / diantisipasi (tipe datanya, nilai-nilai validnya pada enumerasi, struktur data untuk tipe yang kompleks, dsb)

OTel mendefinisikan log data model (skema baku untuk sebuah log) yang menentukan skema dari sebuah log terstruktur:

**Tabel 2. Field dari log data model yang ditetapkan dalam standar OTel**

| Nama field / key pada JSON | Deskripsi | Tipe data |
| :--- | :--- | :--- |
| Timestamp | Kapan suatu event terjadi | Uint64 |
| ObservedTimestamp | Kapan log diproses oleh sistem | Uint64 |
| TraceId | ID dari Trace dalam sebuah request | byte array (16 bytes) |
| SpanId | ID dari Span dalam sebuah | byte array (16 bytes) |
| TraceFlags | W3C trace flag. | byte |
| SeverityText | Log Level | String |
| SeverityNumber | Nilai numerik dari log level | uint32 |
| Body | Data utama sebuah log, misalnya output dari sebuah proses | Apapun |
| Resource | Identifikasi sumber dari sebuah log | Dictionary |
| InstrumentationScope | Digunakan untuk filtering | Dictionary |
| Attributes | Digunakan untuk menyimpan atribut tambahan seperti metadata | Dictionary |

Berikut merupakan contoh sebuah structured log yang menggunakan format JSON yang menunjukkan sebuah event pemanggilan tool:

```json
{
  "Timestamp": 1733055668123456789,
  "ObservedTimestamp": 1733055668123800000,
  "SeverityText": "INFO",
  "SeverityNumber": 9,
  "TraceID": "5e1a3d9b0f4c7e6a2b8d1f0c3a5e9b7d",
  "SpanID": "c2b8d1f0c3a5e9b7",
  "Body": {
    "tool_output_summary": "Successfully fetched weather forecast for location 'Jakarta'. Temperature: 29C, Condition: Sunny.",
    "raw_response_object": {
      "city": "Jakarta",
      "temp_c": 29,
      "humidity": 65,
      "forecast_time": "2025-12-01T21:00:00Z"
    }
  },
  "Attributes": {
    "service.name": "llm-agent-service-prod",
    "deployment.environment": "production",
    "log.event_type": "tool_execution_complete",
    "tool.name": "get_current_weather",
    "tool.execution_status": "SUCCESS",
    "llm.model_name": "gpt-4o",
    "tool.execution_duration_ms": 235,
    "user.id": "USR-4567"
  }
}
```

Bisa diperhatikan kalau konten utama dari sebuah event yaitu hasil pemanggilan tool diletakkan pada body. Sedangkan detail pemanggilannya, termasuk LLM yang digunakan, durasi pemanggilan, nama tool semuanya disimpan sebagai atribut. Perlu diingat sebagian besar parameter ini opsional, informasi mengenai detail setiap field bisa diakses langsung pada [https://opentelemetry.io/docs/specs/otel/logs/data-model/](https://opentelemetry.io/docs/specs/otel/logs/data-model/).

Dalam menyimpan data untuk logging, ada dua tempat utama yang digunakan yaitu pada body (konten utama) dan pada attribute. Walaupun data pada attribute dibebaskan konten dan formatnya, umumnya setiap jenis event distandarkan atributnya. Misalnya:

*   Setiap event menyimpan ID dari request/sesi, nama dari sebuah service yang dibuat, tipe event, serta informasi lainnya yang memudahkan segmentasi dan filtering log berdasarkan atribut.
*   Setiap event menyimpan parameter yang mempengaruhi event tersebut. Contohnya retrieval menyimpan data dokumen hasil retrieval serta atribut berupa parameter retrieval yaitu nilai top-K, threshold, model embedding yang digunakan, versi kode, dsb. Sebagai rule of thumb, segala sesuatu yang perubahannya bisa mempengaruhi hasil suatu event, maka perlu disimpan sebagai parameter.

---

### 1.2.3 Apa saja kegunaan structured log dalam observability?

**Korelasi log**

Korelasi log dilakukan untuk menghubungkan sebuah log dengan data observasi lainnya. Dalam sistem LLM, proses ini umumnya menggunakan TraceID dan SpanID untuk mengelompokkan event yang terjadi ke dalam lingkupnya masing-masing.

![Korelasi Log](https://lms.sdmdigital.id/pluginfile.php/635523/mod_page/content/9/image5.jpg)
**Gambar 7. Ilustrasi proses korelasi log yang tersebar menjadi trace yang terstruktur.**

**Metadata Filtering**

Karena setiap log memiliki atribut yang terstandar untuk setiap jenis event, metadata dapat digunakan untuk filtering dengan mudah, misalnya pada suatu waktu kita ingin melihat semua log yang memberikan event hasil pemanggilan suatu tool untuk melihat variasi hasilnya, atau pada suatu waktu kita ingin membandingkan hasil antara versi tool atau versi prompt tertentu, dsb.

Metadata ini bisa kita gunakan untuk melakukan filtering, misalnya kita membatasi hanya log yang memanggil tool tertentu saja melalui atribut `tool.name`. Atau bisa juga menyaring agar hanya log yang berhubungan dengan user tertentu melalui atribut `user.id` dan lain sebagainya.

**Pemrosesan otomatis**

Karena struktur dan skemanya tetap dan sudah diketahui, mudah dihubungkan, dan mudah difilter, maka pemrosesan otomatis mungkin untuk dilakukan. Misalnya secara berkala query dari user dikumpulkan untuk melakukan analisis terhadap distribusi / pola-pola permintaan dari user, atau ketika melakukan A/B testing, bisa dibandingkan secara otomatis tingkat "penerimaan" jawaban akhir berdasarkan versi prompt tertentu, dan lain sebagainya.

---

### 1.2.4 Best practices logging

1.  Bedakan body dengan attribute. Simpan data yang paling relevan saja dengan suatu event di body. Attribute hanya digunakan untuk menyimpan konteks tambahan yang memberikan konteks mengenai data dan event tersebut untuk memudahkan filtering dan query dalam pemrosesan selanjutnya maupun untuk lebih memahami konfigurasi sistem yang menghasilkan suatu nilai.
2.  Walaupun body dan atribut dibebaskan formatnya, pakai konvensi penamaan yang jelas dengan namespace dan separator untuk memberikan konteks semantik serta apa tujuan dari atributnya. Contoh `retrieval.top_k` memberikan petunjuk bahwa atribut tersebut merupakan nilai top-K untuk proses retrieval. Kalau bisa, ikuti saja penamaan yang distandarkan, misalnya jika ada standar penamaan yang ditetapkan OTel atau disepakati oleh Tim, bisa diikuti.
3.  Optimalkan data atribut untuk melakukan filtering dan query. Atribut sebaiknya menyimpan data yang kardinalitasnya (jumlah nilai uniknya) rendah.
4.  Pisahkan antara penyimpanan dengan korelasi. Untuk data yang berukuran besar, simpan data tersebut secara eksternal, misalnya di bucket S3 atau di database, lalu simpan saja pointernya, entah URL s3 atau ID di database. Kalau memang datanya sudah disimpan sedari awal, langsung saja pakai pointernya.

---

### 1.2.5 Bagaimana caranya implementasi structured logging?

Realitanya, structured logging sudah ditangani oleh sebagian besar framework atau library yang digunakan untuk observability, anda hanya perlu memanggil fungsi yang disediakan dan memberikan informasi sesuai format yang diminta. Akan tetapi tidak ada salahnya mempelajari cara kerjanya secara lebih mendalam.

Dalam Python, kita mengenal library `logging`. Untuk mengatur agar setiap baris log memiliki struktur berbentuk JSON, kita bisa mengatur formatter dari logger. Contoh sederhananya seperti berikut:

```python
import json
import logging
from datetime import datetime, UTC


# formatter untuk setiap log
class JSONFormatter(logging.Formatter):
    def format(self, record):
        # 4 data di bawah ini by default selalu ada
        payload = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
        }
        # setiap baris log adalah json individual
        return json.dumps(payload)

# inisiasi logger
logger = logging.getLogger("demo")
logger.setLevel(logging.DEBUG)


# mengatur agar handler yang
# mengatur bagaimana log mengirimkan output
# agar menggunakan formatter di atas
handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())


logger.addHandler(handler)


logger.info("ini log info")
logger.debug("ini log debug")
logger.warning("ini log warning")
logger.error("ini log error")
```

Dalam OTel, logging secara individual tidak terlalu difokuskan, yang penting adalah setiap span dan event diorganisir dengan benar. Alih-alih log, OTel menggunakan event.

Berikut adalah contoh yang lebih realistis, langsung berfokus pada struktur dan informasi dalam sebuah span yang dipraktekan pada sebuah aplikasi dummy RAG. Sebelum menjalankan kode di bawah, anda perlu instalasi beberapa library terlebih dahulu:

```bash
# uv
uv add opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp opentelemetry-instrumentation-logging
# pip
pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp opentelemetry-instrumentation-logging
```

Contoh Kode Implementasi:

```python
import logging
import time
from typing import List, Dict, Any


from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter
from opentelemetry.instrumentation.logging import LoggingInstrumentor




# setup otel, terutama service.name
resource = Resource.create({"service.name": "dummy-rag-service"})
provider = TracerProvider(resource=resource)
trace.set_tracer_provider(provider)
tracer = trace.get_tracer(__name__)


# idealnya ini dihubungkan dengan collector untuk dikirim ke backend OTLP
# tapi untuk demo, di-print ke terminal langsung saja
console_exporter = ConsoleSpanExporter()
provider.add_span_processor(SimpleSpanProcessor(console_exporter))


# ini metode untuk augmentasi logging python.
# kita tidak perlu repot mengatur sendiri formatnya
LoggingInstrumentor().instrument(set_logging_format=True)
logger = logging.getLogger("dummy_rag")
logger.setLevel(logging.DEBUG)


# retrieval dummy
class DummyVectorStore:


    def __init__(self):
        self.docs = [
            {"id": "doc1", "text": "Jakarta mengalami penurunan tanah karena overextraction of groundwater."},
            {"id": "doc2", "text": "Solusi teknis termasuk managed aquifer recharge dan piping dari sumber permukaan."},
            {"id": "doc3", "text": "Kebijakan publik diperlukan untuk mengatur pemompaan air tanah."},
        ]


    def search(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        tokens = set(query.lower().split())
        scored = []
        for d in self.docs:
            score = sum(1 for t in tokens if t in d["text"].lower())
            scored.append((score, d))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for score, doc in scored[:k]]


# LLM dummy
class DummyLLM:


    def __init__(self, model_name: str = "dummy-llm-0.1"):
        self.model_name = model_name


    def generate(self, prompt: str, max_tokens: int = 128) -> Dict[str, Any]:
        # pretend we "tokenized" the prompt
        token_count = len(prompt.split()) + 10
        # produce a simple response using the prompt
        response = f"[DUMMY GENERATION based on prompt snippet] {prompt[:200]}"
        return {
            "text": response,
            "tokens": token_count,
            "model": self.model_name,
        }




# pipeline RAG dummy
class DummyRAG:
    def __init__(self, store: DummyVectorStore, llm: DummyLLM):
        self.store = store
        self.llm = llm
        self.tracer = tracer


    def ingest(self, docs: List[Dict[str, Any]]):
        # satu span = 1 proses.
        with self.tracer.start_as_current_span("ingest_documents") as span:
            span.set_attribute("component", "ingest")
            span.set_attribute("document.count", len(docs))
            logger.info("Ingesting %d documents", len(docs))
            time.sleep(0.01)
            span.set_attribute("ingest.status", "ok")


    def retrieve(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        with self.tracer.start_as_current_span("retrieve_documents") as span:
            span.set_attribute("component", "retrieval")
            span.set_attribute("retrieval.query", query)
            span.set_attribute("retrieval.top_k", k)
            start = time.time()
            results = self.store.search(query, k=k)
            latency_ms = (time.time() - start) * 1000
            span.set_attribute("retrieval.latency_ms", int(latency_ms))
            span.set_attribute("retrieval.results_count", len(results))
            logger.debug("Retrieved %d docs for query '%s'", len(results), query)
            return results


    def generate(self, prompt: str, retrieved: List[Dict[str, Any]]) -> Dict[str, Any]:
        with self.tracer.start_as_current_span("model.generate") as span:
            span.set_attribute("component", "generation")
            span.set_attribute("gen.ai.model.name", self.llm.model_name)
            span.set_attribute("gen.ai.provider", "local-dummy")
            span.set_attribute("gen.ai.input.prompt", prompt)
            span.set_attribute("gen.ai.context.doc_ids", ",".join(d["id"] for d in retrieved))


            start = time.time()
            out = self.llm.generate(prompt)
            latency_ms = (time.time() - start) * 1000
            span.set_attribute("gen.ai.output.text", out["text"])
            span.set_attribute("gen.ai.output.tokens", int(out["tokens"]))
            span.set_attribute("gen.ai.latency_ms", int(latency_ms))
            logger.info("Model generated %d tokens using model %s", out["tokens"], out["model"])
            return out


    def answer(self, user_query: str) -> Dict[str, Any]:
        with self.tracer.start_as_current_span("rag.request") as span:
            span.set_attribute("component", "rag")
            span.set_attribute("rag.user_query", user_query)


            retrieved = self.retrieve(user_query, k=3)


            prompt_parts = [f"User: {user_query}", "Context:"]
            for d in retrieved:
                prompt_parts.append(f"- {d['id']}: {d['text']}")
            prompt = "\n".join(prompt_parts)


            gen_out = self.generate(prompt, retrieved)


            answer = {
                "query": user_query,
                "answer": gen_out["text"],
                "sources": [d["id"] for d in retrieved],
                "llm": gen_out["model"],
            }
            span.set_attribute("rag.answer.source_count", len(answer["sources"]))
            span.set_attribute("rag.status", "ok")
            logger.debug("RAG answered query with %d sources", len(answer["sources"]))
            return answer




if __name__ == "__main__":
    store = DummyVectorStore()
    llm = DummyLLM()
    rag = DummyRAG(store, llm)


    new_docs = [
        {"id": "doc4", "text": "Contoh dokumen tambahan tentang konservasi air."},
    ]
    rag.ingest(new_docs)


    queries = [
        "penurunan tanah jakarta solusi",
        "bagaimana mengurangi penggunaan air tanah?",
    ]


    for q in queries:
        logger.info("--- memanggil rag dengan query: %s ---", q)
        result = rag.answer(q)
        logger.info("Jawaban: %s", result["answer"])
        logger.info("Sumber: %s", result["sources"])
```

Berikut adalah salah satu span yang menjadi output kode di atas. Perhatikan kalau setiap event (log) individual dikelompokkan dalam satu span.

```json
{
    "name": "retrieve_documents",
    "context": {
        "trace_id": "0x19e2c25125db5984c04a38b1cbb32fed",
        "span_id": "0x19c00239e9db220d",
        "trace_state": "[]"
    },
    "kind": "SpanKind.INTERNAL",
    "parent_id": "0xe968cbf47c022775",
    "start_time": "2025-12-08T11:29:36.981633Z",
    "end_time": "2025-12-08T11:29:36.981633Z",
    "status": {
        "status_code": "UNSET"
    },
    "attributes": {
        "component": "retrieval",
        "retrieval.query": "bagaimana mengurangi penggunaan air tanah?",
        "retrieval.top_k": 3,
        "retrieval.latency_ms": 0,
        "retrieval.results_count": 3
    },
    "events": [
        {
            "name": "retrieve_event",
            "timestamp": "2025-12-08T11:29:36.981633Z",
            "attributes": {
                "message": "Retrieved 3 docs for query bagaimana mengurangi penggunaan air tanah?"
            }
        }
    ],
    "links": [],
    "resource": {
        "attributes": {
            "telemetry.sdk.language": "python",
            "telemetry.sdk.name": "opentelemetry",
            "telemetry.sdk.version": "1.39.0",
            "service.name": "dummy-rag-service"
        },
        "schema_url": ""
    }
}
```

---

### **Mini Quiz**
*(Konten Interaktif H5P)*

> **Refleksi Singkat**
>
> Struktur dan schema adalah inti dari structured logging. Tapi schema yang terlalu ketat bisa menghambat evolusi sistem, sedangkan schema yang terlalu fleksibel membuat analitik menjadi kacau. Bayangkan Anda adalah arsitek observability dalam sebuah startup di Silicon Valley. Anda diminta untuk merancang schema logging untuk sebuah sistem LLM yang terus berkembang. Dalam hitungan bulan, akan lahir puluhan jenis event baru, parameter baru, dan modul baru. Bagaimana anda menentukan bagian mana dari schema yang sudah ditentukan, dan bagian mana dari schema yang dibebaskan ke developer? Siapa saja yang perlu anda ajak berdiskusi untuk menentukan schema yang tepat ini?