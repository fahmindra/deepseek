---
slug: modul-13-4
title: modul-13-4
---
# 1.4 End-to-end observability LLM menggunakan Langfuse

---

Setelah mempelajari detail mengenai implementasi observability, perlu dipahami bahwa berbagai proses tersebut umumnya sudah diimplementasikan di belakang layar pada tooling untuk mengimplementasi observability yang sudah siap pakai. Salah satu tooling yang cukup populer adalah Langfuse. Tidak hanya observability, Langfuse juga berisi berbagai macam fitur end-to-end yang bisa digunakan untuk memastikan keberjalanan sistem LLM di production.

Dalam submodul ini, akan dipelajari bagaimana caranya Langfuse dapat digunakan tidak hanya untuk instrumentasi saja, tapi juga melakukan tracing melalui dashboard dan bermacam fitur lainnya yang tersedia.

---

### 1.4.1 Apa itu Langfuse?

Langfuse merupakan sebuah platform untuk LLMOps yang menyediakan fungsi observability, manajemen prompt, dan evaluasi yang sudah terintegrasi dengan sebuah dashboard yang memudahkan visualisasi data, monitoring, dan tracing. Langfuse tersedia secara open source dan bisa digunakan melalui self-hosting maupun melalui managed platform (berbayar dengan free tier) yang mereka sediakan.

Kelebihan Langfuse adalah fleksibilitas integrasinya dengan berbagai model dan framework aplikasi LLM. Langfuse membagi komponen observability menjadi 3 bagian saja, yaitu session, trace, dan observation.

![Komponen Observability Langfuse](https://lms.sdmdigital.id/pluginfile.php/635525/mod_page/content/7/image4.jpg)
**Gambar 8. Komponen observability pada Langfuse**

Observation bisa dipandang sebagai span dan memiliki sifat yang sama, bisa nested untuk membentuk hierarki dari suatu proses. Session digunakan untuk mengelompokkan trace untuk jenis interaksi yang terjadi berulang, misalnya percakapan dengan chatbot atau sebuah workflow.

Dalam langfuse, ada beberapa cara untuk melakukan instrumentasi:

1.  **Instrumentasi otomatis.** Langfuse menyediakan integrasi otomatis dengan berbagai library dan framework, dimana proses instrumentasi bisa semudah mengatur environment variables lalu mengubah beberapa baris kode saja.
2.  **Instrumentasi manual.** Untuk instrumentasi yang custom atau memerlukan kontrol penuh, Langfuse menyediakan beberapa cara instrumentasi manual:
    *   Menggunakan decorator pada fungsi-fungsi yang ingin diinstrumentasi. Pendekatan ini secara otomatis menangani propagasi konteks serta menangani nested observation.
    *   Menggunakan context manager. Pendekatan ini mirip seperti cara membuat span menggunakan library OTel. Pada pendekatan ini, propagasi konteks dilakukan secara manual.
    *   Secara manual menandai kapan suatu span bermula dan kapan span berakhir.

Secara pragmatis, sebaiknya anda menggunakan instrumentasi otomatis saja agar tidak terlalu pusing dengan detail implementasi instrumentasi dan bisa lebih berfokus ke menyiapkan sistem agar proses diagnosanya lebih mudah ketika ada masalah. Kalau ingin mengetahui lebih lanjut mengenai metode-metode instrumentasi manual pada Langfuse, bisa langsung mengunjungi [dokumentasinya](https://langfuse.com/docs/observability/sdk/python/instrumentation).

---

### 1.4.2 Menjalankan Langfuse dengan self-hosting

Untuk menjalankan Langfuse, anda perlu menginstall:

*   Git
*   Docker

Langkah pertama yang perlu dilakukan adalah cloning repository Langfuse terlebih dahulu:

```bash
git clone https://github.com/langfuse/langfuse.git
cd langfuse
```

Perlu dicatat bahwa melakukan cloning langsung akan menggunakan versi terbaru Langfuse yang mungkin sudah outdated ketika anda membaca modul ini. Rilis terbaru Langfuse saat modul ini ditulis adalah versi v3.135.1. Untuk menggunakan versi tersebut, anda bisa menjalankan perintah berikut di dalam directory hasil cloning sebelum lanjut ke langkah berikutnya:

```bash
git checkout tags/v3.135.1
```

Opsional: Setelah selesai, buka `docker-compose.yml` pada directory tersebut, lalu ubah semua nilai yang ditandai dengan `# CHANGEME`. Ini penting untuk keamanan, tapi untuk percobaan lokal saja mungkin tidak apa apa. Apabila di masa depan nanti anda menggunakan Langfuse di production, jangan lupa ubah nilai-nilai tersebut.

Untuk menjalankan Langfuse, selanjutnya cukup gunakan perintah berikut:

```bash
docker compose up
```

Anda perlu menunggu sebentar karena proses ini perlu men-download image yang diperlukan. Setelah proses download selesai, anda bisa melihat log tercetak di layar terminal. Begitu sudah ada log yang dengan tulisan Ready, artinya Langfuse sudah berjalan.

![Log Langfuse Ready](https://lms.sdmdigital.id/pluginfile.php/635525/mod_page/content/7/image18.png)
**Gambar 8. Log yang menandakan Langfuse sudah berjalan**

Selanjutnya anda bisa membuka browser lalu mengunjungi `http://localhost:3000` untuk mengakses dashboard Langfuse. Buat akun baru, lalu setelah login anda bisa melihat dashboard.

![Dashboard Langfuse](https://lms.sdmdigital.id/pluginfile.php/635525/mod_page/content/7/image12.png)
**Gambar 9. Dashboard Langfuse**

Ikuti menu yang tersedia untuk membuat organisasi dan project baru. Setelah selesai, Anda bisa langsung membuat API Key untuk digunakan secara lokal.

![API Key Langfuse](https://lms.sdmdigital.id/pluginfile.php/635525/mod_page/content/7/image17.png)
**Gambar 10. API Key yang siap digunakan.**

> **WARNING:** pastikan anda menyimpan nilai secret key yang sudah dibuat. Kalau tidak, anda perlu membuat API Keys baru karena secret key hanya bisa dilihat sekali saja.

---

### 1.4.3 Memulai tracing dengan Langfuse

Mari gunakan salah satu kode aplikasi sederhana yang sudah dikembangkan pada hands on sebelumnya sebagai objek instrumentasi.

```python
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain.tools import tool


import langfuse
from langfuse import observe
from langfuse.langchain import CallbackHandler


client = langfuse.get_client()
langfuse_handler = CallbackHandler(update_trace=True)


# berbeda dengan langsung menggunakan API
# langchain memudahkan definisi tool
# langsung pada implementasinya
# menggunakan docstring dan dekorator


@tool
def check_delivery_status(order_id: str):
    """Check the delivery status based on order ID."""
    print(f"Tool check_delivery_status was called with order_id {order_id}")
    return {
        "order_id": order_id,
        "order_status": "In delivery",
        "order_last_location": "Sedang disortir di DC cakung",
        "ETA": "31-12-2025",
    }




@tool
def check_shopping_cart(user_id: str):
    """Retrieve the user's shopping cart based on user ID."""
    print(f"Tool check_shopping_cart was called with user_id {user_id}")
    return {
        "user_id": user_id,
        "items_in_cart": [
            {
                "item_id": 1,
                "item_name": "babibas running shoes",
                "item_type": "shoes",
                "item_details": {"size": 48, "color": "bright pink"},
            },
            {
                "item_id": 22,
                "item_name": "bortusten performance socks",
                "item_type": "socks",
                "item_details": {"size": "XL", "color": "bright yellow"},
            },
            {
                "item_id": 111,
                "item_name": "becs anti-slip shoe lace",
                "item_type": "shoe_accessories",
                "item_details": {"color": "cyan"},
            },
        ],
    }




SYSTEM_PROMPT = """\
Act as a friendly shopping assistant. You will be provided with tools to help you answer the user's queries.
When a tool is not available to help a user's request, suggest to contact a human instead.
Do not make up answers, if a user does not provide the required information, ask them.
Answer should be in fully natural language. It's okay to add small formatting.
Answer in Indonesian by default unless the user asks in english.
"""




# ganti inisiasi kalau menggunakan provider lain
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("API_KEY_LLM")
)
# tidak perlu bingung dulu dengan terminologi "Agent" di sini
# anggap saja LLM biasa dengan tool
agent = create_agent(
    model=llm,
    tools=[check_delivery_status, check_shopping_cart],
    system_prompt=SYSTEM_PROMPT,
)




# By default menggunakan gemini. Ganti nama model kalau menggunakan provider lain
@observe
def chat(query):
    result = agent.invoke(
        {"messages": [{"role": "user", "content": query}]},
        config={"callbacks": [langfuse_handler]},
    )
    return result["messages"][-1].content




print(">> Pemanggilan 1, perlu tool check status pengiriman <<")
print(
    chat(
        "Halo, saya ingin mengecek status pengiriman pesanan saya dengan ID 1112231233"
    )
)
print("====== separator ======")
print(">> Pemanggilan 2, perlu tool check keranjang belanja <<")
print(
    chat("Halo, saya ingin mengecek keranjang belanja saya dengan user ID Ivan#19827")
)
print("====== separator ======")
print(">> Pemanggilan 3, tidak perlu tool apapun, cuma pertanyaan biasa <<")
print(chat("Halo, apa kabar? Siapa kamu?"))
```

Yang menarik dari Langfuse adalah kemudahan integrasinya dengan banyak provider maupun framework LLM. Contohnya, untuk menggunakan tracing Langfuse pada aplikasi yang menggunakan langchain, pertama kita perlu setup beberapa environment variable. Buat sebuah file `.env` pada directory aplikasi anda (jika belum ada), lalu isi dengan nilai berikut:

```text
LANGFUSE_SECRET_KEY = "sk-lf-..."
LANGFUSE_PUBLIC_KEY = "pk-lf-..."
LANGFUSE_BASE_URL = "http://localhost:3000"
```

Secret key dan public key berasal dari API key yang sebelumnya sudah anda buat. Kalau belum ada, bisa langsung membuka settings > API Keys > Create new API Keys pada dashboard. Base URL diisi dengan URL localhost karena menggunakan self-hosting. Silahkan sesuaikan jika menggunakan nomor port yang berbeda.

Selain itu, jika anda menggunakan layanan API yang managed dari Langfuse langsung, anda perlu memilih satu dari dua URL berikut:

```text
LANGFUSE_BASE_URL = "https://cloud.langfuse.com" # pakai yang server EU
LANGFUSE_BASE_URL = "https://us.cloud.langfuse.com" # pakai yang server US
```

Setelah selesai, yang perlu dilakukan untuk instrumentasi aplikasi LangChain menggunakan Langfuse adalah mengimport library langfuse lalu menginisiasi sebuah callback handler.

```python
from langfuse.langchain import CallbackHandler
 
langfuse_handler = CallbackHandler()
```

Tracing secara otomatis dilakukan dengan cara memasukkan handler tersebut ke dalam konfigurasi ketika memanggil LLM menggunakan Langchain:

```python
def chat(query):
    result = agent.invoke(
        {"messages": [{"role": "user", "content": query}]},
        config={"callbacks": [langfuse_handler]},
    )
    return result["messages"][-1].content
```

Setelah itu, aplikasi dapat dijalankan seperti biasa. Setelah selesai, anda bisa melihat trace yang sudah dicatat pada dashboard Langfuse anda.

![Dashboard Trace Langfuse](https://lms.sdmdigital.id/pluginfile.php/635525/mod_page/content/7/image2.png)
**Gambar 11. Trace langsung bisa diakses secara otomatis dari dashboard Langfuse.**

Trace langsung bisa diakses secara otomatis dari dashboard Langfuse. Detail lebih lanjut mengenai setiap trace termasuk strukturnya bisa dilihat pada masing-masing trace.

![Detail Trace Langfuse](https://lms.sdmdigital.id/pluginfile.php/635525/mod_page/content/7/image1.png)
**Gambar 12. Detail sebuah trace**

Menggunakan platform seperti Langfuse menghemat effort untuk menambahkan observability ke dalam sebuah sistem LLM karena baik instrumentasi, penyimpanan, hingga visualisasi sudah diimplementasi dan siap untuk langsung digunakan.

---

### Mini Quiz
*(Konten Interaktif H5P)*

> **Refleksi Singkat**
>
> Anda sudah melihat bagaimana platform seperti Langfuse dapat menyederhanakan implementasi observability dengan menyediakan instrumentasi, penyimpanan, dan visualisasi yang sudah siap pakai.
>
> Namun bayangkan Anda bekerja di organisasi yang sudah memiliki infrastruktur observability yang mapan dengan tooling lain (misalnya kombinasi Prometheus, Grafana, dan Jaeger), dan ada kekhawatiran tentang vendor lock-in jika mengadopsi platform terintegrasi seperti Langfuse.
>
> Jika Anda harus membuat keputusan antara menggunakan platform terintegrasi seperti Langfuse dan membangun solusi observability sendiri dengan komponen-komponen terpisah, faktor-faktor apa saja yang akan Anda pertimbangkan?
>
> Dalam merancang jawabannya, pikirkan hal-hal seperti:
> - Bagaimana trade-off antara kemudahan implementasi versus fleksibilitas dan kontrol penuh?
> - Dalam skenario apa self-hosting lebih masuk akal dibanding menggunakan managed platform, dan sebaliknya?
> - Bagaimana mempertimbangkan biaya tidak hanya dari sisi finansial, tetapi juga dari sisi engineering effort dan maintenance overhead?
> - Bagaimana memastikan bahwa pilihan tooling tidak menghambat migrasi di masa depan jika kebutuhan berubah atau muncul teknologi yang lebih baik?
>
> Sama seperti refleksi sebelumnya. Ketika keputusan teknis bisa mempengaruhi banyak pihak, masalahnya tidak lagi teknis, tapi menjadi sosial dan politik. Sebagai engineer, bagaimana anda bisa meyakinkan non-engineer pada organisasi anda?