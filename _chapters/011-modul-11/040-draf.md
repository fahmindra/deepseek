---
slug: modul-11-4
title: modul-11-4
---
Berikut adalah isi konten yang telah dirapikan ke dalam format Markdown sesuai dengan hirarki aslinya:

# 7.4 Implementasi Agent Menggunakan LangGraph dan n8n

---

Ada dua metode yang populer untuk membangun sebuah *Agent* berbasis *workflow*, yaitu secara programatis, misalnya menggunakan *framework* seperti LangGraph, atau secara visual menggunakan *platform* *open-source* seperti n8n. Dalam modul ini, kita juga akan melakukan eksplorasi singkat bagaimana mengimplementasi *workflow* untuk melakukan *task* sederhana.

---

### 7.4.1 Pendekatan yang Lebih Teknikal dan Low-level: Langgraph

Langgraph adalah *library* untuk menyusun *workflow* menggunakan graf. Dalam Langgraph, ada dua komponen utama yang perlu anda implementasi:

1.  **State**  
    Dalam Langgraph, sebuah *state* direpresentasikan sebagai *dictionary* Python. Sebuah *state* berisi semua nilai yang diperlukan selama suatu *workflow* berjalan untuk menyimpan kondisi terkini, yang dapat digunakan untuk melakukan input pemrosesan maupun untuk logika percabangan. Prinsip apa yang perlu masuk *state* cukup sederhana. Nilai apapun yang diubah ataupun diatur oleh satu tahap *workflow* dan diperlukan pada tahap lainnya, sebaiknya masuk *state*, termasuk *input* dan *output* akhir dari *workflow* tersebut.

    Misalnya untuk aplikasi yang melakukan pencarian di internet untuk melakukan riset terhadap suatu informasi, statenya berbentuk seperti ini:

```python
class State(TypedDict):
    topic: str # topik informasi yang perlu dicari, input workflow
    query: str = "" # untuk melakukan pencarian
    collected_information: list[str] = [] # untuk menyimpan hasil pencarian
    final_report: str = "" # menyimpan output akhir
```

2.  **Node**  
    Sebuah *node* menerima *state* terkini dari proses berjalannya *workflow*, melakukan suatu pemrosesan, melakukan *update* pada *state*, lalu memberikan output berupa *state* yang sudah di-*update*. Logika setiap tahap suatu *workflow* diimplementasi di dalam masing-masing *node*.

    Dalam contoh berikut, *node* yang bertanggung jawab mencari informasi di internet membutuhkan *query* yang disimpan dalam *state*, lalu menyimpan hasil pencariannya dalam *state*.

```python
def search_internet(state: State) -> State:
    query = state["query"]
    search_result = call_search_api(query)
    state.collected_information.extend(search_result)
    return state
```

Seperti yang sudah dipelajari sebelumnya, dalam sebuah *graf*, setiap *node* terhubung ke *node* lain melalui sebuah *edge*. Dalam Langgraph, sebuah *workflow* disusun dengan cara menambahkan setiap *node* ke dalam *workflow* lalu mendefinisikan transisi terarah antar *node*. *Snippet* berikut memberikan gambaran bagaimana sebuah *workflow* bisa disusun dalam *langgraph*.

```python
from langgraph.graph import StateGraph, END

...

# inisiasi workflow sebelum menambahkan semua node
workflow = StateGraph(State)
# node untuk menentukan query pencarian selanjutnya
workflow.add_node("determine_next_query", determine_next_query)
workflow.add_node("search_internet", search_internet)
# node untuk agregasi informasi hasil searching
workflow.add_node("aggregate_information", aggregate_information)
# node untuk melakukan laporan akhir
workflow.add_node("generate_final_report", generate_final_report)

# di bawah ini mendefinisikan edge / transisi antar node

# menentukan workflow mulai dari mana
workflow.set_entry_point("determine_next_query")
# logika transisi sederhana dari node kiri ke node kanan
workflow.add_edge("determine_next_query", "search_internet")
workflow.add_edge("search_internet", "aggregate_information")

# kasus khusus untuk percabangan
workflow.add_conditional_edges(
        # node asal 
        "aggregate_information",
        # fungsi khusus untuk menentukan arah selanjutnya
        should_search_again,
        {
            # setiap pasangan key-value
            # memetakan node tujuan (value)
            # berdasarkan output fungsi khusus (key) di atas
            "search_more": "determine_next_query",
            "continue": "generate_final_report"
        }
    )

# node spesial untuk menandai workflow telah selesai berjalan
workflow.add_edge("generate_final_report", END)
# setelah selesai didefinisikan, dicompile
# agar bisa digunakan
workflow = workflow.compile()
```

Dalam contoh di atas, fungsi `should_search_again` adalah fungsi spesial yang menerima sebuah *state* yang memberikan suatu output, misalnya `boolean` atau `string` yang digunakan untuk menentukan transisi selanjutnya saat ada percabangan.

```python
def should_search_again(state: State) -> str:
    topic = state["topic"]
    collected_information = state["collected_information"]
   
    # suatu pemrosesan untuk menentukan nilai tertentu
    # sesuai definisi percabangan, outputnya berarti hanya
    # search_more atau continue
    decision = ...
    return decision
```

Untuk menggunakan *workflow* yang sudah di-*compile*, anda bisa memanggil metode `.invoke` atau `.ainvoke` dengan input sebuah *state* awal. Dalam kasus ini, kita memberikan *state* yang hanya berisi `topic` saja, sisanya pakai nilai kosong. Menjalankan sebuah *workflow* akan mengembalikan *state* yang umumnya berisi *output*.

```python
# buat state awal yang hanya berisi input
initial_state = State(
    topic="bagaimana cara membuat aplikasi LLM"
)
# jalankan workflow
final_state = workflow.invoke(initial_state)
# ambil output dari state akhir workflow
final_report = final_state["final_report"]
...
```

Menggunakan Langgraph memberikan lebih banyak kontrol, lengkap dengan bermacam fitur misalnya untuk interupsi eksekusi *workflow* ketika butuh input eksternal, misalnya untuk sistem *human-in-the-loop.*

---

### 7.4.2 Pendekatan Visual: n8n

n8n adalah *platform workflow automation* berbasis *open-source* yang memungkinkan Anda membangun alur kerja secara visual. Alih-alih menulis kode, anda dapat menyusun setiap *node* untuk menghubungkan API, menjalankan logika, atau memproses data. Anda dapat melihat dokumentasi dan petunjuk n8n pada link berikut: [n8n documentation](https://docs.n8n.io/).

*Node* merupakan komponen utama dari *workflow* n8n. Setiap *node* merepresentasikan sebuah aksi, misalnya melakukan HTTP request, memanggil LLM, memproses teks, atau menyimpan data ke layanan tertentu.

**Gambar 17. Bentuk *interface* n8n. Anda bisa menyusun *workflow* dengan cara menghubungkan setiap *node* secara visual.**

Salah satu kelebihan utama n8n adalah integrasinya yang sangat luas dengan berbagai aplikasi, baik untuk *trigger* (istilah n8n, *node* yang memicu berjalannya sebuah *workflow*) setiap blok-blok pemrosesannya, maupun penyimpanannya. Anda bisa menggunakan pesan *whatsapp* sebagai *trigger* untuk diproses oleh LLM, lalu hasilnya disimpan dalam *google sheet* tanpa perlu pusing menulis secara manual kode untuk integrasinya.

Dalam konteks membangun *agent*, n8n dapat digunakan untuk:
*   Menerima input dari pengguna (misalnya melalui webhook)
*   Melakukan panggilan API seperti pencarian menggunakan Tavily
*   Membersihkan atau mengubah data sebelum diproses LLM
*   Mengatur *loop* atau *branching* berdasarkan kondisi
*   Mengirimkan hasil akhir kembali ke pengguna

Dengan antarmuka visualnya, n8n sangat cocok untuk eksperimen cepat, prototyping, atau alur kerja yang membutuhkan banyak integrasi layanan tanpa menulis kode yang kompleks.

#### Menggunakan n8n untuk merangkai sebuah workflow

Untuk merangkai sebuah *workflow*, anda hanya perlu membuka menambahkan *node* lalu menyusunnya. Ada beberapa jenis *node* yang penting anda kenal:

1.  ***Trigger Node***  
    Node ini berfungsi sebagai “pemicu” untuk memulai *sebuah workflow* melalui bermacam cara, mulai dari menekan tombol secara manual, mengirimkan data melalui *webhook*, penjadwalan berkala, hingga melalui pesan whatsapp / telegram.

    **Gambar 18. Salah satu bentuk *trigger node* yang menjalankan *workflow* ketika menerima HTTP *request***

2.  ***Action Node***  
    *Node* seperti ini berhubungan dengan integrasi n8n dengan API atau *platform* lain, dan punya banyak kegunaan, mulai dari mendapatkan data dari pemanggilan API, menulis *file* ke sebuah dokumen, dan lain sebagainya.

    **Gambar 19. Contoh *node* yang tersedia untuk integrasi dengan *google sheets***

3.  ***Control flow node***  
    *Node* seperti ini dipakai untuk mengatur alur transisi, misalnya percabangan atau pengulangan.

    **Gambar 20. Contoh *node* untuk percabangan**

4.  ***Node* untuk memanipulasi / mengatur *state* atau data apapun secara programatis**  
    *Node* seperti ini bisa digunakan untuk melakukan *filtering*, ekstraksi data berdasarkan struktur JSON, mengubah 1 format ke format lain, dan berbagai fungsi lainnya.

    **Gambar 21. Contoh *node* untuk membuat program yang bisa mengubah *state* atau data dalam bentuk JSON**

5.  ***Node* spesifik AI**  
    *Node* ini sebenarnya sama juga melakukan suatu proses juga, tapi lebih spesifik untuk menggunakan AI, terutama LLM.

    **Gambar 22. Bermacam integrasi maupun fungsi abstrak untuk memanggil LLM maupun *template* untuk melakukan *task* tertentu**

Setiap *node* menerima input yang distruktur dalam bentuk JSON dan mengeluarkan *output* dengan struktur yang sama. Untuk menyusun *workflow*, anda hanya perlu menyambungkan satu *done dengan lainnya*.

Misalnya, kita menggunakan *trigger* sederhana melalui *manual trigger* yang mengirimkan data yang konstan untuk digunakan mencari informasi pada API Tavily, lalu ingin hasilnya diproses LLM. Bentuknya kira-kira begini. Karena API Tavily mengembalikan JSON, maka kita gunakan *node* pemrosesan data dulu untuk membuat data dalam bentuk teks untuk masuk ke LLM.

![Contoh workflow sederhana](https://lms.sdmdigital.id/pluginfile.php/635494/mod_page/content/25/image6.png)
**Gambar 23. Contoh workflow sederhana**

Anda dapat menghubungkan 1 *node* dengan *node* secara manual maupun menambahkan *node* yang langsung terhubung ke suatu *node* dengan menekan tombol `[+]` yang terhubung ke sebuah *node*.

Kemudian anda perlu mengatur parameter setiap *node*. Misalnya dalam *node* *trigger* kita mengatur data apa yang dikirim, dalam *node* `search` kita memasukkan API key tavily dan mengatur data mana yang dikirimkan *node* sebelumnya yang digunakan sebagai *query*, lalu *node* selanjutnya memproses hasil *`search`* sebelum dikirimkan ke LLM, dan seterusnya.

Sebagai contoh, pertama kita buat dulu input kita dari *manual trigger* dengan menekannya dua kali lalu mengaktifkan `Always Output Data` dalam `settings`, dan kemudian mengatur *mock data*, lalu menyimpannya.

![Cara mengatur manual trigger](https://lms.sdmdigital.id/pluginfile.php/635494/mod_page/content/25/image22.png)
**Gambar 24. Cara mengatur *manual trigger* agar memberikan output yang diatur manual.**

Kemudian kita melakukan hal yang sama untuk node berikutnya, melalui menunya anda bisa langsung mengatur *credentials* untuk Tavily, `resource` atau layanan yang digunakan, dan lainnya yang bisa dilihat dan eksplorasi sendiri.

Salah satu parameter yang penting adalah *query* yang menjadi input. Anda bisa melihat di sebelah kiri output dari *node* sebelumnya. Untuk menggunakan salah satu `field` output tersebut, anda bisa langsung *drag-and-drop* `key` dari `field` tersebut.

![Memasukkan nilai field](https://lms.sdmdigital.id/pluginfile.php/635494/mod_page/content/25/image11.png)
**Gambar 25. Anda bisa memasukkan nilai suatu `field` pada output *node* sebelumnya dengan cara *drag-and-drop* `key` dari `field` tersebut ke kotak isian.**

Kalau berhasil, anda bisa melihat hasil seperti berikut:

**Gambar 26. Contoh hasil tahap di atas**

Anda bisa menekan tombol `execute step` untuk menjalankan *node* tersebut dan melihat hasilnya.

![Contoh hasil output](https://lms.sdmdigital.id/pluginfile.php/635494/mod_page/content/25/image44.png)
**Gambar 27. Contoh hasil *output***

Hal ini perlu anda lakukan untuk semua *node* dengan pengaturan yang berbeda. Setelah selesai melakukan semuanya, anda dapat kembali ke tampilan awal lalu menekan tombol `Execute workflow` atau menekan `Ctrl+Enter` untuk menjalankan *workflow* anda.

---

### 7.4.3 Perbandingan Langgraph dengan n8n

Kedua pendekatan ini memiliki kelebihan dan kekurangannya masing-masing. Tabel berikut memberikan perbandingan singkat antara keduanya:

**Tabel 2. Perbandingan singkat Langgraph dengan n8n**

| Fitur | n8n (*Low-Code*, visual) | LangGraph (*Code-First*) |
| :--- | :--- | :--- |
| **Fokus utama** | Otomasi workflow | Dibuat untuk orkestrasi *agent* dalam sebuah *workflow* dengan alur transisi yang kompleks. |
| **Jenis *Workflow*** | Workflow yang linear dengan minim percabangan. Perlu *workaround* untuk melakukan *looping* kembali ke *node* sebelumnya. | Berbentuk *state machine* yang secara eksplisit mendukung *loop* dan logika *routing* yang lebih kompleks. |
| ***State management*** | Setiap *node* menerima input dalam format tertentu lalu mengeluarkan output dengan format tertentu. | *State* dengan skema yang konsisten diproses oleh setiap *node* yang membaca suatu nilai dari *state* lalu melakukan *update* ke *state*. |
| **Cara membuat *workflow*** | *Drag and drop* secara visual, sangat sedikit memerlukan *coding*. | Definisi *state*, *node* serta logika pemrosesan dan *update* *state* di dalamnya, lalu rangkai dalam kode. |
| **Ekosistem** | Integrasi dengan berbagai macam *platform* yang langsung tersedia. | Integrasi dengan berbagai komponen Langchain secara *native*, bisa di-extend dengan kode. |

---

### 7.4.4 Membangun Workflow Sederhana dengan Langgraph dan n8n

Mari implementasi *workflow* sederhana menggunakan n8n dan Langgraph.

#### Apa yang akan dibangun?

Anda akan membangun sebuah *agent* yang menerima informasi berupa nama sebuah produk, lalu akan mencari *review* di internet untuk membantu anda lebih memahami mengenai sentimen pengguna lain terhadap produk tersebut.

#### Implementasi *workflow* yang disederhanakan dalam n8n

Versi *workflow* yang disederhanakan ini hanya linear saja. Alurnya adalah sebagai berikut:
1.  Menerima nama produk
2.  Melakukan pencarian *review* di internet.
3.  Melakukan ekstraksi poin-poin utama review.
4.  Menghasilkan rekomendasi akhir.

Gunakan perintah berikut untuk melakukan instalasi dan menjalankan n8n melalui *docker*:

```bash
# hanya perlu dilakukan sekali saja
docker volume create n8n_data

# lakukan setiap kali ingin menjalankan n8n
docker run -it --rm --name n8n -p 5678:5678 -e GENERIC_TIMEZONE="Asia/Jakarta" -e TZ="Asia/Jakarta" -e N8N_ENFORCE_SETTINGS_FILE_PERMISSIONS=true -e N8N_RUNNERS_ENABLED=true -v n8n_data:/home/node/.n8n docker.n8n.io/n8nio/n8n
```

**Gambar 28. Tampilan setelah menjalankan perintah di atas berisi URL untuk membuka *editor* n8n.**

Lakukan konfigurasi sesuai petunjuk yang sudah diberikan sebelumnya untuk mengatur agar `Manual trigger` menghasilkan sebuah data. Gunakan data berikut:

```json
[
    {
        "product_name": "<nama produk yang ingin anda masukkan>"
    }
]
```

Susun semua blok AI Agent seperti berikut:

![Susunan blok AI Agent](https://lms.sdmdigital.id/pluginfile.php/635494/mod_page/content/25/image27.png)
**Gambar 29. Susunan blok AI Agent.**

Pada komponen `Google Gemini Chat Model`, buka pengaturannya, klik *dropdown*, lalu pilih `Create A New Credential`. Masukkan API *key* Gemini Anda.

**Gambar 30. Menu untuk memasukkan API key.**

**Gambar 31. Pengaturan `Structured Output Parser`.**

Buka blok `AI Agent` dan isi *prompt* berikut:

![Pengaturan blok AI Agent](https://lms.sdmdigital.id/pluginfile.php/635494/mod_page/content/25/image41.png)
**Gambar 32. Pengaturan blok AI Agent.**

```text
Generate 3 diverse search queries to find reviews for: {{ $json.product_name }}

Include queries that target:
1. General review sites and Reddit discussions
2. YouTube video reviews
3. Specialized review platforms (e.g., TrustPilot, G2, Amazon, etc.)

Return ONLY a JSON array of query strings, nothing else.
Example: ["product name reddit reviews", "product name youtube review", "product name trustpilot"]
```

![Visualisasi workflow berjalan](https://lms.sdmdigital.id/pluginfile.php/635494/mod_page/content/25/image51.png)
**Gambar 33. Visualisasi ketika *workflow* sedang berjalan (kiri) dan ketika selesai (kanan).**

![Output node AI Agent](https://lms.sdmdigital.id/pluginfile.php/635494/mod_page/content/25/image34.png)
**Gambar 34. Output *node* AI Agent setelah *workflow* selesai berjalan.**

Setelah menambahkan *node* `code` Python, isi dengan *script* berikut untuk memecah query:

```python
# untuk akumulasi
itemlist = []
for item in _input.all():
  # logika pemrosesan sederhana,
  # ambil semua query,
  # lalu pecah jadi jsonnya masing-masing dalam satu list
  for query in item.json["output"]["query"]:
    itemlist.append({"query": query})
return itemlist
```

![Pemrosesan yang dilakukan](https://lms.sdmdigital.id/pluginfile.php/635494/mod_page/content/25/image5.png)
**Gambar 35. Pemrosesan yang dilakukan**

Tambahkan *node* HTTP *request* untuk memanggil API serper:

![Pengaturan node HTTP request](https://lms.sdmdigital.id/pluginfile.php/635494/mod_page/content/25/image36.png)
**Gambar 36. Pengaturan untuk *node* HTTP *request***

![Setup credentials Serper](https://lms.sdmdigital.id/pluginfile.php/635494/mod_page/content/25/image37.png)
**Gambar 37. Pengaturan untuk *setup* *credentials* Serper pada *node* HTTP *request***

![Kondisi workflow sekarang](https://lms.sdmdigital.id/pluginfile.php/635494/mod_page/content/25/image43.png)
**Gambar 38. Kondisi *workflow* sekarang.**

**Gambar 39. Bermacam pilihan node Tavily.**

Gunakan pengaturan berikut pada node `extract` Tavily:

![Pengaturan node extract](https://lms.sdmdigital.id/pluginfile.php/635494/mod_page/content/25/image25.png)
**Gambar 40. Pengaturan *node* `extract`**

Tambahkan *node* `code` lagi untuk mengumpulkan hasil ekstraksi:

```python
output_dict = {"all_reviews": ""}

formatted_content = ""
num_content = 0
for item in _input.all():
  for result in item.json["results"]:
    formatted_content += f"[Content {num_content + 1} ]\n"
    formatted_content += f"URL: {result['url']}\n"
    formatted_content += f"Query: {result['raw_content']}\n"
    num_content+=1
output_dict["all_reviews"] = formatted_content
return [output_dict]
```

![Keseluruhan workflow](https://lms.sdmdigital.id/pluginfile.php/635494/mod_page/content/25/image41.png)
**Gambar 41. Keseluruhan *workflow* yang dibangun.**

**Gambar 42. Konfigurasi *node* `Convert to File` (kiri), Hasil akhir (kanan)**

**Gambar 43. Laporan akhir yang sudah di-*render***

#### Implementasi *workflow* penuh dengan Langgraph

Instalasi *library*:

```bash
uv add jinja2 langgraph langchain langchain-google-genai tavily-python json-repair
# atau
pip install jinja2 langgraph langchain langchain-google-genai tavily-python json-repair
```

Import *library*:

```python
import json
import os
from typing import List, Optional, TypedDict

import requests
from jinja2 import Template
from json_repair import repair_json
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph
from tavily import TavilyClient
```

Inisiasi dan *State*:

```python
tavily_client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))
MAX_ITERATION = 3

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=os.getenv("API_KEY_LLM")
)

class AgentState(TypedDict):
    product_name: str
    search_queries: List[str]
    used_queries: List[str]
    raw_search_results: List[dict]
    extracted_content: List[dict]
    extracted_reviews: Optional[dict]
    bias_assessment: Optional[dict]
    better_sources: List[str]
    validator_feedback: Optional[str]
    iteration_count: int
    final_report: Optional[str]
    max_total_extracts: int
    is_sufficient: bool
```

### Mini Quiz
*(Konten Interaktif H5P)*

> **Refleksi Singkat**
>
> Setelah memahami perbedaan n8n dan Langgraph, terlihat ada dua sisi dari implementasi *workflow*, yaitu sisi yang visual dengan fokus implementasi yang cepat dan mudah (n8n), serta sisi yang lebih *low-level* dengan kontrol presisi (Langgraph).
>
> Kalau anda diminta implementasi *workflow* untuk Chatbot SOP perusahaan, Asisten untuk *coding*, atau Bot untuk melakukan *riset*, pendekatan mana yang anda akan pilih? Kenapa?