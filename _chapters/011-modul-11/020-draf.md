---
slug: modul-11-2
title: modul-11-2
---
Berikut adalah isi konten yang telah dirapikan ke dalam format Markdown sesuai dengan hirarki aslinya:

# 7.2 Penggunaan Tool, Reasoning, Serta Managemen State Sebagai Komponen Pembangun Sebuah Agent

---

### 7.2.1 Penggunaan Tool

Sebelumnya telah dipelajari secara singkat mengenai *tool* serta penggunaannya pada modul 5 dan modul 10. Sebuah *tool* mengimplementasi fungsi tertentu yang dapat digunakan oleh LLM. Implementasi dan sebuah tool dibagi menjadi dua:

1.  **Implementasi dari *tool* itu sendiri**, umumnya berbentuk fungsi dalam bahasa pemrograman yang digunakan.
2.  **Membuat definisi *tool*** yang berisi:
    1.  Deskripsi mengenai apa kegunaan dari *tool* tersebut, termasuk petunjuk mengenai penggunaannya. Tujuannya adalah agar LLM mampu memahami kapan *tool* tersebut perlu digunakan.
    2.  *Schema* yang mendeskripsikan apa saja *input* yang dibutuhkan *tool* tersebut, termasuk tipenya (teks, angka, objek) serta spesifikasi dari nilainya, misalnya enumerasi.

![Implementasi Tool](https://lms.sdmdigital.id/pluginfile.php/635492/mod_page/content/34/image4.png)
*Gambar 4. Contoh implementasi tool dalam Python serta bentuk definisinya menggunakan format API openAI.*

Saat digunakan, baik definisi *tool* serta petunjuk penggunaan *tool* disampaikan pada LLM sebagai bagian dari *prompt*. Saat sebuah *tool* perlu digunakan, LLM akan mengembalikan sebuah **respon spesial** yang menyimbolkan pemanggilan *tool*. Respon ini kemudian diproses oleh bagian aplikasi yang memanggil LLM untuk mendapatkan informasi **nama tool** yang ingin digunakan oleh LLM serta **parameter yang diberikan LLM** untuk memanggil *tool* tersebut. **Hasil pemanggilan *tool* kemudian dikembalikan lagi ke LLM** untuk melanjutkan proses menghasilkan jawaban. Proses ini bisa diperjelas dengan ilustrasi berikut:

![Alur Penggunaan Tool](https://lms.sdmdigital.id/pluginfile.php/635492/mod_page/content/34/image49.png)
*Gambar 5. Alur LLM menggunakan tool*

Alur dalam gambar tersebut bisa dijelaskan sebagai berikut:
1.  Definisi tool yang sudah dibuat diberikan ke LLM bersamaan dengan prompt dan bagian lainnya dari input (instruksi, konteks lainnya)
2.  Saat LLM memutuskan memanggil *tool*, respon spesial dikeluarkan lalu dikembalikan ke aplikasi pemanggil LLM untuk diproses.
3.  Nama *tool* serta parameternya yang diberikan LLM melalui respon spesial diproses untuk memanggil *tool* terkait.
4.  Hasil pemanggilan tool dikembalikan ke LLM untuk lanjut menghasilkan jawaban.
5.  Jawaban lanjut dihasilkan seperti biasa.

Semua yang disebutkan di atas, baik definisi *tool*, cara penggunaan *tool* serta respon spesial yang menyimbolkan pemanggilan *tool* memiliki bentuk yang berbeda-beda baik pada tingkatan *vendor* maupun *framework* yang digunakan untuk mengembangkan aplikasi AI. Tapi secara umum memiliki proses yang sama secara keseluruhan.

#### Mengapa penggunaan *tool* begitu berguna untuk LLM?

**Penggunaan *tool* memberikan LLM kemampuan untuk melakukan apapun yang bisa dibuat sebagai program**. Definisi *tool* memberikan LLM pengetahuan mengenai apa saja *tool* yang bisa digunakan, dan antarmuka berupa format respon spesial yang diberikan LLM untuk memanggil tool memberikannya cara untuk memanggil sebuah *tool* secara tidak langsung.

**Kalau LLM tanpa tool:**
*   Memiliki pengetahuan terbatas
*   Hanya bisa menghasilkan teks
*   Tidak bisa berinteraksi dengan dunia nyata

**Dengan tool, LLM dapat:**
*   Mencari sendiri informasi yang dibutuhkan (mengakses database, pencarian internet, membaca file)
*   Melakukan aksi (memanggil API, mengirimkan pesan, menjalankan *browser*)
*   Menyimpan informasi secara dinamis (baca tulis *state* ke file / database)
*   Menggunakan model lain yang lebih terspesialisasi, misalnya *tool* yang memanggil model deteksi objek, atau *tool* yang bisa menghasilkan gambar.

Apapun itu, selama bisa dibuat jadi program dan bisa dimengerti LLM, dia bisa dijadikan *tool*, dan bisa digunakan oleh LLM.

#### Bagaimana cara membuat tool yang baik?

Ada beberapa prinsip penting yang perlu diingat ketika membuat tool.

1.  Deskripsi *tool* harus jelas dan menggambarkan **kegunaannya serta batas kemampuannya**. Sama seperti menulis fungsi saat melakukan pemrograman, implementasi *tool* yang dibuat harus diberikan deskripsi yang jelas agar mudah dipahami LLM.
2.  **Meminimalkan side effects (efek samping).** Tool hanya boleh melakukan apa yang tercantum jelas dalam deskripsinya. Efek samping yang tidak dinyatakan dapat menyebabkan perilaku tidak terduga. Jika ada side effect yang diperlukan, sebutkan secara eksplisit dalam deskripsi dan hasil pemanggilannya agar dapat diantisipasi LLM.
    *   **Contoh:** Tool bernama `cek_saldo` yang juga mengurangi saldo pengguna karena ada biaya pengecekan.
3.  *Error* harus dikembalikan secara eksplisit agar diketahui LLM kalau error, misalnya mengembalikan pesan error saja ketika pemanggilan *tool* error.
4.  Setiap *tool* sebaiknya hanya melakukan satu hal saja yang logis sebagai satu kesatuan.
    *   **Contoh:**
        *   ketika mengimplementasi *tool* untuk melakukan *retrieval* pada database RAG, masuk akal kalau keseluruhan prosesnya (mendapatkan *query* → menghitung *embedding* → melakukan *retrieval* → melakukan *re-ranking* → mengembalikan *top-K* hasil) menjadi satu tool, karena sudah satu kesatuan.
        *   Di sisi lain, kalau ada proses kunjungi *website* → terjemahkan konten → ekstraksi teks → simpan ke file, sebaiknya dipisah. Karena tidak semua *website* perlu diterjemahkan, atau mungkin setelah mengunjungi *website*, ada langkah lain dulu sebelum perlu melakukan ekstraksi teks. Kalau prosesnya dinamis, sebaiknya setiap fungsi menjadi *tool* terpisah.

Sama seperti prinsip-prinsip *clean code* dalam membuat sebuah fungsi, inti membuat tool yang baik adalah memastikan satu *tool* memiliki tujuan yang jelas, deskripsi yang transparan, dan perilaku yang konsisten serta dapat diprediksi, sehingga LLM dapat menggunakannya dengan ekspektasi yang tepat dan hasil yang sesuai.

---

### 7.2.2 Hands-on implementasi tool

Untuk implementasi *tool* sederhana, akan digunakan dua pendekatan berbeda untuk mengilustrasikan perbedaan cara definisi dan penggunaan tool, yaitu menggunakan format API openAI secara langsung dan Langchain.

Untuk bagian ini, terlebih dahulu install semua *library* yang diperlukan:

```bash
# pakai UV
uv add openai langchain langchain-google-genai

# pakai Pip
pip install openai langchain langchain-google-genai
```

Untuk memudahkan ilustrasi, akan digunakan *dummy tool* dengan dua fungsi sederhana, untuk memeriksa status pesanan serta untuk memeriksa keranjang belanja.

```python
def check_delivery_status(order_id: str):
    return {
        "order_id": order_id,
        "order_status": "In delivery",
        "order_last_location": "Sedang disortir di DC cakung",
        "ETA": "31-12-2025",
    }

def check_shopping_cart(user_id: str):
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
```

#### Implementasi menggunakan Library OpenAI

Berikut merupakan implementasi LLM dengan pemanggilan *tool* yang diakses menggunakan *library* OpenAI.

```python
# ganti base URL kalau menggunakan provider lain
MODEL_NAME = "gemini-2.5-flash"
client = OpenAI(
    api_key=os.getenv("API_KEY_LLM"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)


# Definisi tool
tools = [
    {
        "type": "function",
        "function": {
            "name": "check_delivery_status",
            "description": "Check the delivery status based on order ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                },
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_shopping_cart",
            "description": "Retrieve the user's shopping cart based on user ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                },
                "required": ["user_id"],
            },
        },
    },
]


tools_dict = {
    "check_delivery_status": check_delivery_status,
    "check_shopping_cart": check_shopping_cart,
}
```

Kemudian, kita buat *system prompt* sederhana:

```python
SYSTEM_PROMPT = """\
Act as a friendly shopping assistant. You will be provided with tools to help you answer the user's queries.
When a tool is not available to help a user's request, suggest to contact a human instead.
Do not make up answers, if a user does not provide the required information, ask them.
Answer should be in fully natural language. It's okay to add small formatting.
Answer in Indonesian by default unless the user asks in english.
"""
```

Implementasi logika dasar pemrosesan pemanggilan *tool*:

```python
# By default menggunakan gemini. Ganti nama model kalau menggunakan provider lain
def chat(query):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]


    # Pemanggilan pertama
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )


    msg = response.choices[0].message


    # Kalau model memanggil tool, responnya akan berisi
    # tool call
    if msg.tool_calls:
        messages.append(msg)
        for call in msg.tool_calls:
            try:
                ## memetakan nama tool ke fungsinya
                tool_func = tools_dict[call.function.name]
                ## print untuk melihat tool apa yang dipanggil untuk chatnya
                print(f"Tool {call.function.name} was called.")
                args = json.loads(call.function.arguments)


                # LLM cuma memberikan parameter dan menentukan tool yang dipanggil
                # implementasi toolnya dipanggil dalam kode
                results = tool_func(**args)


                # Cara khusus untuk mengembalikan hasil tool call
                # ke model untuk dipakai menghasilkan jawaban
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": json.dumps(results),
                    }
                )
            # error handling kalau tool tidak ditemukan (LLM halusinasi nama tool)
            except KeyError:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": json.dumps(
                            {
                                "status": "error",
                                "message": f"Tool {call.function.name} does not exist.",
                            }
                        ),
                    }
                )
            # error handling kalau ada error lain
            except Exception as ex:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": json.dumps(
                            {
                                "status": "error",
                                "message": f"Tool {call.function.name} errored with an unknown error: {ex}."
                                "Give generic response to the user that something is wrong.",
                            }
                        ),
                    }
                )
        # lanjut menghasilkan jawaban
        final = client.chat.completions.create(model=MODEL_NAME, messages=messages)
        return final.choices[0].message.content


    # Kalau modelnya nggak manggil tool,
    # langsung berikan saja jawabannya
    return msg.content
```

Pengujian dengan 3 kasus input:

```python
print(">> Pemanggilan 1, perlu tool check status pengiriman <<")
print(chat("Halo, saya ingin mengecek status pengiriman pesanan saya dengan ID 1112231233"))
print("====== separator ======")
print(">> Pemanggilan 2, perlu tool check keranjang belanja <<")
print(
    chat("Halo, saya ingin mengecek keranjang belanja saya dengan user ID Ivan#19827")
)
print("====== separator ======")
print(">> Pemanggilan 3, tidak perlu tool apapun, cuma pertanyaan biasa <<")
print(
    chat("Halo, apa kabar? Siapa kamu?")
)
```

**Output:**
```text
>> Pemanggilan 1, perlu tool check status pengiriman <<
Tool check_delivery_status was called.
Terima kasih sudah menunggu! Berdasarkan ID pesanan Anda **1112231233**, status pesanan Anda saat ini adalah **Dalam pengiriman**.

Pesanan Anda terakhir terdeteksi sedang **disortir di DC Cakung**. Perkiraan waktu tiba (ETA) adalah tanggal **31 Desember 2025**.

Apakah ada hal lain yang bisa saya bantu?
====== separator ======
>> Pemanggilan 2, perlu tool check keranjang belanja <<
Tool check_shopping_cart was called.
Baik, Ivan#19827! Saya sudah berhasil mengecek keranjang belanja Anda.

Berikut adalah daftar barang yang ada di keranjang Anda:

*   **babibas running shoes**
    *   Ukuran: 48
    *   Warna: Merah muda cerah
*   **bortusten performance socks**
    *   Ukuran: XL
    *   Warna: Kuning cerah
*   **becs anti-slip shoe lace**
    *   Warna: Cyan

Apakah ada hal lain yang bisa saya bantu?
====== separator ======
>> Pemanggilan 3, tidak perlu tool apapun, cuma pertanyaan biasa <<
Halo! Saya adalah asisten belanja Anda yang siap membantu. Ada yang bisa saya bantu hari ini? 😊
```

#### Implementasi menggunakan Framework LangChain

Berbeda dengan menggunakan API berformat openAI, *framework* seperti `langchain` memberikan kemudahan untuk berbagai macam hal, termasuk membuat definisi *tool*.

```python
import os
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.tools import tool


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
```

Inisialisasi Agent:

```python
os.environ["GOOGLE_API_KEY"] = os.getenv("API_KEY_LLM")
llm = init_chat_model("google_genai:gemini-2.5-flash")

agent = create_agent(
    model=llm,
    tools=[check_delivery_status, check_shopping_cart],
    system_prompt=SYSTEM_PROMPT,
)

def chat(query):
    result = agent.invoke({"messages": [{"role": "user", "content": query}]})
    return result["messages"][-1].content
```

Sama seperti sebelumnya, kita uji implementasinya:

**Hasilnya:**
```text
>> Pemanggilan 1, perlu tool check status pengiriman <<
Tool check_delivery_status was called with order_id 1112231233
Tentu, saya bisa bantu!
Pesanan Anda dengan ID **1112231233** saat ini berstatus **Dalam Pengiriman**.
...
```

---

### 7.2.3 Reasoning

#### Ketika hanya menggunakan tool tidak cukup

Sebuah *tool* hanya memberikan LLM kemampuan untuk untuk memenuhi peran observasi dan aksi dari sebuah *Agent*. Perlu ditentukan ***tool*** **apa yang perlu dipanggil**, **kapan harus dipanggil**, dan **bagaimana urutan pemanggilannya**. Tidak hanya itu, ada kalanya pemanggilan sebuah *tool* mempengaruhi alur maupun *progress* pencapaian sebuah tujuan, sehingga perlu perencanaan ulang.

LLM juga perlu memenuhi peran untuk “berpikir”, yang bisa dicapai dengan pola-pola *reasoning*.

#### ReAct: Reasoning and Acting

Salah satu pola *reasoning* yang populer digunakan untuk implementasi sebuah *agent* adalah ReAct. ReAct terdiri dari dua langkah:
1.  ***Reasoning;*** Memikirkan langkah selanjutnya berdasarkan rencana awal maupun setelah melakukan aksi dan mendapatkan observasi baru.
2.  ***Acting;*** Melakukan suatu aksi yang bisa mengubah keadaan lingkungan dan memunculkan observasi baru.

![Alur ReAct](https://lms.sdmdigital.id/pluginfile.php/635492/mod_page/content/34/image1.png)
*Gambar 6. Alur sistem ReAct*

---

### 7.2.4 Manajemen State

#### Pentingnya *state* bagi sebuah *Agent*

Selama menjalankan proses penyelesaian sebuah *task*, sebuah *Agent* perlu menyimpan *working memory* dalam bentuk *state* yang bertujuan:
*   Mengingat apa tujuan yang ingin dicapai
*   Memantau *progress* dalam mencapai tujuan tersebut
*   Mengingat langkah apa saja yang sudah diambil sebelumnya
*   Merekap hasil observasi perubahan lingkungan sekitarnya

#### Bagaimana bentuk penyimpanan *state* yang ideal?

Ada dua pendekatan menyimpan *state*:
1.  **Sejarah interaksi:** Disimpan sebagai percakapan (aksi disertai observasi).
2.  **Informasi terstruktur:** Menyimpan data dalam format seperti JSON.

**Perbandingan Sejarah Interaksi (Abstrak) vs Terstruktur:**

| Jenis State | Contoh Konten |
| :--- | :--- |
| **Abstrak (Teks)** | LLM: "User ingin mengunjungi website... Gunakan `web_browser_tool`" <br> Web_browser_tool: "Browser sudah dibuka..." <br> LLM: "Saatnya mengecek data dengan `inspect_page_tool`" |
| **Terstruktur (JSON)** | `{"is_browser_open": true, "current_page": "www.asdfghjkl.xyz", "data_found": false, ...}` |

Informasi terstruktur lebih diunggulkan karena lebih mudah diproses secara programatis, hemat konteks, dan dapat divalidasi menggunakan *schema* (seperti Pydantic).

#### Di luar state: Manajemen memory jangka panjang

Penelitian *“Cognitive Architectures for Language Agents”* (CoALA) memisahkan bentuk-bentuk *memory* ini ke dalam kategori:

1.  ***Semantic memory:*** Menyimpan pengetahuan mengenai dunia dan dirinya sendiri (misal: RAG).
2.  ***Episodic memory:*** Menyimpan catatan keputusan yang pernah diambil (*few-shot examples*).

![Bentuk Memory Agent](https://lms.sdmdigital.id/pluginfile.php/635492/mod_page/content/34/image21.png)
*Gambar 7. Bentuk memory yang dimiliki sebuah Agent selain state (working memory). Mekanisme retrieval terpisah diperlukan untuk mengakses memory jangka panjang.*

#### Best practices manajemen *state*

1.  Tetapkan skema yang baku (misal: Pydantic).
2.  Hanya simpan informasi yang penting.
3.  Simpan setiap perubahan untuk keperluan *debugging*.
4.  Amankan *state* dari modifikasi manual oleh *user*.
5.  Ambil hanya nilai yang relevan untuk menghindari *overload* konteks.
6.  Modelkan transisi antar *state* dengan jelas.

---

### Mini Quiz
*(Konten Interaktif H5P)*

> **Refleksi Singkat**
>
> Sebuah agent membutuhkan state jangka pendek dan memory jangka panjang agar dapat menyelesaikan task yang kompleks dan berulang. State membantu agent memantau progres dan konteks pada satu sesi, sementara semantic dan episodic memory membantu agent membawa pembelajaran lintas sesi.
>
> Jika kamu harus merancang agent yang bertugas memproses dokumen hukum jangka panjang untuk satu klien, bagaimana kamu akan membagi peran antara state, semantic memory, dan episodic memory? Jelaskan desainmu dan bagaimana masing-masing bentuk memory berkontribusi terhadap akurasi, efisiensi, dan kemampuan adaptasi agent.