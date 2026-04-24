---
slug: modul-8-4
title: modul-8-4
---
# 4.4 Manajemen State

---

Sebelumnya sudah disinggung bahwa LLM tidak menyimpan *state*. Apa artinya?

Ketika memanggil LLM, setiap pemanggilan hanya bergantung pada input yang diberikan pada waktu itu. Pemanggilan selanjutnya hanya bergantung pada input yang selanjutnya diberikan. Artinya jika pada pemanggilan pertama anda meminta LLM menjelaskan tentang AI, lalu pada pemanggilan kedua anda bertanya mengenai sesuatu yang barusan dijelaskan, LLM tidak akan mengerti apa yang anda maksud.

Tapi kalau kita menggunakan aplikasi seperti ChatGPT, kita bisa bertanya mengenai sesuatu yang sebelumnya dibicarakan pada satu sesi percakapan. Tidak hanya itu, bahkan ChatGPT bisa memahami sifat dan perilaku kita melalui interaksi-interaksi sebelumnya dalam sesi percakapan yang lain. Kalau LLM tidak mampu menyimpan state, bagaimana caranya aplikasi seperti ChatGPT bisa melakukan hal ini?

Jawabannya terletak pada sistem aplikasi AI yang dikembangkan. Karena LLM tidak mampu menyimpan state, maka state ini dimanage oleh sistem. Ketika informasi yang berkaitan dengan state ini diperlukan, sistem akan menyuntikkan informasi ini secara dinamis. Submodul ini akan menjelaskan bagaimana hal ini umumnya dilakukan. Tapi sebelum berbicara mengenai implementasi, mari kenali dulu apa saja bentuk state yang mungkin perlu disimpan.

---

## 4.4.1 Apa saja yang Termasuk "State" dalam Aplikasi AI?

Dalam konteks aplikasi berbasis LLM, **state adalah informasi yang harus disimpan di luar model dan disuntikkan kembali ketika dibutuhkan pada panggilan model agar LLM dapat memahami apa yang sudah terjadi dan apa yang harus dilakukan selanjutnya**. Bentuk state sendiri bermacam-macam, tergantung aplikasi yang dikembangkan, misalnya:

*   Sejarah percakapan untuk chatbot
*   Langkah yang sudah dilakukan dan hasilnya dalam sebuah multi-step workflow.
*   Profil seorang pengguna yang menyimpan preferensi berdasarkan interaksi sebelumnya
*   State eksternal, misalnya data apa saja yang sudah ditulis ke database, atau kondisi terkini dari sebuah codebase

Sebuah sistem yang kompleks pastinya memiliki state dengan sebuah kondisi awal yang kemudian diupdate dengan setiap aksi yang dilakukan oleh LLM saat sistem tersebut berjalan. Misalnya:

*   Setiap kali user mengirimkan chat baru, state percakapan diupdate untuk menambahkan chat baru ini beserta respon dari LLM
*   Setiap kali sebuah bot selesai melakukan suatu task dalam sebuah to-do-list, state berisi daftar task tersebut diupdate untuk menandai task yang sudah selesai dilakukan. Iterasi selanjutnya kemudian menggunakan state tersebut untuk memilih task yang belum dikerjakan.

---

## 4.4.2 Bagaimana Cara Menyuntikkan State ke LLM?

State sebenarnya sama seperti informasi atau konteks lainnya. State bisa dimasukkan sebagai bagian dari prompt. Berikut merupakan contoh sederhana memasukkan state ke dalam prompt secara eksplisit:

```python
from jinja2 import Template
from openai import OpenAI

client = OpenAI(
    api_key="GANTI DENGAN API KEY GEMINI",
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

SYSTEM_PROMPT = """
Jawab pertanyaan user menggunakan bahasa indonesia gaul dengan gaya jakselian.
Outputmu tidak perlu diberi formatting, cukup plain text saja.
"""


USER_PROMPT = Template("""
Berikut merupakan sejarah percakapan sebelumnya. Gunakan informasi ini untuk menjawab user.

{{history}}

{{user_message}}
""")

history = []

message1 = "Halo! namaku bobi. umurku 19 tahun. rumahku ada di jakarta."

response = client.chat.completions.create(
    model="gemini-2.5-flash",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_PROMPT.render(history=history, user_message=message1)
        }
    ]
)

reply1 = response.choices[0].message.content
print(reply1)

history.append(f"USER: {message1}")
history.append(f"ASSISTANT: {reply1}")

message2 = "Siapa namaku? 6 tahun lagi, berapa umurku?"

response = client.chat.completions.create(
    model="gemini-2.5-flash",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_PROMPT.render(history=history, user_message=message2)
        }
    ]
)


reply2 = response.choices[0].message.content
print(reply2)
```

Untuk library OpenAI, spesifik untuk use case chat, menambahkan sejarah percakapan bisa langsung dilakukan dengan menambahkan chat dan respon model sebelumnya ke dalam list message yang diberikan ke fungsi untuk memanggil LLM. Contohnya sebagai berikut:

```python
from openai import OpenAI

client = OpenAI(
    api_key="GANTI DENGAN API KEY GEMINI",
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

SYSTEM_PROMPT = """
Jawab pertanyaan user menggunakan bahasa gaul dengan gaya jakselian.
Outputmu tidak perlu diberi formatting, cukup plain text saja.
"""

message1 = "Halo! namaku bobi. umurku 19 tahun. rumahku ada di jakarta."

response = client.chat.completions.create(
    model="gemini-2.5-flash",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": message1
        }
    ]
)


reply1 = response.choices[0].message.content
print(reply1)

message2 = "Siapa namaku? 6 tahun lagi, berapa umurku?"

response = client.chat.completions.create(
    model="gemini-2.5-flash",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": message1
        },
        {
            "role": "assistant",
            "content": reply1
        },
        {
            "role": "user",
            "content": message2
        }
    ]
)

reply2 = response.choices[0].message.content
print(reply2)
```

Dapat dilihat bahwa state bisa dipandang sebagai salah satu jenis konteks, dan aturan yang sama yang sudah dibahas dalam context engineering juga berlaku untuk state. Tapi berbeda dengan konteks pada umumnya, state bisa berubah-ubah dengan frekuensi yang cukup tinggi. Setiap pemanggilan LLM bisa merubah state yang disimpan. Sehingga walaupun bisa dipandang sebagai salah satu bentuk konteks, perlu perhatian khusus untuk me-manage state ini dalam aplikasi yang dikembangkan.

---

## 4.4.3 Pola Manajemen State untuk Aplikasi AI

Ada dua hal yang perlu diperhatikan dalam memanage state:
1.  Bagaimana cara menyimpan dan membacanya
2.  Format penyimpanannya seperti apa

### Menyimpan dan membaca state
Ketika sedang tidak digunakan, state perlu disimpan untuk sewaktu-waktu dibaca kembali saat dibutuhkan. Pola penyimpanan dan pembacaan ini bergantung sifat dari aplikasi yang dikembangkan. Ada beberapa cara yang umum digunakan untuk baca tulis state:
*   Menyimpan state dalam memory program yang sedang berjalan (temporer)
*   Menyimpan state pada penyimpanan external
*   Menggunakan tool

### Di mana statenya disimpan?
Jika state hanya diperlukan dalam jangka pendek, misalnya untuk sebuah workflow singkat atau satu sesi interaktif yang sejarahnya tidak perlu disimpan, state dapat langsung disimpan pada memory program yang sedang berjalan. Salah satu kelemahan dari pendekatan ini adalah apabila terjadi kegagalan yang menyebabkan crash, state ini bisa hilang dan tidak bisa di-recover untuk lanjut dijalankan setelah aplikasi sudah kembali berjalan.

![Gambar 38. Jenis-jenis State dalam Aplikasi AI](https://lms.sdmdigital.id/pluginfile.php/635458/mod_page/content/15/image16.jpg)
**Gambar 38. Melakukan manajemen state di dalam aplikasi.**

Untuk menangani masalah ini, state dapat disimpan secara external pada penyimpanan yang terpisah dari aplikasi. Harapannya adalah jika aplikasi mengalami crash atau aplikasinya down, state masih bisa tersimpan dan dibaca kembali setelah aplikasi sudah berjalan lagi.

![Gambar 39. Menyimpan state di luar aplikasi dalam sistem sendiri](https://lms.sdmdigital.id/pluginfile.php/635458/mod_page/content/15/image20.jpg)
**Gambar 39. Menyimpan state di luar aplikasi dalam sistem sendiri.**

Ada kalanya state perlu disimpan secara jangka panjang. Salah satu contohnya adalah profil pengguna aplikasi yang ditentukan melalui sejarah interaksinya dengan aplikasi. Contoh lainnya adalah sejarah percakapan atau interaksi jangka panjang yang sesinya berlangsung lama atau perlu disimpan sejarahnya untuk sewaktu-waktu dilanjutkan kembali. Terkadang beberapa use case juga membutuhkan state ini disimpan untuk keperluan seperti debugging maupun monitoring.

Apapun alasannya, jika perlu disimpan dalam jangka panjang, state bisa disimpan pada filesystem dalam bentuk file. Apabila ada concern mengenai kapasitas memory, state bisa diberikan lifetime, yaitu waktu maksimal dimana setelah terlewati, state ini akan dihapus secara otomatis, misalnya jika terlalu lama tidak digunakan.

### Bagaimana mengakses state?
Selain memikirkan dimana state akan disimpan, metode pengaksesannya juga perlu dipikirkan. Ada dua cara menyimpan state, yaitu secara manual dilakukan dalam aplikasi dengan langsung membaca dan menulis state dari dan ke tempat penyimpanannya. Untuk workflow yang lebih kompleks maupun untuk menyederhanakan implementasi, pengaksesan state baik untuk membaca maupun menulisnya bisa dilakukan menggunakan sebuah tool yang secara langsung bisa dipanggil sendiri oleh LLM.

### Format penyimpanan state
State bisa disimpan dalam format apapun yang diserahkan kepada developer untuk menentukannya. Umumnya ada beberapa pertimbangan untuk penyimpanan state:
*   Terstruktur (JSON, XML, Markdown, format lainnya) vs tidak terstruktur (hanya teks)
*   Raw (tidak diubah / diproses) vs Diproses (dirangkum atau diubah menjadi format tertentu)

Umumnya state disimpan dalam bentuk data terstruktur, karena selain mudah dipahami oleh LLM, formatnya juga menjadi baku dan memudahkan juga apabila perlu diubah melalui proses manual.

Untuk menentukan apakah state perlu disimpan apa adanya maupun dalam bentuk terproses akan sangat bergantung pada use case. Misalnya use case seperti chatbot mungkin lebih membutuhkan sejarah percakapan yang dirangkum, terutama untuk percakapan berjangka panjang. Sedangkan pada workflow otomatis, bentuk state yang sudah terstruktur diperlukan apa adanya untuk menjaga detail-detail yang disimpan dalam state.

Bisa juga dilakukan pendekatan hybrid, yaitu state disimpan dalam bentuk raw lalu diproses saat dibaca. Metode ini memberikan the best of both worlds, karena ada kalanya diperlukan fokus pemrosesan state yang berbeda. Misalnya, Rangkuman percakapan mungkin bisa dibuat dengan memfokuskan pada topik-topik yang sedang dibicarakan sekarang dibanding semua topik yang ada dalam sejarah percakapan, sehingga secara berkala nilai state perlu diproses ulang dari raw state.

---

## 4.4.4 Hands on Manajemen State: Rolling Summary untuk Percakapan Panjang

Untuk sesi hands on, kita akan mengimplementasikan state management sederhana untuk percakapan. Sebagai basis, kode yang sebelumnya digunakan akan diubah agar bisa dijalankan secara interaktif.

Kita akan menerapkan salah satu bentuk kompresi konteks yang sudah disebut pada submodul sebelumnya mengenai context engineering, yaitu dengan cara summarization atau merangkum.

```python
from openai import OpenAI
import os

client = OpenAI(
    api_key=os.getenv("API_KEY_LLM"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# bungkus dalam sebuah kelas agar mudah manajemennya
class ChatWithSummary:
    def __init__(self, model="gemini-2.5-flash", summary_interval=10):
        self.model = model
        # setiap kali chat sebanyak summary_interval,
        # kita lakukan summary
        # karena setiap percakapan ada 2 chat (user dan asisten)
        # artinya setiap 5 kali
        self.summary_interval = summary_interval
        self.messages = []
        self.summary = None
        self.message_count = 0
   
    # untuk tracking panjang sejarah
    # percakapan
    def add_message(self, role, content):
        self.messages.append({"role": role, "content": content})
        self.message_count += 1
   
    # untuk menghitung summary ketika diperlukan
    def get_summary(self):
        # kalau belum ada percakapan apapun
        if len(self.messages) < 2:
            return None
       
        # prompt sederhana untuk menyederhanakan sejarah percakapan
        # menjadi summary yang singkat
        summary_prompt = [
            {"role": "system", "content": "Summarize the following conversation concisely, capturing key points and context."},
            *self.messages,
            {"role": "user", "content": "Based on previous chat history, give me the summary."}
        ]
       
        response = client.chat.completions.create(
            model=self.model,
            messages=summary_prompt
        )
        return response.choices[0].message.content
   
    # logika pengecekan apakah perlu generate summary
    def should_summarize(self):
        return self.message_count >= self.summary_interval
   
    # logika untuk menghasilkan summary
    # kalau sejarah percakapan sudah terlalu banyak, buat summary,
    # lalu potong hanya percakapan terkini saja
    # sisanya dari summary
    def create_rolling_summary(self):
       
        print("\n[Membuat summary...]")
        self.summary = self.get_summary()
       
        # potong sejarah percakapan
        # karena informasinya sudah disimpan di
        # summary
        recent_messages = self.messages[-2:]
        self.messages = recent_messages
        # reset dari awal
        self.message_count = 0
       
        print(f"[Summary: {self.summary}]\n")
   
    # untuk mendapatkan konteks percakapan secara dinamis, disuntikkan melalui
    # system prompt dan sejarah percakapan sebelumnya
    def get_context(self):
        context = []
       
        if self.summary:
            context.append({
                "role": "system",
                "content": f"You are a helpful assistant. Answer in indonesian. Previous conversation summary: {self.summary}"
            })
        else:
            context.append({
                "role": "system",
                "content": "You are a helpful assistant. Answer in indonesian."
            })
        # masukan sejarah percakapan
        context.extend(self.messages)
       
        return context
   
    # metode / logika utama
    def chat(self, user_message):
       
        # setiap kali ada pesan, langsung masukkan ke sejarah percakapan
        self.add_message("user", user_message)
       
        response = client.chat.completions.create(
            model=self.model,
            # tidak perlu manual membuat messages lagi
            # karena sudah dihandle logikanya
            messages=self.get_context()
        )
       
        assistant_message = response.choices[0].message.content
        # respon asisten juga masuk sejarah percakapan
        self.add_message("assistant", assistant_message)
       
        # di akhir, menentukan apakah perlu summary
        if self.should_summarize():
            self.create_rolling_summary()
       
        return assistant_message

if __name__ == "__main__":
    chat = ChatWithSummary(summary_interval=8)
   
    print("Silahkan berbincang dengan bot! ketik 'quit' atau 'exit' untuk berhenti.\n")
   
    while True:
        user_input = input("Anda: ")
       
        if user_input.lower() in ['quit', 'exit']:
            print("Sampai jumpa!")
            break
       
        response = chat.chat(user_input)
        print(f"Assisten: {response}\n")
       
        print(f"[Panjang sejarah percakapan: {len(chat.messages)}]")
```

Berikut adalah contoh sejarah percakapan yang menunjukkan terjadinya proses kompresi konteks melalui summarization. Perhatikan bagian yang ditebalkan.

```text
Anda: Halo!
Asisten: Halo! Apa yang bisa saya bantu Anda? 😊

[Panjang sejarah percakapan: 2]
Anda: saya antonio dari italia
Asisten: Halo, Antonio! Senang bertemu dengan Anda. Anda dari Italia, ya? Apa yang bisa saya bantu Anda? 😊

[Panjang sejarah percakapan: 4]
Anda: apakah anda tau soal sepakbola italia? kami punya legenda seperti paolo maldini dan roberto baggio
Asisten: Tentu, saya tahu banyak hal tentang sepakbola Italia! Paolo Maldini dan Roberto Baggio adalah dua legenda yang sangat terkenal di Italia. Maldini adalah bek kelas satu yang bermain untuk AC Milan dan menang banyak trofi, termasuk empat gelar Serie A. Baggio adalah penyerang yang sangat khas, dengan kemampuan mencetak gol dan teknik yang luar biasa. Apa yang ingin Anda tahu tentang mereka? Saya siap membantu! 😊

[Panjang sejarah percakapan: 6]
Anda: italia juga punya makanan enak dan alam yang menakjubkan

[Membuat summary...]
[Summary: Berikut ringkasan percakapan kami:

- **Antonio** mengatakan bahwa ia berasal dari Italia.
- Kedua pihak membicarakan **legenda sepakbola Italia**, khususnya **Paolo Maldini** (bek luar biasa dari AC Milan) dan **Roberto Baggio** (penyerang terkenal).
- Mereka juga membahas **masakan dan alam Italia**, yang dikenal lezat dan indah, seperti pasta, pizza, Dolomites, Tuscany, dan Pantai Liguria.
- Percakapan berjalan dengan antusias, dan ada kesan ingin tahu lebih lanjut tentang topik-topik tersebut.]

Asisten: Benar, Italia memang terkenal dengan masakan lezatnya dan alam yang menakjubkan! Makanan Italia seperti pasta, pizza, risotto, dan banyak lagi adalah favorit banyak orang. Sementara itu, alam 
Italia memiliki gunung, sungai, laut, dan pemandangan yang indah, seperti Dolomites, Tuscany, dan Pantai Liguria. Apakah Anda ingin saya rekomendasikan makanan atau tempat wisata di Italia? 😊

[Panjang sejarah percakapan: 2]
Anda: Siapa nama saya?
Asisten: Nama Anda adalah **Antonio**. 😊 Apakah Anda ingin berbicara lebih lanjut tentang Italia atau topik lain?
```

Setelah percakapan soal bentang alam, sejarah percakapan mengenai nama dari pengguna sudah dihilangkan. Tapi LLM tetap bisa mengetahuinya melalui summary yang disimpan.

---

### Mini Quiz
*(Elemen interaktif H5P di LMS)*

---

### Refleksi Akhir

Bayangkan anda diminta membuat sistem yang terdiri dari banyak layanan berbeda, masing-masing memiliki sebuah LLM. Anda perlu memastikan sebuah state dapat dengan baik disampaikan dari satu bagian sistem ke bagian lainnya.

*   Seperti apa anda akan membuat protokol membagi konteks / state ini?
*   Format apa yang anda gunakan?
*   Bagaimana anda akan menangani state untuk kebutuhan yang berbeda-beda?

Hal ini penting untuk beberapa jenis sistem LLM yang terdiri dari lebih dari satu sistem “cerdas” yang masing-masing dikendalikan oleh sebuah LLM yang bekerjasama untuk menyelesaikan suatu masalah.