---
slug: modul-5-2
title: modul=5-2
---

# 1.2 Persiapan Data untuk Training
---

Sebelum melanjutkan ke pengumpulan data, ada satu hal yang sangat penting untuk dipahami. Mengumpulkan data secara manual memerlukan banyak tenaga, waktu, dan resource. Kecuali memang ada kebutuhan yang penting, sebaiknya memaksimalkan terlebih dahulu data yang sudah tersedia untuk melatih sebuah model, lalu hanya mengumpulkan data untuk tujuan spesifik pengembangan model untuk fine tune model yang sudah dilatih baik melalui SFT maupun CPT jika jumlah datanya memungkinkan.

Atau lebih baik lagi, langsung saja gunakan model yang sudah dilatih yang performanya cukup menjanjikan untuk tujuan pengembangan model. Modul ini hanya akan membahas gambarannya saja bagaimana proses mendapatkan data apabila memang perlu training dari awal.

---

## 1.2.1 Kapan perlu mengumpulkan data?

Ada dua tujuan yang ingin dicapai ketika melakukan pengembangan model:
1.  **Pengembangan Arsitektur/Metode Baru:** Ingin membandingkan arsitektur atau metode training baru dengan yang lain. Dalam kasus ini, prioritasnya adalah menggunakan data yang sudah ada, yang juga digunakan melatih LLM lain.
2.  **Kebutuhan Skenario Spesifik:** Misalnya LLM untuk Bahasa Indonesia, hukum, medis, atau domain khusus lainnya. Dalam kasus ini, perlu dicari apakah tersedia data yang mendekati. Jika tidak ada, barulah berpindah ke pengumpulan data manual.

---

## 1.2.2 Menggunakan data yang sudah tersedia di Hugging Face

Huggingface menyediakan berbagai macam dataset hasil kontribusi pengguna (individu, akademik, maupun industri). Dataset ini mencakup data teks besar untuk *pretraining* hingga data spesifik untuk *post-training*.

![Datasets HuggingFace](https://lms.sdmdigital.id/pluginfile.php/635411/mod_page/content/24/image35.png)
**Gambar 12. Tampilan halaman datasets pada HuggingFace**

Setiap data disimpan dalam bentuk repository dengan **dataset card** yang berfungsi mirip dengan `README.md` di GitHub untuk menjelaskan detail dataset.

![Repository GSM8K](https://lms.sdmdigital.id/pluginfile.php/635411/mod_page/content/24/image21.png)
**Gambar 13. Tampilan repository dataset gsm8k di Hugging Face**

![Dataset Card Details](https://lms.sdmdigital.id/pluginfile.php/635411/mod_page/content/24/image39.png)
**Gambar 14. Detail mengenai data GSM8K yang diberikan pada dataset card nya**

Untuk mencari dataset tertentu, gunakan filter di bagian kiri halaman untuk menyaring berdasarkan modalitas, bahasa, ukuran, format, atau tugas bahasa.

![Filter Dataset](https://lms.sdmdigital.id/pluginfile.php/635411/mod_page/content/24/image38.png)
**Gambar 15. Filter yang bisa digunakan untuk memilih dataset dengan modalitas, ukuran, format, task bahasa, maupun detil tertentu lainnya.**

Sebagai contoh, pada dataset `wikipedia-monthly`, terdapat subfolder khusus untuk berbagai bahasa, termasuk Indonesia.

![Folder Bahasa Indonesia](https://lms.sdmdigital.id/pluginfile.php/635411/mod_page/content/24/image28.png)
**Gambar 16. Berbagai folder dalam dataset wikipedia-monthly yang mewakili dataset untuk setiap bahasa, termasuk Indonesia dalam folder 20250701.id**

#### Mendownload Dataset
Anda bisa mengunduh file secara manual, namun cara yang lebih praktis adalah menggunakan library Python `datasets`.

![File Download Manual](https://lms.sdmdigital.id/pluginfile.php/635411/mod_page/content/24/image29.png)
**Gambar 17. Tampilan file individual yang bisa didownload dalam sebuah repository dataset HuggingFace**

**Instalasi library:**
```bash
# menggunakan uv
uv add datasets
# menggunakan pip
pip install datasets
```

**Contoh kode penggunaan:**
```python
from datasets import load_dataset, get_dataset_config_names, get_dataset_config_info

# Mengecek konfigurasi apa saja yang tersedia
configs = get_dataset_config_names("omarkamali/wikipedia-monthly")
print(configs)
# Output: ['20250701.ab', '20250701.ace', ..., 'latest.id', ...]

# Melihat informasi konfigurasi spesifik (misal: split yang tersedia)
print(get_dataset_config_info("omarkamali/wikipedia-monthly", "latest.id"))

# Mendownload split 'train' dari konfigurasi 'latest.id'
# Catatan: Akan mengunduh data dalam ukuran besar (GB)
data = load_dataset("omarkamali/wikipedia-monthly", "latest.id", split="train")
```

Dataset di HuggingFace tersedia dalam berbagai format seperti JSON, CSV, Parquet, atau teks biasa. Anda dapat menggunakan **Dataset Viewer** untuk melihat struktur data secara langsung di browser.

![Dataset Viewer](https://lms.sdmdigital.id/pluginfile.php/635411/mod_page/content/24/image12.png)
**Gambar 18: Dataset Viewer pada Hugging Face**

**Penting:** Perhatikan lisensi dataset sebelum digunakan, terutama untuk tujuan komersial, karena lisensi membatasi hak penggunaan data tersebut.

## 1.2.3 Mengumpulkan data secara mandiri

Kita tahu setiap tahap training LLM memerlukan jenis data yang berbeda:
*   **Pretraining** memerlukan data teks berjumlah besar untuk melatih secara unsupervised.
*   **Post training** memerlukan dua jenis data berbeda:
    *   Pasangan input dan output untuk tahap instruction tuning maupun tahap lainnya yang supervised.
    *   Data preferensi berupa 1 input, 2 atau lebih output dan preferensi setiap output.

Umumnya ketika mengumpulkan data, ada dua tahap:
1.  **Mengumpulkan data.** Biasanya dengan crawling atau dibuat manual oleh tim annotator.
2.  **Membersihkan data.** Teks tidak selalu berbentuk ideal, jadi perlu dibersihkan dulu, misalnya menghilangkan duplikasi.
3.  **Validasi / pelabelan data.** Data yang dikumpulkan dan sudah disiapkan perlu dilabeli atau divalidasi oleh domain expert.

Mari kita bahas satu per satu.

---

## 1.2.4 Mengumpulkan data

Secara umum, kita bisa membagi data yang ingin dikumpulkan menjadi dua, yaitu data pretraining yang berupa teks dari berbagai domain dalam jumlah yang banyak, serta data post training spesifik use case / kemampuan tertentu yang sudah dilabeli.

Ada beberapa pertimbangan dalam mengumpulkan data:
1.  **Jumlah data.** Semakin banyak data yang digunakan untuk training, dengan asumsi data yang banyak tersebut mencakup berbagai macam kasus, skenario, maupun variasi yang berbeda, maka semakin baik performa model yang dilatih.
2.  **Keragaman.** Cara kita mempersiapkan sebuah model agar mampu menangani keragaman kasus penggunaannya adalah dengan secara eksplisit melatihnya dengan data yang beragam.
3.  **Relevansi dengan kemampuan tertentu.** Jika kita ingin model bisa mengerti bahasa indonesia, maka kumpulkan data dalam bahasa indonesia. Jika kita ingin model bisa memahami istilah medis, maka kita latih menggunakan literatur medis.

Data bisa didapatkan melalui bermacam cara:
1.  Data sintesis menggunakan LLM
2.  Crawling dari internet
3.  Pengumpulan manual. Ini adalah tahap terakhir yang dipilih kalau memang mau tidak mau perlu menyusun data langsung oleh expert manusia.

---

## 1.2.5 Data sintetis menggunakan LLM

Pendekatan ini biasanya lebih umum digunakan untuk membuat dataset untuk post-training yang spesifik pada task tertentu. Kita mengetahui bahwa LLM bisa berhalusinasi atau bisa salah mengerti instruksi, sehingga walaupun datanya bisa dihasilkan secara otomatis, tetap perlu ada proses manual di akhir untuk memastikan kualitas data yang dihasilkan.

Selain itu, diperlukan strategi implementasi yang tepat agar data yang dihasilkan memiliki variasi keadaan yang baik, cakupan skenario yang luas, serta sifat lainnya yang diinginkan dari sebuah dataset. Bisa dibilang proses ini seperti membangun aplikasi LLM yang tujuannya adalah menghasilkan dataset.

**Studi kasus 1: Dataset [Self instruct](https://arxiv.org/abs/2212.10560)**

Untuk membangun self-instruct, tim penelitiannya membuat terlebih dahulu 175 seed data, dimana setiap data terdiri dari:
*   Instruksi ke LLM untuk membuat pasangan pertanyaan dan jawaban berdasarkan kriteria tertentu.
*   Sebuah instance, yaitu respon dari LLM yang terdiri dari pasangan pertanyaan dan jawaban.

Seed data ini berfungsi sebagai data awal yang kemudian divariasikan, caranya adalah LLM diberikan instruksi yang sudah dibuat sebagai contoh, lalu diminta membuat variasi lain dari instruksi-instruksi tersebut.

Keseluruhan prosesnya seperti ini:
1.  Ambil sampel sebanyak 8 instruksi. 8 Instruksi ini dicampur dengan rasio 6 instruksi buatan manusia dan 2 instruksi yang dibuat oleh LLM (kalau sudah ada).
2.  8 Instruksi ini diberikan ke LLM yang diminta untuk instruksi tambahan.
3.  Setiap hasil instruksi ini lalu difilter berdasarkan heuristic sederhana, seperti panjang instruksi, keyword tertentu, dan juga heuristic lainnya seperti memfilter instruksi yang diawali dengan karakter non-ascii dan lain sebagainya.
4.  Setiap instruksi ini kemudian diklasifikasi menggunakan sebuah LLM juga secara few shot berdasarkan sifat atau task yang diminta dari instruksi tersebut.
5.  Setiap instruksi kemudian digunakan untuk membuat instance dari instruksi tersebut, yaitu pasangan input output yang sesuai dengan instruksi yang diminta. Pendekatan yang berbeda dilakukan untuk klasifikasi instruksi yang berbeda.
6.  Setiap pasangan instruksi dan instance ini kemudian difilter lagi menggunakan ROUGE-L untuk menghitung kesamaan instruksi hasil generate LLM dengan instruksi yang berada dalam base data. Hanya instruksi yang cukup berbeda yang bisa masuk ke base data.
7.  Tahap ini diulang lagi dengan base data yang sudah ditambahkan dengan hasil generate LLM.

Yang menarik dari pendekatan ini adalah tidak ada konfigurasi maupun instruksi tambahan yang secara spesifik menarget keragaman data, tetapi hasil evaluasi menunjukkan bahwa hasil instruction tuning yang dilakukan menggunakan data ini menghasilkan model yang memiliki kemampuan cukup baik yang bersaing dengan mode lain yang dilatih menggunakan data yang berasal dari pengumpulan data manual.

![Diversity Visualization](https://lms.sdmdigital.id/pluginfile.php/635412/mod_page/content/25/image51.jpg?time=1768377680275)
**Gambar 19. Keragaman data berdasarkan 20 root verb atau kata benda yang diasosiasikan dengan setiap root verb pada instruksi hasil generate LLM.**

![Evaluation Results](https://lms.sdmdigital.id/pluginfile.php/635412/mod_page/content/25/image57.jpg?time=1768377665274)
**Gambar 20. Perbandingan hasil evaluasi jawaban GPT-3 yang di-tuning menggunakan dataset yang dibuat LLM dengan model lainnya**

**Tabel 3. Hasil evaluasi pada dataset SuperNI**

| Model | ROUGE-L pada benchmark SuperNI |
| :--- | :---: |
| GPT-3 | 6.8 |
| InstructGPT001 | 40.8 |
| GPT-3 + Tuning pada dataset self-instruct | 39.9 |

**Studi kasus 2: Dataset sintetis yang digunakan untuk melatih [Alpaca](https://crfm.stanford.edu/2023/03/13/alpaca.html)**

Untuk melatih model yang mereka namakan Alpaca, tim yang mengembangkan model ini memilih pendekatan data sintetis mirip seperti self-instruct dengan beberapa pengembangan lanjutan:
1.  Menggunakan LLM yang berbeda dengan self-instruct, yaitu text-davinci-003 dibandingkan model davinci yang digunakan self-instruct.
2.  Menggunakan prompt yang lebih jelas dan spesifik untuk generate instruksi.
3.  Menghilangkan langkah klasifikasi jenis instruksi untuk menyederhanakan proses.

Proses ini menghasilkan data yang cukup beragam, dan sama seperti self-instruct, dilakukan juga perbandingan antara model yang dilatih menggunakan dataset Alpaca dan text-davinci-003 pada dataset yang sama yang digunakan self-instruct untuk melakukan evaluasi, dan ditemukan bahwa model yang dilatih menggunakan Alpaca memiliki performa yang mendekati text-davinci-003, walaupun tidak diberikan dengan jelas figur performa tersebut dalam laporan mereka.

**Pendekatan lanjutan**

Sebenarnya inti dari data sintetis adalah bagaimana caranya data yang terbatas (seed data) bisa dikembangkan / divariasikan secara otomatis. Yang menjadi masalah adalah cara memastikan variasi-variasi ini cukup diverse untuk menciptakan data yang lebih efektif.

Selain metode-metode sebelumnya, banyak teknik yang lebih kompleks yang bisa diaplikasikan untuk meningkatkan keragaman dan kualitas data sintesis:
*   IBM LAB menggunakan [sistem berbasis taksonomi](https://research.ibm.com/blog/LLM-generated-data) yang membagi task menjadi kategori knowledge, foundational skill (matematika, reasoning) dan composition yang menggabungkan knowledge dan foundational skill.
*   Meningkatkan kualitas data sintesis secara iteratif menggunakan LLM, misalnya seperti [Evol-Instruct](https://arxiv.org/abs/2304.12244) yang menentukan pengembangan sebuah data, misalnya ditambahkan reasoning, mengubah bentuk pertanyaan, dsb.
*   Melakukan refinement, melalui mekanisme evaluasi otomatis untuk data sintetis yang kemudian bisa memberikan feedback untuk meningkatkan kualitas datanya.

---

## 1.2.6 Melakukan crawling

Metode ini umumnya bisa digunakan untuk mendapatkan data baik untuk pretraining maupun post-training. Metode ini mencari sendiri data spesifik pada internet yang sesuai dengan kebutuhan melalui crawling dan scraping. Crawling menjelajahi struktur halaman sebuah website untuk mendapatkan halaman-halamannya, lalu scraping dilakukan untuk mengambil konten yang diperlukan.

Ada beberapa langkah yang bisa kita terapkan untuk melakukan crawling:
1.  Menentukan sumber crawling.
2.  Mengimplementasi crawler dan logika untuk berpindah halaman.
3.  Menyimpan setiap halaman hasil crawling atau menerapkan logika parsing jika sudah diketahui bagian mana yang ingin diambil.

Crawling bisa dilakukan dengan berbagai cara, tapi di Python, salah satu yang paling mudah digunakan adalah **Scrapy**.

**Instalasi Scrapy:**

```bash
# uv
uv init # inisiasi environment uv
uv add scrapy

# pip
python -m venv env 
# aktivasi environment
env/scripts/activate # linux / macos
env/Scripts/activate.bat # windows
pip install scrapy
```

**Setup Project:**

```bash
# pipenv
python -m scrapy startproject provinsi_wikipedia_scraper
# uv
uv run python -m scrapy startproject provinsi_wikipedia_scraper
# masuk ke direktori project
cd provinsi_wikipedia_scraper
```

**Struktur Direktori:**
```text
provinsi_wikipedia_scraper
├─ provinsi_wikipedia_scraper
│  ├─ items.py
│  ├─ middlewares.py
│  ├─ pipelines.py
│  ├─ settings.py
│  ├─ spiders
│  │  └─ __init__.py
│  └─ __init__.py
└─ scrapy.cfg
```

Sebagai contoh, kita ingin melakukan crawling data provinsi dari Wikipedia (URL: `https://id.wikipedia.org/wiki/Provinsi_di_Indonesia`).

![Target Table](https://lms.sdmdigital.id/pluginfile.php/635412/mod_page/content/25/image37.png)
**Gambar 21. Tabel yang menjadi target**

**Implementasi Spider Sederhana (Print URL):**

```python
import scrapy

class UrlPrintScraper(scrapy.Spider):
    name = "print_url"
    allowed_domains = ["wikipedia.org"]
    start_urls = ["https://id.wikipedia.org/wiki/Provinsi_di_Indonesia"]

    async def parse(self, response):
        province_urls = response.css(
            "table > tbody#mwRA >tr > td > b > a::attr(href)"
        ).getall()
        for url in province_urls:
            print(url)
```

Jalankan dengan: `scrapy crawl print_url`

**Implementasi Spider untuk Mengambil Konten HTML:**

```python
import scrapy

class IndoProvScraper(scrapy.Spider):
    name = "provindo"
    allowed_domains = ["wikipedia.org"]
    start_urls = ["https://id.wikipedia.org/wiki/Provinsi_di_Indonesia"]

    async def parse(self, response):
        province_urls = response.css(
            "table > tbody#mwRA >tr > td > b > a::attr(href)"
        ).getall()
        for url in province_urls:
            yield response.follow(url, callback=self.get_province_html)

    async def get_province_html(self, response):
        yield {"url": response.url, "html": response.text}
```

Jalankan dan simpan hasil: `scrapy crawl provindo -o result.jsonl`

**Implementasi Pipeline untuk Menulis File:**

Ubah `pipelines.py`:
```python
import json
from itemadapter import ItemAdapter

class JsonlWriterPipeline:
    def open_spider(self, spider):
        self.file = open("items.jsonl", "w")

    def close_spider(self, spider):
        self.file.close()

    def process_item(self, item, spider):
        line = json.dumps(ItemAdapter(item).asdict()) + "\n"
        self.file.write(line)
        return item
```

Aktifkan di `settings.py`:
```python
ITEM_PIPELINES = {
   "provinsi_wikipedia_scraper.pipelines.JsonlWriterPipeline": 100,
}
```

![Inspect Element](https://lms.sdmdigital.id/pluginfile.php/635412/mod_page/content/25/image30.png)
**Gambar 22. Menggunakan inspect element untuk mencari komponen tertentu yang berisi konten**

**Best practices crawling:**
1.  Pastikan website memperbolehkan pengambilan data.
2.  Patuhi `robots.txt`.
3.  Atur jeda pengaksesan (`DOWNLOAD_DELAY`).

Pengaturan di `settings.py`:
```python
# Obey robots.txt rules
ROBOTSTXT_OBEY = True
    
# Throttling settings
CONCURRENT_REQUESTS_PER_DOMAIN = 1
DOWNLOAD_DELAY = 1
```

![Settings.py](https://lms.sdmdigital.id/pluginfile.php/635412/mod_page/content/25/image23.png)
**Gambar 23. Pengaturan parameter di atas pada settings.py**

---

## 1.2.7 Mengumpulkan data yang dihasilkan secara natural dari interaksi manusia

Metode ini cocok untuk post-training (fine tuning atau preference alignment). Bentuk data ini meliputi:
*   Interaksi manusia di situs seperti Stack Overflow atau Quora.
*   Sejarah percakapan dari WhatsApp atau Slack.
*   Transkrip diskusi atau rapat.

Keuntungan:
*   Mencerminkan skenario riil di lapangan.
*   Memungkinkan analisis kasus penggunaan untuk evaluasi.
*   Struktur data sudah jelas (tanya-jawab, asisten-pengguna).

![ChatGPT Feedback](https://lms.sdmdigital.id/pluginfile.php/635412/mod_page/content/25/image18.png)
**Gambar 24. Contoh interface chatgpt secara langsung meminta preferensi dari user**

**Penting:** Selalu perhatikan masalah privasi dan pastikan pengumpulan data tidak melanggar privacy policy secara legal maupun etis.


## 1.2.8 Membuat dataset secara manual oleh domain expert

Umumnya tahap ini dilakukan untuk dataset post training. Beberapa jenis sumber data, misalnya yang tertulis juga bisa di-digitasi kalau misalnya ingin mengumpulkan data untuk pretraining, tapi lebih jarang dilakukan.

Kalau memang sulit mendapatkan data dari internet maupun sumber eksternal, pilihan terakhir adalah dengan cara meminta domain expert membuat sebuah dataset secara manual. Misalnya seorang domain expert dari bidang hukum diminta membuat pasangan input-output pertanyaan mengenai kasus hukum serta jawaban yang seharusnya diberikan. Selain itu, tahap ini juga bisa hanya melakukan pelabelan pada data yang sebelumnya sudah dikumpulkan, termasuk validasi.

Berikut adalah beberapa aspek yang berhubungan penting untuk diperhatikan:

*   **Pemetaan domain expertise yang tepat:** Untuk task yang umum, bisa langsung crowdsourcing saja asal jelas instruksinya. Berbeda dengan task yang domainnya sangat khusus, harus menggunakan domain expert.
*   **Memiliki sistem QC yang baik:** termasuk secara berkala melakukan kurasi "gold standard" yang menjadi referensi baku untuk proses pelabelan / pengumpulan data. Sistem QC ini juga termasuk memastikan tingkat annotator agreement yang baik menggunakan metrik seperti Cohen's Kappa atau Krippendorf's Alpha, serta membangun guideline atau petunjuk melakukan pengumpulan data atau anotasi.
*   **Mendesain flow kerja yang baik dan jelas:** untuk memudahkan kinerja annotator, termasuk mendesain UI dan UX yang intuitif serta secara berkala melakukan perbaruan flow kerja yang sudah dibuat seiring ditemukannya kasus-kasus baru selama durasi pengumpulan data.
*   **Mitigasi bias:** dengan cara memastikan komposisi tim annotator terdiri dari orang dengan latar belakang yang seberagam mungkin, termasuk mengajarkan annotator agar paham mengenai bias serta mengenali sinyal-sinyal yang mengindikasikan adanya bias dalam proses yang mereka lakukan.

---

### 1.2.9 Membersihkan data

Dataset yang dikumpulkan, terutama kalau masih berbentuk data mentah dari internet perlu diproses lebih lanjut sebelum bisa digunakan untuk training. Konten utama yaitu teks yang relevan perlu diekstraksi terlebih dahulu, lalu difilter berdasarkan kriteria kualitas tertentu.

#### **Text extraction**

Data dari internet memiliki berbagai macam bentuk, utamanya dalam format file HTML yang digunakan untuk mengembangkan sebuah halaman website. Format ini mengatur struktur website dengan membaginya menjadi bagian bagian tertentu untuk memudahkan pengembang website mengatur desain dan letak setiap bagian tersebut. Untuk melatih LLM, kita lebih peduli pada konten yang terkandung di dalam setiap halaman website tersebut, sehingga perlu di-extract terlebih dahulu. Umumnya library seperti `trafilatura` dalam Python bisa digunakan untuk melakukan ekstraksi ini.

Pada praktiknya, ada kemungkinan penyedia dataset ini sudah melakukan ekstraksi dan menyediakan versi dataset yang sudah diekstraksi hanya teksnya saja, meskipun mungkin beberapa kasus tertentu memerlukan logika ekstraksi teks yang berbeda.

Sebagai contoh, kita ingin melakukan ekstraksi konten dari halaman wikipedia untuk nasi uduk:

![Wikipedia Nasi Uduk](https://lms.sdmdigital.id/pluginfile.php/635413/mod_page/content/35/image41.png)
**Gambar 25. Halaman wikipedia untuk nasi uduk**

Bentuk konten ini jika direpresentasikan dalam bentuk teks adalah sebagai berikut:

![HTML Source Nasi Uduk](https://lms.sdmdigital.id/pluginfile.php/635413/mod_page/content/35/image6.png)
**Gambar 26. Source code HTML halaman wikipedia untuk nasi uduk**

Dan itu bahkan belum masuk ke konten utama yang berisi teks, karena HTML terdiri dari berbagai jenis komponen dan tidak semuanya berfokus pada konten. Berbagai komponen HTML memiliki bermacam tujuan, seperti ada bagian yang berisi keperluan untuk SEO, ada bagian yang murni tujuannya hanya untuk membentuk struktur, dan lain sebagainya. Tujuan text extraction adalah mengambil hanya bagian relevan dari sebuah teks, dalam kasus ini konten artikel yang ingin kita gunakan.

Menggunakan `trafilatura` sendiri cukup simpel. Berikut adalah kode sederhana yang menggunakan trafilatura untuk memproses halaman di atas, asumsikan konten halaman tersebut sudah disimpan dalam file `nasi_uduk.html`.

```python
import trafilatura

with open("data/nasi_uduk.html", encoding="utf8") as f:
    nasi_uduk_html = f.read()
   
text_content = trafilatura.extract(nasi_uduk_html, output_format="txt")

print(text_content)
```

Hasil text extraction pada halaman di atas memberikan hasil seperti berikut:

```text
Nasi uduk
| Nasi uduk | |
|---|---|
| Sajian | Menu utama |
| Tempat asal | Indonesia |
| Daerah | Jakarta, Jawa |
| Suhu penyajian | Panas atau suhu kamar |
| Bahan utama | Nasi dimasak dalam santan dengan lauk pauk |
| Sunting kotak info • L • B | |
Nasi uduk adalah hidangan yang dibuat dari nasi putih yang diaron dan dikukus dengan santan, serta dibumbui dengan pala, kayu manis, jahe,
daun serai, dan merica, yang populer di hidangan Betawi.[1] Nasi uduk tergolong sebagai makanan sepinggan
dengan cita rasa gurih, yakni makanan yang disajikan dalam satu piring, dengan kandungan energi di atas 300
kalori, dan terdiri dari serealia atau umbi sebagai makanan pokok, didampingi oleh sayuran, produk hewani, dan kacang-kacangan.[2]   
Nasi uduk biasa dihidangkan dengan emping goreng, tahu goreng, telur dadar atau telur goreng yang diiris, abon kering, tempe, bawang goreng, ayam goreng,
timun, serta sambal kacang.[butuh rujukan]
Etimologi
[sunting | sunting sumber]Menurut buku "Kuliner Betawi Selaksa Rasa & Cerita" 
(2016) yang disusun oleh Akademi Kuliner Indonesia, istilah uduk secara etimologis bermakna 
"susah", yang menyiratkan bahwa dahulu makanan ini lazim dikonsumsi para kriminal betawi, 
seperti pembunuh dan pencuri yang mendominasi DKI Jakarta.      
Teori lain berpendapat bahwa istilah uduk terkait atau berkerabat dengan istilah aduk, maka nasi uduk bermakna 
"nasi yang diaduk" atau "nasi campur".[3]
Akan tetapi ada sementara pihak yang mengaitkan nasi uduk dengan Sultan Agung dari Mataram. Sultan Agung dari Mataram 
dikatakan menyebut hidangan nasi ini nasi wudu`, dari kata bahasa Arab tawadhu' yang berarti rendah hati di hadapan Tuhan.[4][5] 
Nama uduk serapan dari bahasa Jawa yaitu wuduk.[6][7][8][9][10][11] 
Nasi ini juga sering disebut sega gurih (nasi gurih) mengacu pada rasanya yang didominasi nuansa gurih.[12]
…
```

Data tersebut dapat kemudian dibersihkan lebih lanjut menggunakan RegEx. RegEx yang dibuat perlu disesuaikan dengan apa yang perlu difilter, dalam kasus di atas, kita ingin menghilangkan bagian yang tidak diinginkan seperti:
*   Sisa sisa anotasi ([1], [2] dan seterusnya)
*   Bagian teks seperti [sunting | sunting sumber], [butuh rujukan] dan lain sebagainya
*   Normalisasi karakter unicode
*   Kurung yang kosong
*   Dan bermacam artifak lainnya sisa permbersihan teks.

Contohnya kode di atas dapat diubah menjadi seperti berikut:

```python
import re
import trafilatura
import unicodedata


def clean_wikipedia_text(text):
    if not text or not isinstance(text, str):
        return ""
   
    # teks yang bisa ikut ter-extract dari struktur konten wikipedia.
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\[butuh rujukan\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[butuh klarifikasi\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[sunting\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[sunting \| sunting sumber\]', '', text, flags=re.IGNORECASE)
   
    # artifak teks markup wikipedia
    text = re.sub(r'\{\{[^}]+\}\}', '', text)
    text = re.sub(r'\[\[([^\]|]+\|)?([^\]]+)\]\]', r'\2', text)
   
    # menghilangkan artefak html dan tag
    text = re.sub(r'&nbsp;', ' ', text)
    text = re.sub(r'&[a-z]+;', '', text)
    text = re.sub(r'<[^>]+>', '', text)
   
    # menghilangkan teks dari navigasi
    text = re.sub(r'^(lihat pula|referensi|pranala luar|bacaan lebih lanjut):?\s*$', '', text, flags=re.MULTILINE | re.IGNORECASE)
   
    # normalisasi spasi dan newline
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Max 2 consecutive newlines
    text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces to single space
    text = re.sub(r' \n', '\n', text)  # Remove trailing spaces
    text = re.sub(r'\n ', '\n', text)  # Remove leading spaces
   
    # menghilangkan bullet points untuk list
    text = re.sub(r'^\s*[•·∙▪▫■□●○◆◇★☆→⇒►▸]\s*', '', text, flags=re.MULTILINE)
   
    # normalisasi unicode
    text = unicodedata.normalize('NFKC', text)
   
    # buang kurung yang kosong
    text = re.sub(r'\(\s*\)', '', text)
    text = re.sub(r'\[\s*\]', '', text)
   
    # pastikan tanda baca spasinya benar
    text = re.sub(r'\s+([.,;:!?])', r'\1', text)
    text = re.sub(r'([.,;:!?])([A-Za-z])', r'\1 \2', text)
   
    # buang tanda baca di luar tempat
    text = re.sub(r'^\s*[.,;:]\s*$', '', text, flags=re.MULTILINE)
   
    # strip untuk menghilangkan spasi di awal dan akhir teks
    text = text.strip()
   
    return text


with open("data/nasi_uduk.html", encoding="utf8") as f:
    nasi_uduk_html = f.read()
   
text_content = trafilatura.extract(nasi_uduk_html, output_format="txt")
text_content = clean_wikipedia_text(text_content)


print(text_content)
```

Logika pembersihan seperti ini bisa digabung langsung saat proses crawling, contohnya sebagai bagian dari pipeline tambahan pada contoh Scrapy di atas. Didapatkan teks berikut yang sudah lebih bersih:

```text
| Nasi uduk | |
|---|---|
| Sajian | Menu utama |
| Tempat asal | Indonesia |
| Daerah | Jakarta, Jawa |
| Suhu penyajian | Panas atau suhu kamar |
| Bahan utama | Nasi dimasak dalam santan dengan lauk pauk |
| Sunting kotak info • L • B | |
Nasi uduk adalah hidangan yang dibuat dari nasi putih yang diaron dan dikukus dengan santan, serta dibumbui 
dengan pala, kayu manis, jahe, daun serai, dan merica, yang populer di hidangan Betawi. Nasi uduk tergolong 
sebagai makanan sepinggan dengan cita rasa gurih, yakni makanan yang disajikan dalam satu piring, dengan 
kandungan energi di atas 300 kalori, dan terdiri dari serealia atau 
umbi sebagai makanan pokok, didampingi oleh sayuran, produk hewani, dan kacang-kacangan.
Nasi uduk biasa dihidangkan dengan emping goreng, tahu goreng, telur dadar atau telur goreng yang diiris, 
abon kering, tempe, bawang goreng, ayam goreng, timun, serta sambal kacang.
Etimologi
Menurut buku "Kuliner Betawi Selaksa Rasa & Cerita" (2016)
yang disusun oleh Akademi Kuliner Indonesia, istilah uduk secara etimologis bermakna "susah",
yang menyiratkan bahwa dahulu makanan ini lazim dikonsumsi para kriminal betawi, seperti pembunuh dan pencuri yang mendominasi DKI Jakarta.
Teori lain berpendapat bahwa istilah uduk terkait atau berkerabat dengan istilah aduk, maka nasi uduk bermakna
"nasi yang diaduk" atau "nasi campur".
Akan tetapi ada sementara pihak yang mengaitkan nasi uduk dengan Sultan Agung dari Mataram. Sultan Agung dari Mataram 
dikatakan menyebut hidangan nasi ini nasi wudu`, dari kata bahasa Arab tawadhu' yang berarti rendah hati di hadapan Tuhan.
Nama uduk serapan dari bahasa Jawa yaitu wuduk. Nasi ini juga sering disebut sega gurih (nasi gurih) mengacu pada rasanya
yang didominasi nuansa gurih.
…
```

Skenario yang berbeda bisa jadi memerlukan pemrosesan berbeda. Setiap website yang berbeda memerlukan pemrosesan spesifik, sehingga perlu pola-pola regexnya sendiri dan logika pembersihan teks yang berbeda spesifik untuk format yang ditemui pada website tersebut.

#### **Quality Filtering**

Setelah mendapatkan konten teks, perlu dilakukan filtering untuk memastikan hanya data berkualitas tinggi saja yang masih tersisa untuk dilanjutkan ke proses selanjutnya. Biasanya kualitas ini didefinisikan menggunakan beberapa kategori:
*   Tidak berasal dari situs dewasa maupun sumber konten negatif lainnya
*   Tidak berisi keyword atau topik diskusi yang berbahaya
*   Teksnya melebihi panjang tertentu (diasumsikan makin panjang, makin bermakna)
*   Dokumennya dalam bahasa tertentu

Salah satu pendekatan yang sering digunakan adalah membuat serangkaian filter yang menghitung statistik tertentu dari suatu dokumen. Kalau ada satu filter yang tidak lolos, maka dokumennya dianggap tidak berkualitas. Berikut adalah contoh statistik yang bisa dihitung:
*   Menghitung rata rata panjang kata dalam sebuah dokumen.
*   Menghitung persentase dari baris yang diakhiri tanda baca.
*   Mencari statistik kemunculan kata kunci tertentu

Metode ini sering dikenal sebagai **heuristics**. Heuristics mendefinisikan *rule of thumb* dan digunakan sebagai solusi yang lebih murah dibandingkan melakukan inspeksi manual satu per satu dokumen baik oleh manusia maupun model tertentu.

**Colossal Clean Crawled Corpus (C4)**
C4 merupakan dataset hasil pemrosesan dari Common Crawl yang menggunakan beberapa heuristic untuk filtering kualitas data. Heuristic tersebut antara lain:
*   **Terminal Punctuation Filter:** Membuat asumsi bahwa sebuah kalimat yang valid diakhiri tanda baca.
*   **Sentence & Word Count Filter:** hanya mengambil data dari halaman yang memiliki minimal panjang kalimat dan baris teks tertentu.
*   **Filter berdasarkan kata kata "terlarang":** menggunakan database kata-kata negatif untuk memfilter halaman.
*   **Filtering spesifik web atau situs tertentu:** Filter baris kata atau halaman dengan kata kunci teknis yang tidak relevan (javascript, lorem ipsum, dll).
*   **Filtering berdasarkan bahasa:** menggunakan `langdetect` untuk mendeteksi dokumen bahasa Inggris.

**Fineweb**
Fineweb merupakan dataset dikembangkan oleh Huggingface menggunakan dataset Common Crawl sebagai basisnya. Mereka menemukan 3 statistik yang lebih powerful untuk melakukan filtering kualitas data:
*   **Fraction of lines ending with punctuation:** Filter dokumen berdasarkan persentase baris kalimat yang diakhiri tanda baca.
*   **Duplicated line ratio:** mengukur persentase baris yang memiliki duplikat yang sama persis pada suatu dokumen.
*   **Short line fraction:** persentase jumlah baris yang sangat pendek dalam sebuah dokumen.

FineWeb juga menggunakan **Gopher Quality Filter** yang menggunakan statistik seperti penggunaan *stop words*, jumlah kata, rata-rata panjang kata, rasio simbol tertentu, rasio *bullet lines*, dan rasio kata alfabet.

#### **Masalah dari penggunaan heuristic dan solusinya**
Selain penetapan heuristic, seiring berkembangnya zaman, semakin banyak metode berbasis model yang dieksplor untuk melakukan filtering data. **Fineweb2-HQ** menggunakan classifier berbasis transformer dan FastText untuk memilih sampel dengan kualitas yang tinggi.

#### **Deduplication**

Data duplikat selalu menjadi masalah dalam machine learning. Deduplikasi umumnya dilakukan pada tingkat dokumen dengan cara menghitung derajat kemiripan menggunakan metode seperti hashing.

![Deduplication Illustration](https://lms.sdmdigital.id/pluginfile.php/635413/mod_page/content/35/image2.png)
**Gambar 27. Ilustrasi cara deduplikasi dengan hash**

FineWeb menggunakan metode bernama **MinHash**. Metode ini memecah dokumen menjadi potongan kecil yang disebut **shingles**. Setiap shingle kemudian dihitung nilai hash-nya menggunakan beberapa hashing function.

Alur MinHash:
1.  Bagi setiap dokumen menjadi shingles (misal: word-level 5-gram dengan overlap).
2.  Siapkan sejumlah hash function (FineWeb menggunakan 112).
3.  Untuk setiap hash function, hitung nilai hash terkecil dari setiap shingle dokumen untuk mendapatkan **signature**.

![MinHash Signature](https://lms.sdmdigital.id/pluginfile.php/635413/mod_page/content/35/image45.jpg)
**Gambar 28. Ilustrasi mini bagaimana sebuah signature MinHash bisa dihitung**

FineWeb menggunakan threshold kemiripan sebesar 75%. 112 nilai hash pada signature dibagi menjadi 14 kelompok yang disebut **bucket**. Sebuah dokumen dianggap duplikat jika salah satu bucket-nya sama persis.

![Document Comparison](https://lms.sdmdigital.id/pluginfile.php/635413/mod_page/content/35/image11.png)
**Gambar 29. Ilustrasi membandingkan dua dokumen untuk mencari kemiripan**

#### **PII Masking**
Sebagai bagian dari kebijakan etis, informasi pribadi (PII) seperti email dan nomor telepon harus disamarkan menggunakan regular expression dan diganti dengan data dummy atau placeholder seperti `[PHONE_NUMBER]` dan `[EMAIL_ADDRESS]`.

#### **Datatrove, library untuk implementasi pipeline data cleaning**

Huggingface mengembangkan `datatrove` untuk memudahkan proses data cleaning. Komponen utamanya adalah:

*   **Pipeline:** Abstraksi proses end-to-end dari pemetaan hingga tokenization.
*   **Block:** Komponen penyusun pipeline (baca tulis, ekstraksi teks, filtering heuristic, deduplikasi, dll).
*   **Executor:** Menentukan di mana operasi dijalankan (lokal atau kluster cloud seperti Slurm).

Contoh gambaran pipeline Datatrove untuk FineWeb:

```python
"""
This file contains the code used to process and create the
FineWeb dataset (https://huggingface.co/datasets/HuggingFaceFW/fineweb)
"""

from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.pipeline.dedup import MinhashDedupCluster, MinhashDedupFilter, MinhashDedupSignature
from datatrove.pipeline.dedup.minhash import MinhashConfig, MinhashDedupBuckets
from datatrove.pipeline.extractors import Trafilatura
from datatrove.pipeline.filters import (
    C4QualityFilter,
    FineWebQualityFilter,
    GopherQualityFilter,
    GopherRepetitionFilter,
    LanguageFilter,
    URLFilter,
)
from datatrove.pipeline.formatters import PIIFormatter
from datatrove.pipeline.readers import JsonlReader, WarcReader
from datatrove.pipeline.tokens import TokensCounter
from datatrove.pipeline.writers.jsonl import JsonlWriter
from datatrove.utils.hashing import HashConfig

DUMP_TO_PROCESS = "CC-MAIN-2023-50" 
MAIN_OUTPUT_PATH = "s3://some_s3_bucket"
FILTERING_OUTPUT_PATH = f"{MAIN_OUTPUT_PATH}/base_processing"

main_processing_executor = SlurmPipelineExecutor(
    job_name=f"cc_{DUMP_TO_PROCESS}",
    pipeline=[
        WarcReader(f"s3://commoncrawl/crawl-data/{DUMP_TO_PROCESS}/segments/"),
        URLFilter(exclusion_writer=JsonlWriter(f"{FILTERING_OUTPUT_PATH}/removed/1_url/{DUMP_TO_PROCESS}")),
        Trafilatura(favour_precision=True),
        LanguageFilter(exclusion_writer=JsonlWriter(f"{FILTERING_OUTPUT_PATH}/2_non_english/")),
        GopherQualityFilter(exclusion_writer=JsonlWriter(f"{FILTERING_OUTPUT_PATH}/removed/4_gopher_qual/{DUMP_TO_PROCESS}")),
        FineWebQualityFilter(exclusion_writer=JsonlWriter(f"{FILTERING_OUTPUT_PATH}/removed/6_fineweb_qual/{DUMP_TO_PROCESS}")),
        JsonlWriter(f"{FILTERING_OUTPUT_PATH}/output/{DUMP_TO_PROCESS}"),
    ],
    # ... konfigurasi executor lainnya
)
```

Keseluruhan tahap ini menghasilkan dataset FineWeb yang berkualitas tinggi untuk pretraining LLM.

### 1.2.10 Kurasi Dataset

Banyak kemampuan yang kita mungkin inginkan dari LLM yang ingin dikembangkan. Kita ingin LLM memiliki kemampuan memahami bahasa baik itu formal maupun bahasa sehari-hari. LLM juga diinginkan mampu memahami teks dan percakapan dalam berbagai bahasa, bermacam konteks kultural, hingga menguasai bermacam topik seperti matematika dan pemrograman. Kemampuan-kemampuan ini secara langsung berhubungan dengan data diberikan ke LLM untuk dipelajari. Kemajuan arsitektur seperti apapun tidak akan bisa menggantikan data yang tepat. Arsitektur secanggih apapun tidak akan bisa menghasilkan LLM yang memiliki kemampuan coding jika datasetnya hanya berisi cerita pendek dan puisi.

Dalam melakukan kurasi dataset, harus terlebih dahulu dipahami dan ditentukan beberapa hal berikut:
*   Kemampuan spesifik domain apa saja yang diinginkan dari LLM yang dikembangkan
*   Apakah memungkinkan menggunakan data dari internet saja atau perlu pengumpulan data sendiri untuk setiap domain ini?

Pada praktiknya, kurasi dataset dilakukan dengan menggabungkan beberapa dataset spesifik domain tertentu, baik dataset yang sudah ada maupun dataset yang dikumpulkan sendiri. Walaupun terkesan simpel, kurasi dataset melibatkan lebih dari hanya mencampurkan berbagai macam data dengan domain umum maupun spesifik domain tertentu. Perbedaan domain yang signifikan antara konsep bahasa umum dan konsep-konsep spesifik domain tertentu bisa bertabrakan, terutama pada fase-fase awal melakukan training.

Untuk sebagian besar LLM modern, pengembangnya melakukan [proses training dalam beberapa tahap](https://arxiv.org/abs/2502.02737). Dalam setiap tahap, digunakan komposisi dataset dengan rasio yang berbeda untuk setiap komponennya. Komposisi ini disebut sebagai **mixture**.

![Gambar 30](https://lms.sdmdigital.id/pluginfile.php/635414/mod_page/content/25/image54.jpg?time=1768378574489)
**Gambar 30. Kata yang sama bisa memiliki pengertian yang jauh berbeda pada domain berbeda**

Bagaimana cara menentukan komposisi ini? Ada beberapa cara yang bisa dilakukan. Yang paling mudah adalah mencari skenario sejenis, misal model yang ukuran parameter atau bahkan arsitekturnya mirip dengan model yang ingin dilatih serta memiliki tujuan yang sama, hanya saja mungkin ada sedikit perbedaan. Akan tetapi metode ini tidak selalu bisa diandalkan, apalagi jika tujuan pengembangan model ini sangat spesifik.

Metode lain yang digunakan adalah dengan secara langsung menggunakan sebagian sampel dari dataset yang dikurasi ini, misal hanya mengambil 50-100B token dari dataset berukuran 11T token. Sampel ini kemudian digunakan untuk melakukan eksperimen kecil dengan tujuan menguji performa model jika dilatih menggunakan data tersebut tanpa perlu melakukan training secara penuh. Dengan asumsi sampel yang diambil representatif terhadap distribusi sebenarnya dari mixture yang diujikan, jika ditunjukkan performa yang cukup baik dibandingkan dengan konfigurasi mixture lain, ini bisa menjadi sinyal untuk mencari mixture yang optimal.

Selain dilakukan dengan melatih suatu model dari awal, eksperimen ini juga bisa dilakukan menggunakan checkpoint tertentu, misal setelah model sudah dilatih dengan 70% dari total token latih yang direncanakan. Walaupun membutuhkan resource ekstra, melakukan eksperimen ini dapat secara langsung menguji data yang akan digunakan pada model yang kita ingin latih dan memberikan feedback yang lebih berarti dan relevan.

![Gambar 31](https://lms.sdmdigital.id/pluginfile.php/635414/mod_page/content/25/image3.jpg?time=1768378533723)
**Gambar 31. Ilustrasi proses eksperimen menentukan mixture data dengan membandingkan performa hasil eksperimen**

Selain itu, masih ada satu pertanyaan lain. Bagaimana caranya menentukan kapan suatu tahap training berakhir untuk dilanjutkan ke tahap selanjutnya dengan mixture yang berbeda? Cara yang paling simpel adalah menentukan secara langsung melalui jumlah step atau mungkin jumlah token yang sudah "dilihat" atau digunakan model untuk training. Misal total token dari data yang kita miliki ada sebanyak 1.4T, kita bisa memutuskan saja misalnya mendedikasikan 900B token pertama untuk mixture tertentu, lalu dilanjutkan mixture lainnya untuk 250B token berikutnya, kemudian mixture yang berbeda lagi untuk sisanya.

![Gambar 32](https://lms.sdmdigital.id/pluginfile.php/635414/mod_page/content/25/image33.jpg?time=1768378507923)
**Gambar 32. Mixture data berubah-ubah setiap proses training (step tertentu atau saat performa berhenti meningkat) untuk domain tertentu sehingga perlu diubah rasio datanya.**

Cara lain untuk menentukan kapan perlu mengubah rasio adalah dengan menggunakan sinyal melalui pengukuran performa pada metrik tertentu secara berkala. Misal kita mengukur metrik kemampuan bahasa umum diikuti dengan metrik khusus matematika. Jika kita mengamati kalau metrik kemampuan matematika sudah sulit meningkat, mungkin sudah waktunya menggunakan mixture yang lebih banyak komposisi konten matematikanya.

---

### 1.2.11 Melatih Tokenizer

Tokenizer diperlukan untuk memecah sebuah teks menjadi bagian-bagian individual yang disebut token, yang kemudian dipetakan menjadi representasi numerik berbentuk embedding yang bisa diproses oleh LLM. Pada praktiknya, apabila data kita memilki komposisi yang sama dengan suatu model yang sudah ada, atau domain yang ditarget tidak terlalu spesifik / niche, tidak ada yang salah dengan menggunakan tokenizer yang sudah dilatih untuk model lain.

Ada beberapa pertimbangan yang penting untuk dipikirkan sebelum memutuskan apakah perlu atau tidak melatih tokenizer sendiri:
*   **Bahasa:** Kita ingin mengembangkan LLM dalam bahasa apa? Tokenizer yang dilatih pada bahasa tertentu tidak akan efisien ketika diminta melakukan tokenization untuk bahasa lainnya.
*   **Domain:** Domain apa saja yang kita target? Di luar bahasa, beberapa domain misalnya matematika dan kode memerlukan representasi khusus agar bisa diproses secara efektif.
*   **Skenario Dataset:** Apakah kita sudah mengetahui skenario apa saja yang terkandung dalam dataset kita? Ini melibatkan memahami komposisi bahasa, domain, serta komposisi lainnya.

Tokenizer sebaiknya dilatih menggunakan data dengan distribusi yang sama dengan data yang nantinya akan digunakan untuk training maupun inference.

#### **Vocabulary size**
Vocabulary berisikan semua token yang akan dikenali oleh model yang mau dilatih. Walaupun ukuran vocabulary yang besar memungkinkan representasi teks yang lebih efisien karena bisa menyimpan lebih banyak kombinasi dalam teks, ukuran vocabulary secara langsung juga mempengaruhi ukuran embedding yang harus disimpan dan ukuran output dari LLM.
Umumnya, ditemukan bahwa sekitar 50.000 cukup untuk bahasa inggris, tapi model multilingual memerlukan ukuran vocabulary lebih dari 100.000 untuk merepresentasikan berbagai macam aksara serta bahasa secara efektif. Model modern *state of the art* umumnya memiliki ukuran vocabulary lebih dari 128.000.

#### **Algoritma Tokenization**
Umumnya LLM modern menggunakan algoritma **Byte-Pair Encoding** atau **BPE**. Ada beberapa pendekatan lain seperti WordPiece atau SentencePiece, tapi BPE masih mendominasi algoritma yang digunakan untuk tokenization.

#### **Mengukur "performa" tokenizer**
*   **Fertility:** Mengukur rata-rata dari berapa banyak token yang diperlukan untuk merepresentasikan sebuah kata. Salah satu pendekatan adalah word-to-token ratio, character-to-token ratio, atau byte-to-token ratio.
*   **Proportion of continued words (PCW):** Mengukur persentase kata yang dipecah menjadi banyak bagian. Semakin kecil, artinya representasi yang dihasilkan semakin efisien.

#### **Menangani OOV (out of vocabulary words) dan Special tokens**
Apa yang terjadi ketika LLM menerima kata kata yang tidak pernah dia lihat sebelumnya? Umumnya tokenization yang dilakukan oleh LLM dilakukan menggunakan subword tokenization, artinya setiap token bisa terdiri dari komponen-komponen penyusun sebuah kata.
Untuk bagian kata yang tidak ditemukan dalam tokenizer, biasanya diberikan sebuah token khusus, misalnya token `<UNKNOWN>`.

Untuk LLM modern, ada beberapa token khusus yang penting:

**Tabel 4. Berbagai macam special token pada LLM Modern**

| Nama | Bentuk umum token | Kegunaan |
| :--- | :--- | :--- |
| Beginning of sentence | `<BOS>` | Menandai awal dari sebuah kalimat / dokumen |
| End of sentence | `<EOS>` | Menandai akhir dari sebuah kalimat / dokumen. Biasanya menjadi trigger untuk attention masking. |
| End of turn | `<EOT>` | Menandai akhir percakapan, biasa digunakan untuk instruction turning |
| Token penanda peran | `<USER>`, `<SYSTEM>`, `<ASSISTANT>` | Menandai peran dalam percakapan |
| Token untuk padding | `<PAD>` | Untuk mengisi ruang kosong agar panjang semua sequence dalam batch sama. |
| Reserved token | - | Dibiarkan kosong/ disiapkan untuk future proofing. |
| Token untuk menandai tool calling | `<TOOL></TOOL>`, dsb | Digunakan pada model tool calling |
| Token penanda reasoning | `<THINK></THINK>` | Digunakan pada model reasoning |

Contoh beberapa baris pertama konfigurasi untuk model **DeepSeek R1**:

```json
{
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [
    {
      "id": 128000,
      "content": "<｜begin▁of▁sentence｜>",
      "single_word": false,
      "lstrip": false,
      "rstrip": false,
      "normalized": false,
      "special": true
    },
    {
      "id": 128001,
      "content": "<｜end▁of▁sentence｜>",
      "single_word": false,
      "lstrip": false,
      "rstrip": false,
      "normalized": false,
      "special": true
    },
    ...
  ]
}
```

Berikut adalah kode contoh bagaimana Anda bisa melatih tokenizer sendiri menggunakan library `transformers`:

```python
from datasets import load_dataset
from transformers import AutoTokenizer
import os

# kita gunakan dataset dari huggingface saja, yang penting bahasa indonesia
dataset = load_dataset(
    "id_newspapers_2018", split="train[:10000]"
)  # batasin aja 10k sampel buat contoh

# kita pakai tokenizer model lain, yaitu qwen3
# tujuannya agar special tokens dan pengaturan lain dari tokenizer asalnya
# yang ingin kita gunakan tidak perlu dibuat ulang
base_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")

# iterator untuk mendapatkan 1 batch teks untuk melatih tokenizer
def batch_iterator(batch_size=1000):
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i : i + batch_size]
        # ini perlu diatur kalau struktur di datasetnya berbeda
        yield batch["text"] if "text" in batch else batch["content"]


# Training tokenizer
new_tokenizer = base_tokenizer.train_new_from_iterator(
    text_iterator=batch_iterator(),
    vocab_size=32000,  # diatur saja sesuai vocabulary size yang diinginkan
    min_frequency=2,  # berapa kali 2 byte harus muncul berdampingan untuk dianggap pair
)


# simpan tokenizernya
output_dir = "./indonesian_tokenizer"
os.makedirs(output_dir, exist_ok=True)
new_tokenizer.save_pretrained(output_dir)
print(f"Tokenizer saved to {output_dir}")

# Contoh pemakaian
new_tokenizer = AutoTokenizer.from_pretrained("./indonesian_tokenizer")

# tes tokenizernya
test_text = "Selamat pagi! Bagaimana kabar Anda hari ini?"
tokens = new_tokenizer.encode(test_text)
decoded = new_tokenizer.decode(tokens)

print("\n=== Tes tokenizer ===")
print(f"Original: {test_text}")
print(f"Tokens: {tokens}")
print(f"Decoded: {decoded}")
print(f"Vocab size: {new_tokenizer.vocab_size}")
```

**Output contoh:**
```text
Tokenizer tersimpan ke folder ./indonesian_tokenizer

=== Tes tokenizer ===
Original: Selamat pagi! Bagaimana kabar Anda hari ini?
Tokens: [15676, 2060, 14, 6770, 3453, 1329, 579, 370, 44]
Decoded: Selamat pagi! Bagaimana kabar Anda hari ini?
Vocab size: 32000
```

---

### **Studi Kasus**

Salah satu kasus spesifik yang penting untuk mempertimbangkan melatih tokenizer sendiri adalah jika domain atau bahasa yang ditargetkan tidak memiliki representasi yang cukup di berbagai tokenizer yang tersedia. Dalam penelitian "Cendol: Open Instruction-tuned Generative Large Language Models for Indonesian Languages" yang bertujuan mengembangkan model yang mampu berinteraksi menggunakan bahasa indonesia dan bahasa daerah, para penulisnya melatih tokenizer sendiri untuk bahasa indonesia dan bahasa-bahasa daerah karena tokenizer dari model yang mereka gunakan, mT5 dan LLaMA2, tidak memiliki representasi yang cukup untuk bahasa-bahasa ini dalam proses latihnya.

Mereka melakukan vocabulary adaptation dan mengamati peningkatan efisiensi tokenization sebesar sekitar 21%, yang berujung kepada peningkatan efisiensi training sebesar 11%.

---

### **Mini Quiz**
*(Iframe Placeholder for Mini Quiz)*

---

> **Refleksi Singkat:**
> 
> Anda sudah mempelajari kalau masalah utama dari melatih model machine learning apapun, termasuk LLM adalah data. Sehingga muncul pendekatan data sintetis menggunakan LLM. Tapi bisa dilihat kalau data yang dihasilkan adalah untuk melakukan *post-training*.
> 
> Apa kira kira yang menghalangi pendekatan serupa dilakukan untuk pretraining? Pendekatan apa yang bisa dilakukan untuk menyelesaikan masalah tersebut?
> 
> Baru baru ini, PlelAs, sebuah perusahaan asal Perancis meneliti bagaimana caranya **menghasilkan data secara sintesis untuk melakukan *pretraining***. Pendekatan ini bahkan terbukti cukup efektif untuk melatih kemampuan reasoning. Coba baca [blog](https://pleias.fr/blog/blogsynth-the-new-data-frontier) yang mereka rilis sebagai eksplorasi tambahan.

