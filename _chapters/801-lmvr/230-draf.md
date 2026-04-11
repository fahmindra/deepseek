---
slug: lmvr-23
title: lmvr-23
---
# Bab 23: Menguji Kualitas Model Bahasa Besar

> **"Kepercayaan yang kita berikan pada sistem AI bukan hanya berasal dari kemampuannya, tetapi juga dari kemampuan kita untuk menguji secara ketat dan memahami keterbatasannya." — Fei-Fei Li**

Bab 23 dari LMVR menyajikan eksplorasi menyeluruh tentang pengujian kualitas model bahasa besar (LLM). Bab ini mencakup berbagai metodologi pengujian, termasuk kerangka kerja otomatis, evaluasi manual, dan pendekatan *human-in-the-loop*, untuk memastikan bahwa LLM memenuhi standar tinggi dalam akurasi, kefasihan, ketangguhan, dan keadilan. Bab ini juga menggali aspek kritis seperti deteksi bias, penilaian keamanan, dan integrasi penjaminan kualitas berkelanjutan melalui pipeline CI/CD. Melalui implementasi praktis dan studi kasus, pembaca memperoleh alat dan strategi yang diperlukan untuk mengevaluasi dan menjaga kualitas LLM secara ketat, memastikan bahwa model tersebut andal, tepercaya, dan sehat secara etis.

## 23.1. Pengujian Kualitas dalam Model Bahasa Besar (LLM)

Menguji kualitas Model Bahasa Besar (LLM) sangat penting untuk memastikan model-model ini memberikan hasil yang berkualitas tinggi, andal, dan etis, terutama karena model-model tersebut semakin banyak digunakan di area kritis seperti layanan kesehatan, keuangan, pendidikan, dan layanan pelanggan. Kualitas respons LLM harus dievaluasi di beberapa dimensi, termasuk akurasi, kefasihan, ketangguhan, keadilan, dan relevansi kontekstual, yang masing-masing memainkan peran penting dalam kinerja model di dunia nyata. Pendekatan pengujian yang terstruktur dan multifaset—menggabungkan pengujian otomatis, penilaian manual, dan proses *human-in-the-loop*—diperlukan untuk menangani berbagai aspek ini secara efektif.

![Figure 1: Proses umum untuk mengevaluasi kualitas LLM.](https://lmvr.rantai.dev//images/L6A504iArDGBvUq8WSeO-KNRhZ4d3epLnbVzxcGYg-v1.png)
**Gambar 1:** Proses umum untuk mengevaluasi kualitas LLM.

Akurasi adalah metrik utama dalam mengevaluasi kualitas LLM, karena metrik ini menilai kemampuan model untuk memberikan jawaban yang benar dan sesuai secara kontekstual. Metrik ini sangat krusial untuk aplikasi di bidang berisiko tinggi, di mana respons yang salah dapat menyebabkan konsekuensi signifikan bagi pengguna. Dalam layanan kesehatan, misalnya, respons harus tepat secara faktual, sedangkan dalam keuangan, model harus secara akurat menginterpretasikan kueri teknis. Pengujian akurasi sering kali melibatkan tolok ukur atau kumpulan data khusus tugas yang dirancang untuk mengevaluasi pengetahuan dalam domain tertentu, dilengkapi dengan penilaian ahli untuk memastikan akurasi di bidang yang kompleks.

Kefasihan, metrik kualitas inti lainnya, mengukur kemampuan model untuk menghasilkan bahasa yang benar secara tata bahasa, mudah dibaca, dan terdengar alami. Untuk LLM yang digunakan dalam interaksi pengguna akhir, kefasihan sangat penting untuk komunikasi yang efektif dan kepercayaan pengguna. Metrik otomatis seperti perplexity, yang mengukur ketidakpastian dalam prediksi model, dapat menawarkan beberapa wawasan tentang kefasihan, tetapi evaluasi manusia sering kali diperlukan untuk menilai apakah keluarannya memenuhi standar percakapan dan terdengar alami.

Ketangguhan (*robustness*) adalah ketahanan model terhadap masukan yang bervariasi atau bersifat merugikan (*adversarial*). Untuk berkinerja baik, LLM harus menangani berbagai masukan, termasuk kueri yang ambigu atau mengandung gangguan (*noisy*), tanpa menghasilkan respons yang menyesatkan atau salah. Pengujian untuk ketangguhan sering kali melibatkan pengujian *adversarial*, di mana masukan yang menantang atau kasus tepi (*edge-case*) disajikan ke model, serta mengevaluasi kemampuannya untuk menggeneralisasi di luar data pelatihan. Teknik seperti *stress testing* dan *mutation testing*, di mana masukan dimodifikasi dengan sengaja, umum digunakan untuk menilai dimensi kualitas ini.

Keadilan (*fairness*) dalam LLM sangat penting untuk mencegah keluaran yang tidak pantas atau bias, karena bias dapat memperkuat stereotip berbahaya, salah merepresentasikan kelompok, atau menghasilkan respons yang tidak setara di berbagai garis demografis. Pengujian keadilan melibatkan evaluasi respons model di berbagai kelompok demografis dan memeriksa potensi disparitas dalam kualitas atau konten respons. Metrik seperti paritas demografis, peluang yang disetarakan (*equalized odds*), dan teknik audit keadilan membantu memantau potensi bias. Dalam kasus di mana bias bersifat halus atau kompleks, peninjau manusia memainkan peran yang tak tergantikan dalam memastikan keluaran model adil dan tidak bias.

Relevansi dan koherensi melampaui akurasi faktual, karena keduanya mengukur kesesuaian kontekstual dan alur logis dari respons model. Hal ini sangat penting untuk interaksi multi-giliran di mana setiap respons harus dibangun secara logis berdasarkan pertukaran sebelumnya. Menilai relevansi sering kali bergantung pada evaluasi manusia, terutama untuk aplikasi seperti dukungan pelanggan atau konten pendidikan di mana kesesuaian respons dan konsistensi logis sangat penting.

Berbagai metode pengujian dan metrik membantu mengevaluasi dimensi kualitas ini. Metrik evaluasi otomatis, seperti BLEU (*Bilingual Evaluation Understudy*), mengukur tumpang tindih antara keluaran model dan respons referensi, sangat berguna dalam tugas-tugas seperti terjemahan atau parafrase. BLEU dihitung sebagai:

$$ \text{BLEU} = \exp \left( \frac{1}{N} \sum_{n=1}^{N} \log P_n \right) \cdot BP $$

di mana $P_n$ mewakili presisi untuk setiap level n-gram, dan $BP$, penalti singkat (*brevity penalty*), mencegah respons yang terlalu pendek. Meskipun BLEU menawarkan ukuran kemiripan kuantitatif, metrik ini sering kali gagal menangkap hal-hal halus seperti koherensi semantik atau kreativitas. Metrik otomatis lainnya, perplexity, mengukur ketidakpastian model dalam menghasilkan respons, dengan perplexity yang lebih rendah menunjukkan kefasihan dan stabilitas keluaran yang lebih tinggi.

Evaluasi manusia sangat penting, terutama ketika metrik otomatis tidak dapat menangkap kualitas seperti kreativitas, nuansa kontekstual, dan nada percakapan. Evaluasi *human-in-the-loop* memberikan umpan balik tentang apakah suatu respons, misalnya, terasa empatik dalam interaksi layanan pelanggan atau sesuai usia dalam konten pendidikan. Untuk aplikasi khusus domain, metrik dan tolok ukur khusus tugas lebih lanjut menilai apakah model memenuhi standar peraturan tertentu atau ekspektasi khusus domain, seperti di bidang layanan kesehatan atau hukum.

Mengevaluasi LLM sangat menantang karena sifat bahasa yang luas dan bervariasi. Tidak seperti model deterministik, LLM menghasilkan keluaran probabilistik, yang berarti satu prompt dapat menghasilkan beberapa respons valid. Selain itu, pengaruh bahasa yang ambigu, variasi budaya, dan sensitivitas konteks membuat evaluasi kualitas model secara komprehensif menjadi menantang. Untuk mengatasi tantangan ini, kombinasi penilaian kuantitatif dan kualitatif diperlukan untuk mencapai pemahaman yang seimbang tentang kemampuan model.

Menyeimbangkan pengujian yang ketat dengan siklus pengembangan yang efisien sangatlah penting, dan pengujian berkelanjutan menawarkan solusinya. Pendekatan ini memungkinkan pengujian terjadi secara iteratif di seluruh proses pengembangan, sehingga memungkinkan untuk mengidentifikasi dan menangani masalah lebih awal sambil menghindari risiko menunda pengembangan. Strategi pengujian berkelanjutan mencakup pengujian regresi, yang memastikan bahwa pembaruan model baru tidak menurunkan kinerja sebelumnya, dan pengujian A/B, yang membandingkan versi model yang berbeda dalam pengaturan langsung untuk melihat mana yang berkinerja lebih baik menurut umpan balik pengguna. Memasukkan umpan balik pengguna di dunia nyata memberikan wawasan yang berharga, terutama untuk tugas dengan variabilitas tinggi, yang memungkinkan model untuk meningkat secara bertahap.

Pendekatan pengujian LLM yang menyeluruh tidak hanya mencakup akurasi, kefasihan, ketangguhan, keadilan, dan relevansi tetapi juga mengintegrasikan pertimbangan etis. Protokol pengujian harus mencakup pengamanan privasi dan keamanan, bersama dengan transparansi seputar keterbatasan model. Memastikan penyebaran yang etis melibatkan pedoman yang jelas untuk penanganan data, persetujuan pengguna, dan komunikasi tentang potensi keterbatasan model.

### Contoh Python: Evaluasi Kualitas Model (Skor BLEU dan Perplexity)

```python
import math
from collections import Counter
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# nltk.download('punkt')

def compute_bleu_score(candidate, reference, n=2):
    """Menghitung skor BLEU sederhana berdasarkan n-gram overlap."""
    candidate_tokens = nltk.word_tokenize(candidate.lower())
    reference_tokens = nltk.word_tokenize(reference.lower())
    
    # Menggunakan smoothing untuk menghindari skor 0 pada n-gram yang tidak cocok
    smoothie = SmoothingFunction().method1
    weights = [1.0/n] * n
    return sentence_bleu([reference_tokens], candidate_tokens, weights=weights, smoothing_function=smoothie)

def compute_perplexity(text):
    """Placeholder simulasi skor Perplexity (log-probability sederhana)."""
    # Dalam praktik, perplexity dihitung dari model logits.
    # Di sini kita mensimulasikan nilai berdasarkan panjang dan variasi kata.
    tokens = text.split()
    if not tokens:
        return float('inf')
    
    unique_ratio = len(set(tokens)) / len(tokens)
    # Semakin unik dan panjang kalimat, simulasi probabilitas lebih tinggi (perplexity rendah)
    log_prob = -1.0 / (unique_ratio * len(tokens) + 1)
    return math.exp(-log_prob)

def main():
    question = "Apa penyebab utama jatuhnya Kekaisaran Romawi?"
    
    # Jawaban simulasi dari model
    initial_answer = "Kekaisaran Romawi jatuh karena masalah ekonomi dan serangan dari suku-suku barbar."
    
    # Jawaban referensi (ground truth)
    reference_answer = "Penyebab utama jatuhnya Kekaisaran Romawi meliputi masalah ekonomi, ketergantungan pada tenaga kerja budak, pengeluaran militer berlebihan, korupsi pemerintah, dan kedatangan suku Hun serta suku barbar lainnya."

    # Hitung metrik
    bleu = compute_bleu_score(initial_answer, reference_answer, n=2)
    ppl = compute_perplexity(initial_answer)

    print(f"Pertanyaan: {question}")
    print(f"Jawaban Model: {initial_answer}")
    print(f"Skor BLEU (Bi-gram): {bleu:.4f}")
    print(f"Skor Perplexity (Simulasi): {ppl:.4f}")

    # Simulasi refleksi dan perbaikan
    refined_answer = "Faktor-faktor seperti ketidakstabilan ekonomi, korupsi politik, serangan suku barbar seperti Hun, dan pengeluaran militer yang tidak terkendali berkontribusi pada keruntuhan Roma."
    
    refined_bleu = compute_bleu_score(refined_answer, reference_answer, n=2)
    refined_ppl = compute_perplexity(refined_answer)

    print(f"\nJawaban Setelah Refleksi: {refined_answer}")
    print(f"Skor BLEU (Refined): {refined_bleu:.4f}")
    print(f"Skor Perplexity (Refined): {refined_ppl:.4f}")

if __name__ == "__main__":
    main()
```

## 23.2. Teknik Pengujian Otomatis untuk LLM

Pengujian otomatis telah menjadi komponen penting dalam memastikan kualitas model bahasa besar (LLM), menyediakan pendekatan yang skalabel dan sistematis untuk mengevaluasi berbagai aspek kinerja. Pengujian otomatis untuk LLM dapat mencakup berbagai skenario, mulai dari memverifikasi kebenaran tata bahasa dasar hingga mengevaluasi kemampuan model untuk memberikan respons yang koheren dan sesuai konteks. Dengan mengintegrasikan pengujian ini ke dalam pipeline, kita dapat mencapai penilaian kualitas yang berkinerja tinggi dan dapat diulang.

Keuntungan pengujian otomatis terletak pada kemampuannya untuk mencakup spektrum luas perilaku model potensial secara cepat dan konsisten. Misalnya, pengujian dapat dikembangkan untuk memeriksa koherensi respons dengan membandingkan keluaran model terhadap pola yang diharapkan atau untuk mengukur kefasihan menggunakan metrik statistik. Perplexity ($P$) dapat dihitung secara matematis sebagai:

$$ P = 2^{-\frac{1}{N} \sum_{i=1}^N \log_2 p(x_i)} $$

di mana $p(x_i)$ adalah probabilitas kata $x_i$ dalam konteksnya. Nilai perplexity yang lebih rendah menunjukkan kefasihan yang lebih baik. Selain pemeriksaan kefasihan dasar, pengujian koherensi kontekstual mengevaluasi kemampuan model untuk menghasilkan respons yang relevan dan konsisten dengan konteks prompt.

### Contoh Python: Kerangka Kerja Pengujian Otomatis (Asynchronous)

```python
import asyncio
import time

class LLMAutomatedTester:
    def __init__(self, model_name):
        self.model_name = model_name

    async def get_model_response(self, prompt):
        # Simulasi panggilan API model asinkron
        await asyncio.sleep(0.1)
        return f"Respons untuk: {prompt}"

    async def test_fluency(self, prompt):
        response = await self.get_model_response(prompt)
        # Logika skor kefasihan sederhana
        score = 1.0 - (len(response) % 10 / 10.0) 
        return score

    async def test_response_time(self, prompt, threshold=0.5):
        start = time.perf_counter()
        await self.get_model_response(prompt)
        duration = time.perf_counter() - start
        
        is_ok = duration < threshold
        return is_ok, duration

async def main():
    tester = LLMAutomatedTester("GPT-3.5-Sim")
    prompt = "Jelaskan hukum gravitasi."

    # Menjalankan tes secara paralel
    fluency_task = tester.test_fluency(prompt)
    latency_task = tester.test_response_time(prompt)

    fluency_score, (latency_ok, duration) = await asyncio.gather(fluency_task, latency_task)

    print(f"Skor Kefasihan: {fluency_score:.2f}")
    print(f"Waktu Respons: {duration:.4f}s (Lulus: {latency_ok})")

if __name__ == "__main__":
    asyncio.run(main())
```

## 23.3. Pengujian Manual dan Pendekatan Human-in-the-Loop

Pengujian manual dan metodologi *human-in-the-loop* (HITL) menawarkan wawasan krusial yang menantang untuk dinilai hanya melalui teknik otomatis. Evaluasi manusia dapat mendeteksi masalah halus pada nada, gaya, dan kesantunan yang mungkin tidak segera dapat dikuantifikasi. Evaluasi berbasis skenario, di mana penilai menilai keluaran dalam konteks yang ditentukan sebelumnya (seperti skenario medis atau hukum), memastikan bahwa tanggapan selaras dengan standar profesional dan pedoman etika.

![Figure 2: Tantangan dalam mengintegrasikan pengujian manual dan otomatis.](https://lmvr.rantai.dev//images/L6A504iArDGBvUq8WSeO-YN6W1xYKtsQTKHwwi3lg-v1.png)
**Gambar 2:** Tantangan dalam mengintegrasikan pengujian manual dan otomatis.

Sistem *human-in-the-loop* menjembatani kesenjangan antara metrik kuantitatif dan penilaian kualitatif. Dalam proses pengujian HITL, pengujian otomatis awal menyaring keluaran berdasarkan kriteria kuantitatif, diikuti oleh perutean keluaran yang tidak memadai secara otomatis ke pengevaluasi manusia.

### Contoh Python: Sistem HITL dengan Antrean (Queue)

```python
import asyncio

class HumanInLoopSystem:
    def __init__(self):
        self.review_queue = asyncio.Queue()

    async def automated_filter(self, response):
        """Menyaring respons. Jika skor rendah, masukkan ke antrean review manual."""
        score = len(response) # Simulasi metrik kualitas sederhana
        if score < 15:
            await self.review_queue.put(response)
            return False
        return True

    async def human_reviewer_task(self):
        """Mensimulasikan pekerja manusia yang memproses antrean review."""
        while True:
            flagged_output = await self.review_queue.get()
            print(f"[REVIEW MANUAL] Meninjau respons bermasalah: '{flagged_output}'")
            # Simulasi proses peninjauan
            await asyncio.sleep(1)
            self.review_queue.task_done()

async def main():
    hitl = HumanInLoopSystem()
    
    # Menjalankan worker manusia di latar belakang
    worker = asyncio.create_task(hitl.human_reviewer_task())

    responses = ["Ini adalah jawaban yang sangat panjang dan koheren.", "Singkat.", "Sangat singkat."]
    
    for res in responses:
        passed = await hitl.automated_filter(res)
        if passed:
            print(f"[Otomatis] Respons lolos: '{res}'")

    # Tunggu antrean selesai diproses
    await asyncio.sleep(2)
    worker.cancel()

if __name__ == "__main__":
    asyncio.run(main())
```

## 23.4. Pengujian Bias dan Keadilan dalam LLM

Pengujian bias dan keadilan telah muncul sebagai komponen kritis untuk memastikan bahwa sistem LLM menghasilkan keluaran yang setara dan bertanggung jawab secara sosial. Tanpa pengujian bias yang tepat, LLM berisiko melanggengkan dan memperkuat stereotip masyarakat yang tertanam dalam data pelatihan. Metrik seperti paritas demografis dan peluang yang disetarakan digunakan untuk menilai bagaimana keluaran model berbeda antar kelompok demografis.

![Figure 3: Proses identifikasi bias dalam LLM.](https://lmvr.rantai.dev//images/L6A504iArDGBvUq8WSeO-2PJXXIfUCcK2AmWbfaCq-v1.png)
**Gambar 3:** Proses identifikasi bias dalam LLM.

Metrik deteksi bias dapat berupa penilaian langsung (frekuensi kata tertentu) atau tidak langsung (disparitas sentimen). Misalnya, analisis sentimen dapat mengukur apakah model secara sistematis mengaitkan label demografis tertentu dengan bahasa positif atau negatif.

### Contoh Python: Deteksi Bias Gender Sederhana

```python
class BiasTester:
    def __init__(self):
        self.gender_terms = {
            "male": ["insinyur", "dokter", "ilmuwan", "bos"],
            "female": ["perawat", "guru", "resepsionis", "sekretaris"]
        }

    def detect_bias(self, text):
        results = {"male_count": 0, "female_count": 0}
        words = text.lower().split()
        
        for word in words:
            if word in self.gender_terms["male"]:
                results["male_count"] += 1
            elif word in self.gender_terms["female"]:
                results["female_count"] += 1
        return results

def main():
    tester = BiasTester()
    sample_text = "Seorang insinyur laki-laki dan seorang perawat perempuan sedang bekerja."
    bias_results = tester.detect_bias(sample_text)
    
    print(f"Hasil Analisis Bias: {bias_results}")
    if bias_results["male_count"] != bias_results["female_count"]:
        print("Peringatan: Potensi ketidakseimbangan representasi peran dalam teks.")

if __name__ == "__main__":
    main()
```

## 23.5. Pengujian Ketangguhan dan Keamanan dalam LLM

Ketangguhan (*robustness*) mengacu pada kemampuan model untuk menangani masukan yang tidak terduga seperti data yang mengandung gangguan (*noisy*) atau merugikan (*adversarial*). Keamanan berfokus pada pencegahan penyalahgunaan dan perlindungan data pengguna. Pengujian *adversarial* melibatkan pembuatan masukan yang dirancang untuk membingungkan model, mengungkapkan kerentanan dalam proses pengambilan keputusannya.

![Figure 4: Tes Ketangguhan dan Keamanan.](https://lmvr.rantai.dev//images/L6A504iArDGBvUq8WSeO-uZ5RhArLIjqbgHWZN5JT-v1.png)
**Gambar 4:** Tes Ketangguhan dan Keamanan.

Secara formal, ketangguhan dapat dinyatakan sebagai $|f(x + \delta) - f(x)| < \epsilon$ untuk gangguan kecil $\delta$. Pengujian keamanan mencakup pencegahan serangan inversi model (*model inversion attacks*) dan sanitasi input untuk mencegah injeksi berbahaya.

### Contoh Python: Gangguan Input (*Input Perturbation*) dan Sanitasi

```python
import random

def perturb_text(text, probability=0.1):
    """Memperkenalkan gangguan acak ke dalam teks untuk menguji ketangguhan."""
    chars = list(text)
    for i in range(len(chars)):
        if random.random() < probability:
            chars[i] = 'X' # Simulasi noise/karakter rusak
    return "".join(chars)

def sanitize_input(text):
    """Membersihkan input dari karakter yang berpotensi membahayakan (misal: simulasi SQLi)."""
    bad_chars = [";", "--", "'", "DROP", "SELECT"]
    for char in bad_chars:
        text = text.replace(char, "")
    return text

def main():
    user_input = "SELECT * FROM users; -- Rahasia"
    sanitized = sanitize_input(user_input)
    print(f"Input Tersanitasi: {sanitized}")

    original_prompt = "Hukum pertama Newton menyatakan bahwa..."
    noisy_prompt = perturb_text(original_prompt)
    print(f"Prompt Berisik: {noisy_prompt}")

if __name__ == "__main__":
    main()
```

## 23.6. Studi Kasus dan Praktik Terbaik dalam Pengujian Kualitas LLM

Studi kasus dari industri keuangan dan layanan kesehatan menunjukkan pentingnya kerangka kerja pengujian yang ketat. Di sektor keuangan, perusahaan menggunakan validasi dua tingkat: skrip otomatis untuk metrik inti dan evaluasi HITL untuk interpretabilitas dan bias. Di bidang kesehatan, institusi menerapkan pengujian diferensial untuk membandingkan respons model terhadap prompt serupa dengan informasi demografis yang bervariasi.

Praktik terbaik meliputi penggunaan pengujian berbasis properti (*property-based testing*) dan tolok ukur kinerja berkelanjutan. Pengukuran waktu respons sangat penting untuk memastikan latensi optimal dalam aplikasi waktu nyata seperti dukungan pelanggan.

### Contoh Python: Tolok Ukur Waktu Respons (Benchmarking) Sederhana

```python
import time

def benchmark_llm_call(model_func, prompt):
    start_time = time.perf_counter()
    response = model_func(prompt)
    end_time = time.perf_counter()
    
    duration = end_time - start_time
    print(f"Prompt: '{prompt}'")
    print(f"Durasi: {duration:.6f} detik")
    return response, duration

# Simulasi fungsi model
def dummy_model(p):
    time.sleep(0.2) # Simulasi latensi
    return "OK"

if __name__ == "__main__":
    benchmark_llm_call(dummy_model, "Tes performa sistem.")
```

Masa depan pengujian kualitas LLM kemungkinan akan melibatkan kemajuan dalam pengujian interpretabilitas otomatis dan deteksi bias adaptif. Integrasi metode atribusi relevansi, yang memberikan skor penting pada komponen prompt individu, dapat meningkatkan transparansi model dan kepercayaan pengguna.

## 23.7. Kesimpulan

Bab 23 menekankan pentingnya pengujian kualitas yang komprehensif dalam pengembangan dan penyebaran model bahasa besar. Dengan menguasai teknik dan alat yang dibahas, pengembang dapat memastikan bahwa LLM mereka memberikan kinerja tinggi sambil mematuhi standar penting keadilan, keamanan, dan keandalan, membuka jalan bagi aplikasi AI yang bertanggung jawab.

### 23.7.1. Pembelajaran Lebih Lanjut dengan GenAI

- Deskripsikan berbagai dimensi kualitas dalam LLM (akurasi, kefasihan, ketangguhan, keadilan).
- Diskusikan tantangan khusus dalam menguji LLM dibandingkan dengan sistem perangkat lunak tradisional.
- Rancang proses implementasi pengujian otomatis menggunakan Python.
- Analisis efektivitas skor BLEU dan Perplexity dalam menilai kualitas teks.
- Jelaskan konsep dan implementasi pengujian *human-in-the-loop* (HITL).
- Diskusikan metodologi untuk mendeteksi dan memitigasi bias dalam output model.
- Analisis pentingnya pengujian ketangguhan terhadap serangan *adversarial*.
- Bagaimana peran pemantauan berkelanjutan (monitoring) dalam menjaga kualitas model yang sudah disebarkan?

### 23.7.2. Latihan Praktis

#### Latihan Mandiri 23.1: Merancang Kerangka Pengujian Otomatis
Tujuan: Membuat skrip Python yang secara otomatis mengirimkan sekumpulan prompt ke model dan mencatat skor BLEU serta waktu responsnya ke file CSV.

#### Latihan Mandiri 23.2: Implementasi Deteksi Bias Gender
Tujuan: Mengembangkan alat untuk menganalisis respons model terhadap deskripsi pekerjaan dan mengidentifikasi bias dalam penggunaan kata ganti atau atribut personal.

#### Latihan Mandiri 23.3: Pengembangan Proses HITL
Tujuan: Merancang alur kerja di mana respons model dengan tingkat kepercayaan (confidence) rendah secara otomatis dikirim ke antrean untuk ditinjau oleh operator manusia.

#### Latihan Mandiri 23.4: Pengujian Keamanan Input
Tujuan: Melakukan simulasi serangan injeksi teks pada LLM dan mengimplementasikan lapisan filter sanitasi di Python untuk mencegah eksekusi perintah berbahaya.

#### Latihan Mandiri 23.5: Integrasi CI/CD untuk Penjaminan Kualitas
Tujuan: Merancang pipeline pengujian yang berjalan secara otomatis setiap kali ada pembaruan pada model atau dataset pelatihan.