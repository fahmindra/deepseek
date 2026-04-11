---
slug: lmvr-20
title: lmvr-20
---
# Bab 20: Pengantar Prompt Engineering

> **"Kualitas prompt yang kita buat secara langsung memengaruhi kecerdasan dan efektivitas model yang kita bangun. Ini bukan hanya tentang data atau algoritma; ini tentang mengajukan pertanyaan yang tepat." — Andrew Ng**

> **Ringkasan Bab:**
> Bab 20 dari LMVR memberikan pengantar komprehensif tentang *prompt engineering*, menekankan peran krusial dari prompt yang dirancang dengan baik dalam memandu perilaku model bahasa besar (LLM). Bab ini mencakup dasar-dasar desain prompt, pengembangan alat *prompt engineering* menggunakan sistem yang efisien, dan penerapan teknik tingkat lanjut untuk meningkatkan kinerja model di berbagai tugas. Bab ini juga menggali pertimbangan etis dan praktis dari *prompt engineering*, memastikan bahwa model yang disebarkan adil, transparan, dan efektif. Melalui contoh praktis dan studi kasus, pembaca akan memperoleh keterampilan yang diperlukan untuk membuat, menyempurnakan, dan menyebarkan prompt yang memaksimalkan potensi LLM dalam aplikasi dunia nyata.

---

## 20.1. Dasar-dasar Prompt Engineering

*Prompt engineering* adalah keterampilan dasar untuk memandu model bahasa besar (LLM) dalam menghasilkan respons yang spesifik, akurat, dan relevan secara kontekstual. Seiring LLM menjadi lebih kuat dan terintegrasi ke dalam berbagai aplikasi, kemampuan untuk membentuk output mereka melalui prompt yang dirancang dengan cermat menjadi sangat penting bagi pengembang dan peneliti. *Prompt engineering* melibatkan perancangan input—prompt—yang mengarahkan pola respons, nada, dan konten model, secara efektif bertindak sebagai jembatan antara kemampuan dasar model dan output yang diinginkan. Bagian ini mengeksplorasi dasar-dasar *prompt engineering*, mulai dari prinsip desain prompt hingga teknik praktis untuk meningkatkan perilaku model.

![Gambar 1: Prompt Engineering dan Aplikasi GenAI/LLM.](https://lmvr.rantai.dev//images/L6A504iArDGBvUq8WSeO-SDn9THCa3ZirzhflNJ4L-v1.png)
**Gambar 1:** Prompt Engineering dan Aplikasi GenAI/LLM.

Pada intinya, *prompt engineering* berpusat pada pemahaman hubungan antara prompt input dan output yang dihasilkan model. Struktur, pilihan kata, dan panjang prompt semuanya memengaruhi cara LLM menafsirkan dan menanggapi kueri pengguna. Misalnya, prompt instruksional memberikan instruksi eksplisit untuk memandu model, sementara prompt percakapan dirancang untuk memfasilitasi interaksi timbal balik yang alami. Secara matematis, *prompt engineering* dapat dibingkai sebagai fungsi $f: P \rightarrow R$, di mana $P$ mewakili ruang prompt dan $R$ menunjukkan ruang respons. Dengan mengontrol struktur dan konten dari $P$, pengembang dapat memengaruhi kemungkinan output tertentu yang diinginkan dalam $R$, seperti relevansi, koherensi, dan konsistensi nada.

Dampak desain prompt pada respons model sangat mendalam, terutama ketika prompt disesuaikan dengan tugas-tugas tertentu. Misalnya, prompt khusus tugas untuk tugas analisis sentimen mungkin berisi kata-kata yang secara jelas menentukan nada emosional yang dicari, sementara prompt instruksional untuk pembuatan kode mungkin berisi bahasa yang menguraikan konvensi pengkodean atau ekspektasi penanganan kesalahan. Prompt yang terlalu samar mungkin menghasilkan hasil yang ambigu atau tidak relevan, sedangkan prompt yang terlalu kompleks mungkin menyebabkan kesalahan atau interpretasi yang tidak diinginkan.

Membuat prompt yang efektif membutuhkan kesadaran akan tantangan umum, seperti potensi prompt untuk memperkenalkan bias. Contohnya mungkin melibatkan penggunaan bahasa yang bermuatan budaya atau asumsi bias yang dapat miringkan output LLM, yang secara tidak sengaja memperkuat stereotip. Untuk mengatasi risiko ini, pengembang harus menggunakan teknik *prompt engineering* yang peka terhadap konteks dan inklusif dalam bahasa.

Berikut adalah implementasi Python yang mendemonstrasikan pipeline *prompt engineering* sederhana, di mana berbagai format prompt dibuat dan disiapkan untuk tokenisasi.

### Contoh Python: Pipeline Prompting Dasar

```python
from transformers import AutoTokenizer

class PromptType:
    INSTRUCTIONAL = "instructional"
    CONVERSATIONAL = "conversational"
    TASK_SPECIFIC = "task_specific"

def create_prompt(text, prompt_type):
    if prompt_type == PromptType.INSTRUCTIONAL:
        return f"Harap ikuti instruksi berikut dengan teliti: {text}"
    elif prompt_type == PromptType.CONVERSATIONAL:
        return f"Mari kita mengobrol: {text}"
    elif prompt_type == PromptType.TASK_SPECIFIC:
        return f"Tugas untuk model: {text}"
    else:
        return text

def main():
    # Memuat tokenizer dari Hugging Face
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Menentukan prompt untuk pengujian
    prompts = [
        ("Buat ringkasan dari dokumen berikut.", PromptType.INSTRUCTIONAL),
        ("Bagaimana cuaca hari ini?", PromptType.CONVERSATIONAL),
        ("Klasifikasikan pesan ini sebagai spam atau bukan.", PromptType.TASK_SPECIFIC)
    ]

    for text, p_type in prompts:
        formatted_prompt = create_prompt(text, p_type)
        tokens = tokenizer.encode(formatted_prompt, add_special_tokens=True)
        
        print(f"Tipe: {p_type}")
        print(f"Prompt: {formatted_prompt}")
        print(f"ID Token: {tokens[:10]}... (Total: {len(tokens)})\n")

if __name__ == "__main__":
    main()
```

Studi kasus menyoroti kemanjuran *prompt engineering* di berbagai aplikasi. Dalam dukungan pelanggan, perusahaan telah meningkatkan akurasi respons dengan merancang prompt yang memberikan konteks pada respons. Tren di lapangan berkembang pesat menuju spesifisitas prompt yang lebih besar dan integrasi ke dalam alur kerja ujung-ke-ujung, seperti *prompt chaining*—di mana beberapa prompt disusun berurutan untuk memperhalus respons.

---

## 20.2. Membangun Alat Prompt Engineering dengan Rust

Membangun toolkit *prompt engineering* yang efektif memerlukan pendekatan modular yang menyeimbangkan performa, konkurensi, dan adaptabilitas. Komponen toolkit sering kali mencakup templat prompt (*prompt templates*) untuk menyusun input, kerangka pengujian untuk memvalidasi efektivitas prompt, dan penganalisis output untuk mengukur akurasi serta kualitas respons model.

Dalam membangun toolkit, pengembang dapat memanfaatkan kemampuan pemrosesan teks yang efisien. Penggunaan templat prompt sebagai komponen modular memungkinkan perancangan prompt yang dapat digunakan kembali dan beradaptasi dengan konteks yang berbeda, mengurangi waktu dan sumber daya yang dihabiskan untuk merakit prompt secara manual.

Kemampuan pemrosesan paralel sangat penting dalam pengujian prompt karena sering kali perlu menguji beberapa variasi prompt pada dataset besar. Hal ini dapat diformalkan sebagai fungsi $f(P) \rightarrow M$, di mana $P$ mewakili sekumpulan prompt, dan $M$ menunjukkan matriks metrik untuk setiap prompt.

Berikut adalah kode Python yang menyediakan contoh toolkit *prompt engineering* modular.

### Contoh Python: Toolkit Prompting Modular

```python
import re
from concurrent.futures import ThreadPoolExecutor

class PromptTemplate:
    def __init__(self, template):
        self.template = template

    def apply(self, context):
        result = self.template
        for key, value in context.items():
            # Mengganti {{ key }} dengan value
            placeholder = r"\{\{\s*" + re.escape(key) + r"\s*\}\}"
            result = re.sub(placeholder, str(value), result)
        return result

def analyze_mock_response(prompt_text):
    # Simulasi analisis kualitas respons
    tokens = prompt_text.split()
    return {
        "token_count": len(tokens),
        "unique_tokens": len(set(tokens)),
        "is_complex": len(tokens) > 10
    }

def test_prompts_parallel(templates, context):
    results = []
    # Menggunakan ThreadPool untuk simulasi pemrosesan paralel
    with ThreadPoolExecutor() as executor:
        formatted_prompts = [t.apply(context) for t in templates]
        analysis_gen = executor.map(analyze_mock_response, formatted_prompts)
        results = list(analysis_gen)
    return results

def main():
    # Definisikan templat
    templates = [
        PromptTemplate("Tolong selesaikan tugas ini: {{ task }} dalam bahasa {{ language }}."),
        PromptTemplate("Lakukan tugas dalam bahasa {{ language }}: {{ task }}.")
    ]
    
    context = {"task": "Ringkas dokumen ini", "language": "Indonesia"}
    
    # Uji efektivitas
    analysis_results = test_prompts_parallel(templates, context)
    
    for i, res in enumerate(analysis_results):
        print(f"Hasil Analisis Templat {i+1}: {res}")

if __name__ == "__main__":
    main()
```

Toolkit yang dirancang dengan baik harus menyediakan API yang berinteraksi mulus dengan berbagai tugas LLM. Praktik terbaik untuk memelihara alat ini melibatkan sistem versi untuk templat prompt, yang memungkinkan pengembang melacak iterasi prompt dari waktu ke waktu.

---

## 20.3. Merancang Prompt yang Efektif untuk Berbagai Aplikasi

Merancang prompt yang efektif adalah aspek fundamental dari *prompt engineering*, terutama saat bertujuan untuk mengoptimalkan akurasi output LLM di berbagai tugas yang berbeda. Setiap aplikasi—baik pembuatan konten, tanya jawab, atau peringkasan—memerlukan prompt yang disesuaikan secara khusus.

![Gambar 2: Tantangan dalam desain prompt.](https://lmvr.rantai.dev//images/L6A504iArDGBvUq8WSeO-OIqoaksXY8BWG472QWWp-v1.png)
**Gambar 2:** Tantangan dalam desain prompt.

Secara matematis, desain prompt dapat direpresentasikan sebagai masalah optimasi di mana tujuannya adalah memaksimalkan kualitas respons model diberikan sebuah prompt $p$ dan persyaratan tugas $T$. Secara formal, kita definisikan $f(p; T) \rightarrow R$. Pengetahuan domain sangat penting dalam proses ini; misalnya, prompt untuk peringkasan dokumen hukum mungkin memerlukan istilah teknis hukum dan pemahaman tentang struktur dokumen tersebut.

### Contoh Python: Desain Prompt Berbasis Aplikasi

```python
class PromptApplication:
    CONTENT_GEN = "content_gen"
    QA = "qa"
    SUMMARY = "summary"

def create_app_prompt(app_type, data):
    if app_type == PromptApplication.CONTENT_GEN:
        return f"Tulis cerita kreatif tentang: {data}"
    elif app_type == PromptApplication.QA:
        return f"Jawab pertanyaan berikut dengan tepat: {data}"
    elif app_type == PromptApplication.SUMMARY:
        return f"Ringkas dokumen berikut secara ringkas: {data}"
    return data

def main():
    # Simulasi pengujian untuk berbagai aplikasi
    apps = [
        (PromptApplication.CONTENT_GEN, "dunia masa depan"),
        (PromptApplication.QA, "Apa ibu kota Perancis?"),
        (PromptApplication.SUMMARY, "Dokumen ini membahas tentang...")
    ]

    for app_type, data in apps:
        p = create_app_prompt(app_type, data)
        print(f"Aplikasi: {app_type}\nPrompt: {p}\n")

if __name__ == "__main__":
    main()
```

---

## 20.4. Mengevaluasi dan Menyempurnakan Prompt

Mengevaluasi dan menyempurnakan prompt memungkinkan pengembang mengoptimalkan prompt untuk akurasi dan relevansi yang lebih baik. Proses ini dapat dikonseptualisasikan sebagai fungsi optimasi $E: P \rightarrow Q$, di mana $E$ adalah metrik evaluasi dan $Q$ adalah skor kualitas.

![Gambar 3: Proses untuk mengoptimalkan desain prompt.](https://lmvr.rantai.dev//images/L6A504iArDGBvUq8WSeO-ksvKFPfdCNq9TcseE0OH-v1.png)
**Gambar 3:** Proses untuk mengoptimalkan desain prompt.

Penyempurnaan berkelanjutan sangat penting karena kemampuan LLM dan kebutuhan aplikasi terus berkembang. Pengembang dapat menggunakan metrik tingkat token (seperti tumpang tindih token/ROUGE) untuk mengukur keselarasan semantik.

### Contoh Python: Evaluasi dan Refinement Sederhana

```python
def calculate_mock_metrics(response, expected):
    # Simulasi perhitungan metrik sederhana
    res_set = set(response.lower().split())
    exp_set = set(expected.lower().split())
    
    overlap = len(res_set.intersection(exp_set))
    relevance = (overlap / len(exp_set)) * 100 if exp_set else 0
    
    return {"relevance": relevance, "clarity": 85.0} # Clarity placeholder

def refine_prompt(original_prompt, metrics):
    if metrics["relevance"] < 70.0:
        return original_prompt + " Pastikan untuk menyertakan detail teknis yang akurat."
    return original_prompt

def main():
    prompt = "Ringkas dokumen ini."
    response = "Ini adalah ringkasan singkat."
    expected = "Dokumen ini memberikan tinjauan mendalam tentang teknologi baru."
    
    metrics = calculate_mock_metrics(response, expected)
    print(f"Metrik Awal: {metrics}")
    
    refined = refine_prompt(prompt, metrics)
    print(f"Prompt yang Disempurnakan: {refined}")

if __name__ == "__main__":
    main()
```

---

## 20.5. Teknik Prompt Engineering Tingkat Lanjut

Teknik tingkat lanjut seperti *few-shot prompting*, *zero-shot learning*, dan *chain-of-thought prompting* memperluas kemampuan LLM secara signifikan.

*Few-shot prompting* dapat direpresentasikan sebagai fungsi pemetaan $f: \{(x_1, y_1), \dots, (x_n, y_n)\} \rightarrow y_{n+1}$. *Chain-of-thought prompting* mendorong model untuk memecah tugas kompleks menjadi langkah-langkah penalaran sekuensial, diformalkan sebagai $f(x) = (s_1, s_2, \dots, s_n) \rightarrow y$.

### Contoh Python: Few-shot vs Zero-shot

```python
def get_prompt_style(style_type, task, examples=None):
    if style_type == "few_shot" and examples:
        prompt = "Ikuti contoh berikut:\n"
        for ex in examples:
            prompt += f"Input: {ex['in']}\nOutput: {ex['out']}\n"
        prompt += f"Input: {task}\nOutput:"
        return prompt
    else:
        return f"Tugas: {task}\nJawablah dengan ringkas."

def main():
    examples = [
        {"in": "Rust", "out": "Bahasa pemrograman aman memori."},
        {"in": "Python", "out": "Bahasa pemrograman tingkat tinggi yang populer."}
    ]
    
    task = "TypeScript"
    
    print("--- Zero-Shot ---")
    print(get_prompt_style("zero_shot", task))
    
    print("\n--- Few-Shot ---")
    print(get_prompt_style("few_shot", task, examples))

if __name__ == "__main__":
    main()
```

---

## 20.6. Pertimbangan Etis dan Praktis

Tantangan etis utama dalam *prompt engineering* adalah mengelola risiko output yang bias. Secara matematis, respons model adalah fungsi dari prompt $P$ dan bobot yang dipelajari $W$, dinotasikan $f(P, W) \rightarrow R$. Bias dapat muncul jika $P$ memperkuat atribut tertentu dalam $W$.

### Contoh Python: Deteksi Bias Sederhana

```python
def detect_bias(text, sensitive_terms):
    detected = [term for term in sensitive_terms if term in text.lower()]
    fairness_score = 100.0 - (len(detected) * 10.0)
    return max(0.0, fairness_score), detected

def main():
    response = "Kandidat ini sangat cocok karena menunjukkan sifat kepemimpinan yang agresif."
    # Contoh kata yang mungkin perlu diaudit dalam konteks tertentu
    sensitive_words = ["pria", "wanita", "tua", "muda", "agresif"]
    
    score, found = detect_bias(response, sensitive_words)
    print(f"Skor Keadilan: {score}")
    print(f"Kata sensitif terdeteksi: {found}")

if __name__ == "__main__":
    main()
```

---

## 20.7. Studi Kasus dan Arah Masa Depan

*Prompt engineering* telah terbukti menjadi alat transformatif di berbagai bidang, mulai dari layanan kesehatan hingga keuangan. Tren masa depan mencakup otomatisasi prompt, pembuatan prompt dinamis, dan prompting personal yang disesuaikan dengan profil pengguna.

![Gambar 4: Contoh aplikasi LLM di beberapa industri seperti Layanan Kesehatan dan Keuangan.](https://lmvr.rantai.dev//images/L6A504iArDGBvUq8WSeO-V2xkZ9mocFYAeWFQAPXU-v1.png)
**Gambar 4:** Contoh aplikasi LLM di beberapa industri seperti Layanan Kesehatan dan Keuangan.

### Contoh Python: Generator Prompt Adaptif

```python
class AdaptivePromptGenerator:
    def __init__(self, base_prompt):
        self.base_prompt = base_prompt

    def generate(self, last_score):
        if last_score < 75:
            return self.base_prompt + " Mohon berikan jawaban yang lebih terstruktur dan jelas."
        return self.base_prompt

def main():
    gen = AdaptivePromptGenerator("Ringkas laporan keuangan ini.")
    # Skor rendah dari evaluasi sebelumnya
    current_prompt = gen.generate(last_score=60)
    print(f"Prompt yang diadaptasi: {current_prompt}")

if __name__ == "__main__":
    main()
```

Pengembangan aplikasi LLM mengikuti pipeline terstruktur, dimulai dari pemilihan model fondasi, *prompt engineering*, tuning, validasi, dan akhirnya penyebaran.

![Gambar 5: Arsitektur umum Aplikasi LLM dengan Prompt Engineering.](https://lmvr.rantai.dev//images/L6A504iArDGBvUq8WSeO-n5O8bwtgHH2ABky8BOku-v1.png)
**Gambar 5:** Arsitektur umum Aplikasi LLM dengan Prompt Engineering.

---

## 20.8. Kesimpulan

Bab 20 membekali pembaca dengan pengetahuan dan alat untuk menguasai *prompt engineering*. Dengan memahami nuansa desain prompt dan dampaknya terhadap perilaku model, pembaca dapat membuat prompt yang mengoptimalkan kinerja LLM sambil tetap mematuhi pedoman etika.

### 20.8.1. Pembelajaran Lebih Lanjut dengan GenAI

*   Jelaskan prinsip-prinsip fundamental dari *prompt engineering*.
*   Diskusikan tantangan utama dalam membuat prompt yang efektif untuk menghindari bias.
*   Deskripsikan proses pembangunan toolkit *prompt engineering* berbasis sistem.
*   Analisis dampak struktur prompt terhadap akurasi respons model.
*   Jelajahi peran pengetahuan domain dalam merancang prompt khusus aplikasi.
*   Jelaskan teknik *chain-of-thought prompting* dan manfaatnya untuk penalaran kompleks.
*   Bagaimana mekanisme umpan balik pengguna dapat digunakan untuk menyempurnakan prompt?

### 20.8.2. Latihan Praktis

#### Latihan Mandiri 20.1: Membuat Prompt Khusus Tugas
Tujuan: Merancang dan menguji variasi prompt untuk aplikasi layanan pelanggan guna memaksimalkan empati dan ketepatan jawaban.

#### Latihan Mandiri 20.2: Membangun Toolkit Prompting
Tujuan: Mengembangkan sistem sederhana di Python yang dapat mengelola templat prompt dan menyimpan hasil evaluasi respons model ke dalam file JSON.

#### Latihan Mandiri 20.3: Refinement Iteratif
Tujuan: Mengambil satu prompt peringkasan yang buruk, mengevaluasinya secara manual, dan memperbaikinya melalui tiga tahap iterasi.

#### Latihan Mandiri 20.4: Implementasi Chain-of-Thought
Tujuan: Merancang prompt yang menginstruksikan model untuk "berpikir langkah demi langkah" guna menyelesaikan masalah logika atau matematika.

#### Latihan Mandiri 20.5: Audit Etika pada Prompt
Tujuan: Mengidentifikasi potensi bias dalam prompt sistem rekrutmen otomatis dan merancang ulang prompt tersebut agar lebih netral secara gender dan latar belakang.