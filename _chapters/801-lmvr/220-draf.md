---
slug: lmvr-22
title: lmvr-22
---
# Bab 22: Teknik Prompt Engineering Tingkat Lanjut

> **"Teknik prompting tingkat lanjut sangat penting untuk membuka potensi penuh dari model bahasa, memungkinkan kita untuk memandu perilaku mereka dan meningkatkan kinerja mereka dengan cara yang semakin canggih." — Yann LeCun**

> **Ringkasan Bab:**
> Bab 22 menggali teknik prompt engineering tingkat lanjut untuk model bahasa besar, dengan fokus pada metode inovatif yang melampaui prompting few-shot dan zero-shot tradisional. Bab ini mencakup Chain of Thought Prompting, yang meningkatkan interpretabilitas model dengan menghasilkan langkah-langkah penalaran menengah; Meta Prompting, yang memengaruhi perilaku model melalui prompt yang dibuat secara strategis; Self-Consistency Prompting, yang memastikan keandalan dengan menghasilkan beberapa output untuk konsistensi; dan Generate Knowledge Prompting, yang memanfaatkan prompt untuk memancing pengetahuan tertentu. Selain itu, bab ini mengeksplorasi teknik lanjutan seperti Prompt Chaining, Tree of Thoughts, Automatic Prompt Engineering, Active-Prompt, ReAct Prompting, Reflexion Prompting, Multi-Modal Chain of Thought, dan Graph Prompting, yang masing-masing menawarkan cara unik untuk meningkatkan kinerja dan adaptabilitas model.

## 22.1. Chain of Thought Prompting

Chain of Thought (CoT) prompting adalah teknik tingkat lanjut dalam pemrosesan bahasa alami yang dirancang untuk mendorong model bahasa besar (LLM) menghasilkan langkah-langkah penalaran menengah sebelum memberikan jawaban akhir. Pendekatan ini sangat berharga dalam skenario yang menuntut penalaran multi-langkah, deduksi logis, atau pemecahan informasi yang kompleks, karena memandu model untuk "berpikir keras." Dengan meniru metode pemecahan masalah manusia, CoT prompting meningkatkan akurasi dan interpretabilitas output model, memungkinkan model untuk mencapai kesimpulan yang lebih jelas dan lebih andal. Menstrukturkan prompt untuk meminta langkah-langkah menengah ini memungkinkan model untuk mensimulasikan proses berpikir yang koheren, meningkatkan transparansi, dan memungkinkan pengguna untuk mengikuti logika model lebih dekat, yang sangat bermanfaat untuk aplikasi di mana jalur keputusan harus jelas dan dapat dibenarkan.

![Gambar 1: Ilustrasi CoT dari https://www.promptingguide.ai/techniques/cot](https://lmvr.rantai.dev//images/L6A504iArDGBvUq8WSeO-8iqqzQOhiK9GLxDo5DqS-v1.webp)
**Gambar 1:** Ilustrasi CoT (Kredit: promptingguide.ai)

Dalam CoT prompting, output model distrukturkan untuk mengikuti urutan langkah-langkah penalaran, direpresentasikan secara matematis sebagai $R = (r_1, r_2, \ldots, r_n)$, di mana setiap $r_i$ adalah langkah menengah yang menuju ke jawaban akhir $A$. Struktur ini mengubah output menjadi urutan kondisional, didefinisikan sebagai $P(A | X) = \prod_{i=1}^n P(r_i | r_{1:i-1}, X)$, di mana $X$ adalah input awal. Setiap langkah $r_i$ menjadi lapisan dalam jalur pengambilan keputusan, memungkinkan pengguna untuk mengamati dan memverifikasi setiap komponen dari rantai penalaran tersebut. Struktur ini tidak hanya meningkatkan transparansi tetapi juga memungkinkan koreksi yang ditargetkan jika ditemukan kesalahan dalam langkah apa pun, meningkatkan ketangguhan dan keandalan dalam aplikasi berisiko tinggi.

### Contoh Python: Chain of Thought Prompting (Menggunakan LangChain)

```python
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain

# Penyiapan Prompt CoT: Prompt untuk penalaran multi-langkah
template = """Selesaikan masalah ini langkah demi langkah:

Masalah: Sebuah kereta menempuh jarak 60 mil dalam 1 jam, kemudian 80 mil dalam 2 jam. Berapa total jarak yang ditempuh?

Langkah 1: Tentukan jarak yang ditempuh dalam jam pertama.
Langkah 2: Tentukan jarak yang ditempuh dalam dua jam berikutnya.
Langkah 3: Jumlahkan jarak dari setiap langkah untuk menemukan total jarak.

Solusi:"""

prompt = PromptTemplate(template=template, input_variables=[])
llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key="sk-proj-...")

chain = LLMChain(llm=llm, prompt=prompt)

# Eksekusi Prompting CoT
result = chain.invoke({})
print(f"Hasil Chain of Thought: {result['text']}")
```

## 22.2. Meta Prompting

Meta prompting adalah teknik canggih dalam prompt engineering yang memungkinkan pengembang untuk memengaruhi tidak hanya konten tetapi juga nada, gaya, adaptabilitas, dan perilaku keseluruhan dari respons model bahasa. Dengan melangkah melampaui prompt langsung yang sederhana, meta prompting menciptakan lapisan abstraksi di mana prompt awal mendefinisikan pedoman, instruksi, atau kriteria spesifik yang akan diikuti model saat menghasilkan respons berikutnya. Pendekatan ini memungkinkan model untuk menyesuaikan output-nya secara dinamis berdasarkan konteks, niat pengguna, atau persyaratan spesifik.

![Gambar 2: Ilustrasi Meta prompting dari https://www.promptingguide.ai/techniques/meta-prompting](https://lmvr.rantai.dev//images/L6A504iArDGBvUq8WSeO-QKNUj0jiZFpwnsmNVkHB-v1.webp)
**Gambar 2:** Ilustrasi Meta prompting (Kredit: promptingguide.ai)

Secara matematis, kita dapat mendefinisikan meta prompting sebagai fungsi iteratif dan referensi diri di mana prompt $P$ tidak secara langsung menghasilkan respons $R$ melainkan menghasilkan prompt yang diperhalus $P'$ yang membentuk output model berikutnya. Jika $f$ adalah fungsi yang mewakili pembuatan output model, maka dalam meta prompting, kita bertujuan untuk merancang $P$ sedemikian rupa sehingga $f(P) = P'$ dan $f(P') = R$.

### Contoh Python: Meta Prompting (Gaya Percakapan)

```python
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Meta Prompt: Menentukan nada dan gaya
system_template = "Untuk percakapan ini, gunakan nada yang ramah dan percakapan yang cocok untuk siswa muda."
human_template = "Jelaskan konsep gravitasi dalam istilah sederhana."

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", system_template),
    ("human", human_template),
])

llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key="sk-proj-...")
response = llm.invoke(chat_prompt.format_messages())

print(f"Respons dengan Meta Prompting: {response.content}")
```

## 22.3. Self-Consistency Prompting

Self-consistency prompting adalah teknik tingkat lanjut yang bertujuan untuk meningkatkan keandalan dan akurasi output LLM, khususnya dalam tugas-tugas CoT. Metode ini mengganti decoding serakah tunggal dengan strategi yang mengambil sampel beberapa jalur penalaran yang beragam. Jawaban yang paling konsisten di antara berbagai jalur penalaran yang berbeda ini kemudian dipilih sebagai output akhir.

Secara matematis, self-consistency dapat direpresentasikan sebagai mekanisme pemungutan suara atau konsensus. Misalkan $R = \{r_1, r_2, \ldots, r_n\}$ adalah sekumpulan $n$ respons yang dihasilkan untuk prompt $P$. Output akhir $R_{\text{konsisten}}$ dipilih berdasarkan frekuensi atau koherensi dari respons-respons tersebut:

$$ R_{\text{consistent}} = \operatorname{arg\,max}_{r \in R} \sum_{i \neq j} \text{similarity}(r_i, r_j) $$

### Contoh Python: Self-Consistency Prompting

```python
from langchain_openai import ChatOpenAI
from collections import Counter

llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key="sk-proj-...")
prompt = "Apa manfaat utama menggunakan Rust untuk pemrograman sistem?"

responses = []
# Menghasilkan 5 respons untuk konsistensi
for _ in range(5):
    res = llm.invoke(prompt)
    responses.append(res.content)

# Memilih respons yang paling konsisten (menggunakan frekuensi sederhana sebagai proxy)
# Dalam praktik, gunakan similarity scoring yang lebih kompleks
most_common_response = Counter(responses).most_common(1)[0][0]

print(f"Respons Paling Konsisten: {most_common_response}")
```

## 22.4. Generate Knowledge Prompting

Generate Knowledge Prompting bertujuan untuk memancing respons yang informatif, kaya konteks, dan didorong oleh pengetahuan dari LLM. Alih-alih fokus pada jawaban langsung, teknik ini mendorong model untuk menarik representasi dasarnya dan "menghasilkan" pengetahuan dalam bentuk penjelasan atau wawasan mendalam sebelum memberikan jawaban akhir.

![Gambar 3: Ilustrasi Generate Knowledge prompting dari https://www.promptingguide.ai/techniques/knowledge](https://lmvr.rantai.dev//images/L6A504iArDGBvUq8WSeO-ZaemenODLpEl19dUvawc-v1.webp)
**Gambar 3:** Ilustrasi Generate Knowledge prompting (Kredit: promptingguide.ai)

Secara matematis, ini dapat dilihat sebagai fungsi respons multi-tahap:
$$R = f(P) = \sum_{i=1}^{n} k_i$$
di mana setiap $k_i$ mewakili komponen pengetahuan individu yang dihasilkan sebagai respons terhadap prompt.

### Contoh Python: Generate Knowledge Prompting

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key="sk-proj-...")

# Prompt yang mendorong model untuk menguraikan topik secara mendalam
prompt = """Jelaskan konsep 'ekosistem' dalam ekologi. 
Bahas komponen-komponennya, fungsinya, dan berikan contoh dari berbagai jenis yang berbeda."""

response = llm.invoke(prompt)
print(f"Respons Berbasis Pengetahuan: {response.content}")
```

## 22.5. Prompt Chaining

Prompt chaining meningkatkan keandalan LLM dengan memecah tugas kompleks menjadi urutan prompt yang lebih kecil dan saling terkait. Respons dari satu prompt digunakan sebagai input untuk prompt berikutnya. Metode ini meningkatkan transparansi karena pengembang dapat melacak dan men-debug setiap langkah.

Secara matematis, ini dapat direpresentasikan sebagai proses di mana status setiap prompt $P_n$ bergantung pada respons prompt sebelumnya $R_{n-1}$:
$$ R_f = f(P_1, R_1, P_2, R_2, \dots, P_n, R_n) $$

### Contoh Python: Prompt Chaining

```python
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key="sk-proj-...")

# Tahap 1: Mengumpulkan latar belakang
p1 = "Berikan informasi latar belakang tentang kinerja pasar terbaru sebuah perusahaan teknologi."
res1 = llm.invoke(p1).content

# Tahap 2: Mengidentifikasi masalah berdasarkan latar belakang
p2 = f"Berdasarkan latar belakang berikut, identifikasi masalah utama yang dihadapi perusahaan: {res1}"
res2 = llm.invoke(p2).content

# Tahap 3: Menyarankan solusi
p3 = f"Mempertimbangkan masalah yang teridentifikasi: {res2}, sarankan solusi yang dapat mengatasinya."
res3 = llm.invoke(p3).content

print(f"Solusi Akhir: {res3}")
```

## 22.6. Tree of Thoughts Prompting

Tree of Thoughts (ToT) adalah kerangka kerja yang memungkinkan model untuk menangani tugas kompleks yang memerlukan eksplorasi strategis. ToT mengatur proses penalaran ke dalam struktur pohon, di mana setiap "pemikiran" adalah langkah menengah. Model dapat mengevaluasi berbagai jalur pemikiran menggunakan algoritma pencarian seperti BFS atau DFS.

![Gambar 4: Ilustrasi CoT, CoT-SC dan ToT dari https://www.promptingguide.ai/techniques/tot](https://lmvr.rantai.dev//images/L6A504iArDGBvUq8WSeO-9X5RNf8TTq3k8KPfpjJ2-v1.webp)
**Gambar 4:** Ilustrasi ToT (Kredit: promptingguide.ai)

$$ \text{Jalur Terbaik} = \operatorname{arg\,max}_{b \in T(P)} f(b) $$

### Contoh Python: Tree of Thoughts Sederhana

```python
def evaluate_path(path):
    # Skor berdasarkan keberadaan kata kunci sukses
    score = len(path)
    if "pertumbuhan" in path.lower(): score += 10
    return score

# Simulasi ToT sederhana
root_prompt = "Sarankan pendekatan untuk meningkatkan keterlibatan pengguna di platform sosial."
# LLM menghasilkan 3 cabang awal (simulasi)
branches = [
    "Fokus pada fitur video pendek.",
    "Tingkatkan sistem gamifikasi.",
    "Perbaiki algoritma feed berita."
]

best_path = max(branches, key=evaluate_path)
print(f"Jalur Pemikiran Terbaik: {best_path}")
```

## 22.7. Automatic Prompt Engineer

Automatic Prompt Engineering (APE) mengotomatiskan pembuatan dan evaluasi prompt menggunakan algoritma. APE membingkai prompt engineering sebagai masalah sintesis dan pencarian bahasa alami.

$$P^* = \operatorname{arg\,max}_{P \in P} f(P)$$

### Contoh Python: Optimasi Prompt Sederhana

```python
def generate_candidates(base):
    return [base + " dengan fokus pada pertumbuhan pengguna", 
            base + " dengan mempertimbangkan data demografis"]

def evaluate_mock(prompt):
    return len(prompt) # Simulasi skor

base_prompt = "Analisis tren media sosial"
candidates = generate_candidates(base_prompt)
best_prompt = max(candidates, key=evaluate_mock)

print(f"Prompt yang Dioptimalkan: {best_prompt}")
```

## 22.8. Penalaran Otomatis (Automatic Reasoning)

Penalaran Otomatis memberdayakan model untuk menyimpulkan kesimpulan dan menerapkan logika terstruktur tanpa instruksi detail. ART (Automatic Reasoning Tool) memungkinkan LLM untuk menghasilkan langkah penalaran dan menjeda proses untuk memanggil alat eksternal jika diperlukan.

![Gambar 6: Ilustrasi Penalaran Otomatis dari https://www.promptingguide.ai/techniques/art](https://lmvr.rantai.dev//images/L6A504iArDGBvUq8WSeO-jq1Q1mXEIVGZEV8ZbBJK-v1.webp)
**Gambar 6:** Ilustrasi ART (Kredit: promptingguide.ai)

### Contoh Python: Prompt Penalaran Sederhana

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key="sk-proj-...")
reasoning_prompt = """Evaluasi opsi investasi berikut untuk investor konservatif:
Opsi 1: Pengembalian tinggi, risiko tinggi.
Opsi 2: Pengembalian moderat, risiko moderat.
Opsi 3: Pengembalian rendah, risiko rendah.
Identifikasi opsi terbaik dan jelaskan alasannya secara logis."""

print(llm.invoke(reasoning_prompt).content)
```

## 22.9. Active-Prompt

Active-Prompt adalah pendekatan dinamis di mana model mengidentifikasi pertanyaan yang paling tidak pasti hasilnya dan memintanya untuk dianotasi oleh manusia. Hasil anotasi kemudian dimasukkan kembali ke dalam prompt untuk memperbaiki respons di masa depan.

$$ P_{i+1} = A(P_i, F_i, R_i) $$

### Contoh Python: Loop Active-Prompt Sederhana

```python
def adjust_prompt(last_response, feedback):
    if "klarifikasi" in feedback:
        return f"Dapatkah Anda mengklarifikasi bagian ini: {last_response}"
    return last_response

# Simulasi interaksi
current_prompt = "Selamat datang di dukungan pelanggan. Ada yang bisa dibantu?"
feedback = "mohon klarifikasi"
new_prompt = adjust_prompt("Saya mengerti masalah Anda.", feedback)
print(f"Prompt Baru: {new_prompt}")
```

## 22.10. ReAct Prompting

ReAct (Reason + Act) memungkinkan LLM menghasilkan jejak penalaran dan tindakan khusus tugas secara bergantian. Ini membantu memitigasi halusinasi fakta dengan membiarkan model berinteraksi dengan sumber eksternal (seperti Wikipedia).

![Gambar 7: Ilustrasi teknik ReAct prompting dari https://www.promptingguide.ai/techniques/react](https://lmvr.rantai.dev//images/L6A504iArDGBvUq8WSeO-4LBxbjx68Jg11eFYQmEa-v1.webp)
**Gambar 7:** Ilustrasi ReAct (Kredit: promptingguide.ai)

$$ R_i = \begin{cases} \text{reflect}(R_{i-1}) & \text{jika } i \text{ ganjil} \\ \text{act}(R_{i-1}) & \text{jika } i \text{ genap} \end{cases} $$

### Contoh Python: Struktur ReAct Sederhana

```python
def react_logic(user_q):
    # Tahap Pemikiran (Thought)
    thought = f"Saya perlu menganalisis risiko dari pertanyaan: {user_q}"
    # Tahap Tindakan (Action)
    action = f"Berdasarkan analisis '{thought}', rekomendasinya adalah investasi konservatif."
    return thought, action

t, a = react_logic("Apa investasi terbaik untuk masa pensiun saya?")
print(f"Pemikiran: {t}\nTindakan: {a}")
```

## 22.11. Reflexion Prompting

Reflexion Prompting memungkinkan model untuk melakukan evaluasi diri dan merevisi responsnya sendiri. Ini sangat berguna di bidang yang memerlukan presisi tinggi seperti sains atau hukum.

$$ R_{i+1} = f(P, R_i) $$

### Contoh Python: Loop Refleksi Diri

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key="sk-proj-...")
initial_q = "Apa penyebab utama jatuhnya Kekaisaran Romawi?"

# Langkah 1: Jawaban awal
ans = llm.invoke(initial_q).content

# Langkah 2: Refleksi dan perbaikan
reflection_p = f"Tinjau jawaban berikut: '{ans}'. Identifikasi ketidakakuratan dan berikan versi yang lebih komprehensif."
final_ans = llm.invoke(reflection_p).content

print(f"Jawaban Akhir Setelah Refleksi:\n{final_ans}")
```

## 22.12. Multi-Modal Chain of Thought Prompting

Multi-Modal CoT memungkinkan LLM mengintegrasikan input dari berbagai jenis data (teks, gambar, audio). Model mensintesis informasi dari setiap sumber untuk membangun konteks yang detail sebelum memberikan jawaban.

![Gambar 8: Ilustrasi multi-modal CoT prompt dari https://www.promptingguide.ai/techniques/multimodalcot](https://lmvr.rantai.dev//images/L6A504iArDGBvUq8WSeO-EKlgBB8ZDa8cN97FVshz-v1.png)
**Gambar 8:** Ilustrasi multi-modal CoT prompt (Kredit: promptingguide.ai)

$$ f(P) = f(T, I) = f(f_{\text{text}}(T), f_{\text{image}}(I)) $$

### Contoh Python: Analisis Multimodal Sederhana (Logika)

```python
def multimodal_analysis(description, image_url):
    # Simulasi analisis teks
    text_insight = f"Analisis sejarah dari: {description}"
    # Simulasi analisis gambar
    image_insight = f"Fitur visual dari gambar di {image_url}"
    
    return f"{text_insight}. Selain itu, {image_insight} menunjukkan konteks tambahan."

print(multimodal_analysis("Helm perunggu Yunani kuno", "http://example.com/item.jpg"))
```

## 22.13. Graph Prompting

Graph Prompting memanfaatkan graf pengetahuan untuk meningkatkan pemahaman model. Dengan mengodekan hubungan hierarkis antar entitas dalam format graf, model dapat menafsirkan dependensi yang kompleks dengan lebih efektif.

![Gambar 9: Motivasi Graph Prompting dari Liu et.al (https://arxiv.org/pdf/2302.08043).](https://lmvr.rantai.dev//images/L6A504iArDGBvUq8WSeO-BnhLKylRYnWC2nOLELWt-v1.png)
**Gambar 9:** Motivasi Graph Prompting.

### Contoh Python: Prompt Berbasis Graf (Menggunakan Dict)

```python
knowledge_graph = {
    "Fisika": ["Matematika", "Teknik"],
    "Biologi": ["Kimia", "Kedokteran"]
}

def graph_prompt(query):
    related = knowledge_graph.get(query, [])
    context = ", ".join(related)
    return f"Jelaskan hubungan antara {query} dan disiplin terkaitnya: {context}."

print(graph_prompt("Fisika"))
```

## 22.14. Kesimpulan

Bab 22 menyediakan eksplorasi komprehensif tentang teknik prompt engineering tingkat lanjut. Dengan menguasai metode-metode ini, pembaca dapat secara signifikan meningkatkan efektivitas dan fleksibilitas model bahasa besar, mendorong batas kemampuan aplikasi AI di dunia nyata.

### 22.14.1. Pembelajaran Lebih Lanjut dengan GenAI

- Jelaskan teknik Chain of Thought (CoT) secara mendalam dan mekanismenya.
- Bagaimana Meta Prompting memengaruhi pembuatan respons model?
- Bahas metode Self-Consistency untuk meningkatkan keandalan respons.
- Analisis peran Generate Knowledge Prompting dalam pencarian pengetahuan target.
- Rinci teknik Prompt Chaining untuk interaksi yang kompleks.
- Jelajahi struktur Tree of Thoughts (ToT) untuk eksplorasi solusi sistematis.
- Bagaimana bias dapat dideteksi dalam teknik prompting tingkat lanjut?
- Diskusikan trade-off antara spesifisitas dan generalitas prompt.

### 22.14.2. Latihan Praktis

#### Latihan Mandiri 22.1: Implementasi Chain of Thought
Tujuan: Merancang prompt multi-langkah untuk masalah logika matematika yang kompleks.

#### Latihan Mandiri 22.2: Membuat Meta Prompt
Tujuan: Mengembangkan meta-prompt yang mengatur model untuk berbicara sebagai "Penasihat Keuangan" yang sangat formal.

#### Latihan Mandiri 22.3: Eksplorasi Self-Consistency
Tujuan: Menjalankan kueri yang sama sebanyak 5 kali dan membuat kode Python untuk memilih jawaban yang paling sering muncul.

#### Latihan Mandiri 22.4: Menerapkan Generate Knowledge
Tujuan: Membuat prompt yang meminta model mengumpulkan fakta tentang ekologi sebelum membuat kebijakan lingkungan.

#### Latihan Mandiri 22.5: Membangun Rantai Prompt (Prompt Chaining)
Tujuan: Membuat sistem di mana respons pertama (ringkasan berita) diproses oleh prompt kedua (analisis sentimen) dan prompt ketiga (saran tindakan).