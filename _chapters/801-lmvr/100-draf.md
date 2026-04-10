---
slug: lmvr-10
title: lmvr-10
---

# Bab 10: Model Bahasa Fondasi Terbuka

> **"Masa depan AI terletak pada gerakan sumber terbuka, di mana model-model fondasi dapat diadaptasi dan ditingkatkan oleh komunitas, mendorong inovasi dan memastikan bahwa manfaat AI dapat diakses oleh semua orang." — Yann LeCun**

Bab 10 dari LMVR menawarkan eksplorasi mendalam tentang membangun, menyetel halus, dan menyebarkan model bahasa fondasi (LLM) terbuka menggunakan Python dan ekosistem Hugging Face. Bab ini dimulai dengan memperkenalkan signifikansi LLM fondasi terbuka dan keuntungan memanfaatkan model sumber terbuka dalam NLP. Kemudian disediakan panduan rinci untuk menyiapkan lingkungan pengembangan LLM, termasuk memuat dan menyetel halus model pra-terlatih untuk tugas-tugas tertentu. Bab ini juga mencakup penyebaran LLM, mendiskusikan strategi untuk mengoptimalkan skalabilitas, latensi, dan efisiensi sumber daya, serta mengeksplorasi teknik kustomisasi untuk memperluas kemampuan model. Terakhir, bab ini membahas tantangan dan arah masa depan dalam pengembangan LLM, menekankan peran efisiensi dalam mendorong batas-batas pencapaian model ini.

## 10.1. Pengantar Model Bahasa Fondasi

Munculnya model bahasa fondasi (LLM) telah merevolusi NLP dengan memungkinkan model yang dilatih pada dataset ekstensif untuk menangkap pola linguistik yang rumit dan dependensi kontekstual. Model seperti GPT dan BERT, yang dirancang dengan arsitektur serbaguna, unggul di berbagai tugas NLP, mulai dari analisis sentimen hingga tanya-jawab. Fleksibilitas mereka memungkinkan mereka untuk disetel halus untuk tugas-tugas tertentu dengan data minimal, menjadikannya landasan kemajuan NLP baru-baru ini. Model fondasi sumber terbuka dari Hugging Face lebih lanjut mendemokratisasi AI dengan menyediakan arsitektur yang transparan dan dapat dimodifikasi yang mendorong eksperimen dan inovasi luas. Peneliti, startup, dan perusahaan dapat menyesuaikan model tangguh ini untuk memenuhi kebutuhan spesifik tanpa bergantung pada sistem kepemilikan yang tertutup dan haus sumber daya.

Model fondasi sumber terbuka menumbuhkan aksesibilitas dan fleksibilitas, memungkinkan pengembang dari semua sektor—termasuk inovator skala kecil—untuk mengadaptasi model berkinerja tinggi ke aplikasi unik mereka, baik dalam layanan kesehatan, keuangan, atau media. Open LLM Leaderboard dari Hugging Face memperkaya ekosistem ini dengan menyediakan model-model tersebut secara terbuka, mempromosikan transparansi, dan mendorong kontribusi yang beragam dalam komunitas AI. Kemampuan adaptasi LLM sumber terbuka memberdayakan pengembang untuk membangun solusi khusus domain secara hemat biaya, memajukan inovasi berbasis AI di bidang khusus tanpa ketergantungan pada sistem yang mahal.

![Gambar 1: Open LLM Leaderboard dari Hugging Face.](https://lmvr.rantai.dev//images/L6A504iArDGBvUq8WSeO-GISEu9AbEpxEUcptLz49-v1.png)
**Gambar 1:** Open LLM Leaderboard dari Hugging Face.

Perbedaan utama dalam ekosistem AI adalah antara model fondasi dan model khusus tugas. Model fondasi bertindak sebagai arsitektur pra-terlatih yang digeneralisasi untuk menangkap pola bahasa, sedangkan model khusus tugas mengoptimalkan kemampuan fondasi ini untuk aplikasi tertentu. Transparansi dan reproduktifitas model sumber terbuka memungkinkan peneliti untuk memahami, memvalidasi, dan meningkatkan strukturnya, memastikan kinerja yang kuat di berbagai aplikasi. Namun, sifat sumber terbuka dari model fondasi juga memperkenalkan pertimbangan etis. Karena model ini sering dilatih pada data publik, bias dalam dataset pelatihan dapat merambat melalui model, menyebabkan output yang miring atau tidak diinginkan. Selain itu, data ekstensif yang diperlukan untuk melatih model semacam itu menimbulkan kekhawatiran privasi, karena model mungkin secara tidak sengaja mempelajari informasi sensitif. Menangani tantangan ini memerlukan upaya aktif dalam komunitas sumber terbuka untuk menghilangkan bias model dan menjaga privasi pengguna, mempromosikan pengembangan AI yang bertanggung jawab.

Menyiapkan lingkungan dimulai dengan memuat model fondasi seperti GPT-2 dan mengimplementasikan pipeline inferensi dasar. Pipeline ini akan menangani pemuatan model, tokenisasi, dan pembuatan teks, yang memungkinkan kita bereksperimen dengan dinamika input-output.

### Contoh Python: Tokenisasi dan Decoding GPT-2

```python
from transformers import GPT2Tokenizer

def main():
    # Memuat tokenizer GPT-2 dari Hugging Face
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Mendefinisikan prompt input
    input_text = "Dampak dari model sumber terbuka pada AI modern adalah"
    
    # Melakukan tokenisasi
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    
    # Mendekode token kembali menjadi teks yang dapat dibaca
    # Dalam skenario nyata, input_ids ini akan dikirim ke model untuk inferensi
    decoded_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    
    print(f"Teks Asli: {input_text}")
    print(f"Teks yang didekode: {decoded_text}")
    print(f"ID Token: {input_ids[0].tolist()}")

if __name__ == "__main__":
    main()
```

Bergerak ke inferensi tingkat lanjut, model fondasi dapat diperluas untuk menangani berbagai tugas NLP seperti klasifikasi teks dan pengenalan entitas bernama (NER). Memperluas pipeline memberikan eksplorasi yang lebih luas tentang kapasitas model dan memungkinkan pengembang untuk menilai kinerja di berbagai aplikasi.

### Contoh Python: Simulasi Klasifikasi Teks dan NER

```python
from transformers import pipeline

# Fungsi untuk klasifikasi teks
def classify_text(text):
    classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    result = classifier(text)
    return result[0]['label']

# Fungsi untuk Pengenalan Entitas Bernama (NER)
def recognize_entities(text):
    ner = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")
    results = ner(text)
    return [(res['word'], res['entity']) for res in results]

def main():
    # Tugas 1: Klasifikasi Teks
    text_cls = "Teknologi baru ini sangat inovatif dan mendobrak"
    print(f"Hasil Klasifikasi: {classify_text(text_cls)}")

    # Tugas 2: Named Entity Recognition (NER)
    text_ner = "Elon Musk meluncurkan model Tesla baru di California"
    entities = recognize_entities(text_ner)
    print("Entitas yang dikenali:")
    for entity, label in entities:
        print(f"Entitas: {entity}, Label: {label}")

if __name__ == "__main__":
    main()
```

Model fondasi terbuka sangat mudah beradaptasi dan menemukan aplikasi di berbagai industri. Tren terbaru dalam LLM fondasi menekankan teknik optimasi, seperti kuantisasi dan distilasi model, untuk membuat model lebih kecil dan lebih efisien untuk disebarkan di lingkungan dunia nyata.

## 10.2. Menyiapkan Lingkungan untuk Pengembangan LLM

Menyiapkan lingkungan yang disesuaikan untuk pengembangan model bahasa (LLM) memungkinkan pengembang untuk memanfaatkan kontrol tingkat sistem dan fitur keamanan memori untuk tugas-tugas pembelajaran mesin. Dibandingkan dengan bahasa seperti Python, bahasa seperti Rust memberikan lingkungan yang lebih aman dengan biaya kurva pembelajaran yang lebih curam tetapi menawarkan keuntungan performa yang signifikan. Namun, dalam banyak kasus, pendekatan hibrida yang menggabungkan sumber daya deep learning Python yang luas dengan kemampuan sistem bahasa berkinerja tinggi menciptakan lingkungan yang sangat cocok untuk aplikasi LLM skala besar yang berkinerja tinggi.

Untuk memulai dengan Hugging Face di Python, pengembang menginstal library utama menggunakan pip:

```bash
pip install transformers datasets torch accelerate
```

Setelah library terinstal, menyiapkan contoh dasar memberikan keakraban dengan fungsionalitas model fondasi.

### Contoh Python: Simulasi Pembuatan Teks Sederhana

```python
from transformers import AutoTokenizer

def main():
    # Memuat tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Prompt input
    input_text = "Menjelajahi kekuatan Python dalam pembelajaran mesin."
    
    # Tokenisasi
    encoding = tokenizer.encode(input_text)
    
    # Simulasi pembuatan teks dengan memodifikasi token (hanya untuk ilustrasi proses)
    # Dalam aplikasi nyata, ini dihasilkan oleh model.forward()
    output_tokens = [token + 1 for token in encoding] 
    
    # Dekode kembali
    generated_text = tokenizer.decode(output_tokens)
    
    print(f"Teks yang dihasilkan (Simulasi): {generated_text}")

if __name__ == "__main__":
    main()
```

Dalam industri, penyiapan ini semakin banyak diterapkan di bidang-bidang yang menuntut kinerja tinggi dan keamanan, seperti keuangan, layanan kesehatan, dan sistem otonom.

## 10.3. Memuat dan Menggunakan Model Pra-terlatih

Model pra-terlatih adalah landasan NLP modern, menawarkan kemampuan luas tanpa mengharuskan model dibangun dari nol. Alur kerja ini memungkinkan pengembang untuk melewati proses pelatihan yang ekstensif, sebaliknya memanfaatkan pola linguistik yang terakumulasi dan pengetahuan umum yang dikodekan dalam model ini. Keuntungan transfer learning bahkan lebih menonjol di domain khusus seperti keuangan atau layanan kesehatan.

Seri LLaMA (Large Language Model Meta AI), yang dikembangkan oleh Meta AI, dirancang untuk melakukan tugas pemrosesan bahasa alami (NLP) secara efisien. Model-model ini, seperti varian LLaMA-2-7b, fokus pada penyediaan kinerja tinggi dengan parameter yang lebih sedikit dibandingkan dengan model bahasa besar lainnya.

![Gambar 3: Demo LLaMA 2 di Hugging Face (Ref: https://huggingface.co/spaces/lmz/candle-llama2).](https://lmvr.rantai.dev//images/L6A504iArDGBvUq8WSeO-8Tv8PqU5Q2zFhtK8hIds-v1.png)
**Gambar 2:** Demo LLaMA 2 di Hugging Face.

### Contoh Python: Pembuatan Teks dengan LLaMA 2

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    model_id = "meta-llama/Llama-2-7b-hf"
    
    # Memerlukan token akses Hugging Face yang valid
    # tokenizer = AutoTokenizer.from_pretrained(model_id)
    # model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")

    prompt = "Teorema favorit saya adalah "
    
    # inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    # output = model.generate(**inputs, max_new_tokens=50, temperature=0.8, do_sample=True)
    # print(tokenizer.decode(output[0], skip_special_tokens=True))
    print("(Contoh memerlukan akses ke model LLaMA 2 dan GPU)")

if __name__ == "__main__":
    main()
```

Model Phi adalah keluarga model bahasa yang dikembangkan untuk melakukan berbagai tugas NLP dengan efisiensi dan presisi yang ditingkatkan. Model Phi bertujuan untuk menyeimbangkan ukuran model dan efisiensi komputasi, memungkinkan inferensi yang lebih cepat dan konsumsi energi yang lebih rendah tanpa mengorbankan akurasi.

### Contoh Python: Pembuatan Teks dengan Model Phi-2

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def main():
    model_id = "microsoft/phi-2"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32, trust_remote_code=True)

    prompt = "Teorema favorit saya adalah "
    inputs = tokenizer(prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=50, temperature=0.8, do_sample=True)
    
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Hasil: {text}")

if __name__ == "__main__":
    main()
```

## 10.4. Penyetelan Halus LLM Fondasi Terbuka

Penyetelan halus (fine-tuning) sangat penting untuk mengadaptasi model bahasa besar ke tugas atau domain tertentu, memungkinkan mereka untuk menangani aplikasi khusus yang mungkin tidak ditangkap sepenuhnya oleh model pra-terlatih umum. Beberapa strategi yang ada meliputi penyetelan halus terawasi (supervised fine-tuning), few-shot learning, dan pra-pelatihan khusus domain.

### Contoh Python: Trainer Sederhana untuk Analisis Sentimen (PyTorch)

```python
import torch
import torch.nn as nn
import torch.optim as optim

class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, num_labels):
        super(SimpleClassifier, self).__init__()
        self.linear = nn.Linear(input_dim, num_labels)

    def forward(self, x):
        return self.linear(x)

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleClassifier(768, 2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    # Data dummy (embeddings dan labels)
    inputs = torch.randn(8, 768).to(device)
    labels = torch.tensor([1, 0, 1, 0, 1, 0, 1, 0]).to(device)

    model.train()
    for epoch in range(5):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

if __name__ == "__main__":
    train()
```

Penyetelan halus model pra-terlatih telah menjadi pendekatan standar di berbagai industri. Tren terbaru menekankan efisiensi sumber daya dan adaptasi lintas domain menggunakan teknik seperti LoRA (Low-Rank Adaptation).

## 10.5. Menyebarkan LLM

Menyebarkan model bahasa besar (LLM) ke lingkungan produksi menghadirkan serangkaian tantangan unik, yang memerlukan perencanaan matang seputar skalabilitas, latensi, dan efisiensi sumber daya. Memantau dan memelihara LLM yang disebarkan sangat penting untuk memitigasi penyimpangan model (model drift). Penggunaan alat orkestrasi seperti Docker dan Kubernetes sangat menguntungkan dalam penyebaran LLM.

![Gambar 4: Proses penyebaran LLM.](https://lmvr.rantai.dev//images/L6A504iArDGBvUq8WSeO-SpRcgXO66RZMCHQbAS59-v1.png)
**Gambar 4:** Proses penyebaran LLM.

### Contoh Python: REST API untuk Prediksi Model (FastAPI)

```python
from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import pipeline

app = FastAPI()
classifier = pipeline("sentiment-analysis")

class RequestBody(BaseModel):
    text: str

@app.post("/predict")
def predict(body: RequestBody):
    result = classifier(body.text)
    return {"result": result}

if __name__ == "__main__":
    import uvicorn
    # uvicorn.run(app, host="0.0.0.0", port=8000)
```

Pemantauan kinerja model yang disebarkan sangat penting untuk mempertahankan efektivitasnya di produksi. Metrik seperti latensi inferensi, penggunaan memori, dan akurasi model harus dianalisis secara berkala.

## 10.6. Memperluas dan Menyesuaikan LLM

Model LLM fondasi terbuka menyediakan kerangka kerja fleksibel yang dapat diadaptasi untuk memenuhi kebutuhan spesifik. Melalui kustomisasi arsitektur, seperti menambahkan lapisan baru atau mengubah mekanisme atensi, pengembang dapat mendorong batas-batas LLM fondasi.

### Contoh Python: Menambahkan Lapisan Kustom pada Model BERT

```python
import torch.nn as nn
from transformers import BertModel

class CustomBertModel(nn.Module):
    def __init__(self, model_name):
        super(CustomBertModel, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.custom_layer = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU()
        )
        self.classifier = nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        refined_output = self.custom_layer(pooled_output)
        return self.classifier(refined_output)

# model = CustomBertModel("bert-base-uncased")
```

Kustomisasi memungkinkan model AI yang responsif dan disesuaikan untuk memenuhi permintaan yang muncul, memungkinkan pengembangan aplikasi inovatif di berbagai bidang seperti sistem otonom dan analitik data.

## 10.7. Tantangan dan Arah Masa Depan

Pengembangan dan penyebaran LLM fondasi terbuka membawa serangkaian tantangan kritis, mulai dari masalah skalabilitas, privasi data, hingga tanggung jawab etis. Tren yang muncul seperti model multimodal, few-shot learning, dan interpretabilitas menghadirkan batas-batas baru untuk dieksplorasi.

Teknik optimasi model seperti *pruning*—metode yang menghapus koneksi redundan dalam jaringan saraf—sangat berguna untuk menyebarkan LLM pada perangkat dengan sumber daya terbatas.

### Contoh Python: Pruning Sederhana pada Model PyTorch

```python
import torch.nn.utils.prune as prune
import torch.nn as nn

def apply_pruning(model):
    # Terapkan pruning 20% pada bobot lapisan linier
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=0.2)
            prune.remove(module, 'weight') # Membuat pruning permanen
    print("Pruning selesai.")

# model = nn.Linear(512, 512)
# apply_pruning(model)
```

Di masa depan, aplikasi LLM siap untuk meluas melampaui tugas-tugas berbasis teks saat ini, dengan pemrosesan multimodal dan waktu nyata yang menjadi semakin relevan.

## 10.8. Kesimpulan

Bab 10 membekali pembaca dengan pengetahuan dan alat untuk secara efektif membangun, menyesuaikan, dan menyebarkan LLM fondasi terbuka. Dengan menguasai teknik-teknik ini, pembaca akan siap untuk berkontribusi pada pengembangan dan demokratisasi AI yang berkelanjutan.

### 10.8.1. Pembelajaran Lebih Lanjut dengan GenAI

- Jelaskan signifikansi model bahasa fondasi terbuka dalam konteks NLP.
- Deskripsikan proses pengaturan lingkungan pengembangan untuk LLM.
- Diskusikan peran ekosistem Hugging Face dalam pemuatan model pra-terlatih.
- Analisis trade-off antara menggunakan model pra-terlatih dan melatih model dari nol.
- Diskusikan strategi penyetelan halus yang berbeda untuk LLM.
- Jelajahi tantangan penyebaran model bahasa besar di produksi.
- Jelaskan proses kuantisasi model dan dampaknya terhadap latensi.
- Analisis pentingnya kualitas dan keragaman data dalam fine-tuning.
- Jelajahi pertimbangan etis dalam menggunakan LLM fondasi terbuka.
- Deskripsikan proses integrasi model multimodal.

### 10.8.2. Latihan Praktis

#### Latihan Mandiri 10.1: Penyetelan Halus LLM Terbuka untuk Tugas Spesifik
Tujuan: Melatih model fondasi pra-terlatih pada dataset analisis sentimen lokal dan mengevaluasi kinerjanya.

#### Latihan Mandiri 10.2: Menyebarkan LLM dengan Kontainerisasi
Tujuan: Membangun API REST sederhana dan membungkusnya menggunakan Docker agar siap dijalankan di server mana pun.

#### Latihan Mandiri 10.3: Implementasi Kuantisasi Model
Tujuan: Mengurangi ukuran model menggunakan teknik kuantisasi dan membandingkan kecepatan inferensi sebelum dan sesudah optimasi.

#### Latihan Mandiri 10.4: Menambahkan Lapisan Kustom
Tujuan: Memodifikasi arsitektur model BERT dengan menambahkan lapisan tambahan di bagian akhir untuk menyesuaikan representasi output.

#### Latihan Mandiri 10.5: Menangani Data di Luar Distribusi
Tujuan: Mengimplementasikan mekanisme untuk mendeteksi input yang sangat berbeda dari data pelatihan (out-of-distribution) saat inferensi.