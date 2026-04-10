---
slug: lmvr-13
title: lmvr-13
---

# Bab 13: Inferensi dan Penyebaran LLM

> **"Kekuatan nyata dari AI tidak hanya terletak pada melatih model-model besar, tetapi juga pada menyebarkannya secara efektif dan efisien di berbagai lingkungan untuk menciptakan dampak nyata bagi dunia." — Andrew Ng**

Bab 13 dari LMVR berfokus pada inferensi dan penyebaran (deployment) model bahasa besar yang efisien. Bab ini dimulai dengan menjelaskan pentingnya mengoptimalkan pipeline inferensi dan mengeksplorasi berbagai teknik seperti kuantisasi model, pemangkasan (pruning), dan pemrosesan batch. Kemudian dibahas mengenai penyebaran LLM di lingkungan produksi, penggunaan API, kontainerisasi, dan alat orkestrasi seperti Docker dan Kubernetes. Bab ini juga menggali penskalaan beban kerja inferensi, baik secara horizontal maupun vertikal, dan menyoroti tantangan unik penyebaran di perangkat edge, termasuk batasan sumber daya dan efisiensi daya. Selain itu, bab ini membahas aspek kritis dalam mengamankan dan memelihara model yang telah disebarkan, memastikan keandalan dan keamanan jangka panjang. Melalui studi kasus dunia nyata, bab ini memberikan wawasan praktis dalam menyebarkan LLM secara efektif.

## 13.1. Pengantar Inferensi dan Penyebaran

Inferensi adalah proses di mana model bahasa besar (LLM) mengubah parameter yang telah dipelajari menjadi prediksi dan wawasan yang dapat ditindaklanjuti, yang berfungsi sebagai langkah terakhir yang menjembatani pelatihan model dengan aplikasi praktis. Inferensi adalah tempat model terlatih menerapkan pemahamannya pada tugas-tugas dunia nyata, menghasilkan output berdasarkan data input baru tanpa modifikasi lebih lanjut pada parameter model. Inferensi untuk LLM sering kali melibatkan menanggapi kueri bahasa alami, menghasilkan ringkasan, atau menawarkan rekomendasi berbasis bahasa. Strategi penyebaran yang sukses untuk inferensi LLM melibatkan pertimbangan seperti skalabilitas, latensi, dan efisiensi sumber daya, yang masing-masing penting untuk memastikan responsivitas model dan efektivitas biaya di lingkungan produksi. Berbeda dengan pelatihan, yang memprioritaskan akurasi melalui iterasi yang ekstensif, inferensi menekankan kecepatan dan konsistensi, di mana latensi dan penggunaan memori menjadi sangat penting.

![Figure 1: Alur proses inferensi dan penyebaran LLM.](https://lmvr.rantai.dev//images/L6A504iArDGBvUq8WSeO-sEAla7D6J7qjkg82lm8Z-v1.png)
**Gambar 1:** Alur proses inferensi dan penyebaran LLM.

Penyebaran LLM menghadirkan tantangan unik dibandingkan dengan pelatihan, karena inferensi menuntut pemrosesan yang cepat dan overhead yang rendah sambil melayani volume permintaan yang berpotensi tinggi. Perbedaan utama antara pelatihan dan inferensi meliputi sifat beban komputasi, pola akses memori, dan aliran data. Selama pelatihan, fokusnya adalah pada propagasi mundur (backpropagation) dan optimasi parameter model, yang sering kali memerlukan pemanfaatan memori yang besar dan daya komputasi untuk mengakomodasi pemrosesan batch yang besar. Sebaliknya, beban kerja inferensi biasanya lebih ringan tetapi harus diproses dalam waktu yang mendekati nyata (near real-time), sering kali satu permintaan pada satu waktu atau dalam batch yang lebih kecil untuk memenuhi permintaan pengguna. Pergeseran ini menempatkan tuntutan baru pada strategi alokasi sumber daya, karena inferensi sering kali mendapat manfaat dari pengoptimalan yang mengurangi penggunaan memori dan meminimalkan latensi komputasi.

Inferensi dan penyebaran memerlukan serangkaian trade-off, terutama dalam menyeimbangkan akurasi model, kecepatan pemrosesan, dan pemanfaatan sumber daya. Akurasi yang lebih tinggi mungkin melibatkan penggunaan arsitektur model yang lebih kompleks, yang cenderung mengonsumsi lebih banyak sumber daya komputasi dan menyebabkan waktu respons yang lebih lama. Sebaliknya, mengurangi kompleksitas model dapat mempercepat inferensi tetapi mungkin mengorbankan akurasi. Rust sangat cocok untuk menyetel keseimbangan ini karena manajemen memori yang efisien dan eksekusi latensi rendah, memungkinkan pengembang untuk mengoptimalkan inferensi model untuk berbagai lingkungan penyebaran. Ini termasuk cloud, on-premises, dan lingkungan edge, masing-masing dengan batasan dan kelebihannya sendiri. Penyebaran cloud menawarkan skalabilitas tinggi dan cocok untuk menangani beban yang berfluktuasi, meskipun mungkin memperkenalkan latensi karena ketergantungan jaringan. Penyebaran on-premises memberikan lebih banyak kontrol atas privasi data dan latensi tetapi memerlukan investasi perangkat keras awal yang besar. Penyebaran edge membawa inferensi lebih dekat ke pengguna akhir, mengurangi latensi dan ketergantungan jaringan, yang sangat berharga dalam aplikasi seluler dan perangkat IoT.

Mengoptimalkan pipeline inferensi sangat penting untuk memastikan bahwa LLM memberikan respons yang akurat dan tepat waktu dalam aplikasi dunia nyata. Pipeline inferensi yang efektif menangani pra-pemrosesan data, pemuatan model, dan waktu respons melalui praktik pengkodean yang efisien dan pemanfaatan perangkat keras. Mengurangi jejak model melalui teknik seperti kuantisasi, yang melibatkan pengurangan presisi bobot model (misalnya, dari FP32 ke INT8), dapat mengurangi penggunaan memori dan mempercepat perhitungan. Optimalisasi inferensi lebih lanjut disempurnakan dengan mempertimbangkan pemrosesan batch data, mekanisme caching, dan penyeimbangan beban (load balancing) untuk menangani volume permintaan puncak tanpa mengorbankan waktu respons.

Contoh Python berikut menunjukkan pipeline inferensi untuk model berbasis BERT guna menghasilkan embedding untuk tugas kesamaan kalimat (sentence similarity).

### Contoh Python: Inferensi BERT untuk Kesamaan Kalimat

```python
from sentence_transformers import SentenceTransformer, util
import torch
import time

def main():
    # 1. Memuat model pra-terlatih
    # Model 'all-MiniLM-L6-v2' sangat efisien untuk inferensi cepat
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # 2. Daftar kalimat contoh
    sentences = [
        "The cat sits outside",
        "A man is playing guitar",
        "I love pasta",
        "The new movie is awesome",
        "The cat plays in the garden",
        "A woman watches TV",
        "The new movie is so great",
        "Do you like pizza?"
    ]

    # 3. Menghasilkan Embeddings
    start_time = time.time()
    embeddings = model.encode(sentences, convert_to_tensor=True)
    print(f"Bentuk tensor embedding: {embeddings.shape}")
    print(f"Waktu yang dibutuhkan untuk encoding: {time.time() - start_time:.4f} detik")

    # 4. Menghitung Kesamaan Kosinus antar semua pasangan
    cosine_scores = util.cos_sim(embeddings, embeddings)

    # 5. Mencari pasangan dengan skor tertinggi
    similarities = []
    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            similarities.append((cosine_scores[i][j].item(), i, j))

    # Urutkan berdasarkan skor tertinggi
    similarities.sort(key=lambda x: x[0], reverse=True)

    print("\nTop 5 Pasangan Kalimat Paling Mirip:")
    for score, i, j in similarities[:5]:
        print(f"Skor: {score:.2f} | '{sentences[i]}' VS '{sentences[j]}'")

if __name__ == "__main__":
    main()
```

Penyebaran LLM di dunia nyata sering kali melibatkan tantangan seperti latensi penyajian model dan alokasi sumber daya yang efisien. Latensi penyajian model dapat dikelola dengan menggunakan caching untuk permintaan yang sering, memuat komponen model secara awal (pre-loading), dan mengurangi waktu komputasi dengan teknik kuantisasi atau pemangkasan. Penyeimbangan beban juga sangat penting dalam aplikasi dengan lalu lintas tinggi untuk mendistribusikan permintaan secara merata ke seluruh node server.

## 13.2. Mengoptimalkan Pipeline Inferensi

Mengoptimalkan pipeline inferensi sangat penting untuk menyebarkan LLM secara efisien, dengan tujuan memaksimalkan kecepatan dan mengurangi biaya komputasi tanpa mengorbankan akurasi model. Teknik seperti kuantisasi model, pemangkasan, dan pemrosesan batch umum digunakan. Kuantisasi mengurangi bit presisi dari bobot model, sementara pemangkasan menghapus parameter yang kurang kritis. Pemrosesan batch menggabungkan beberapa input untuk diproses secara bersamaan, meningkatkan throughput.

![Gambar 2: Pipeline inferensi untuk penyebaran LLM.](https://lmvr.rantai.dev//images/L6A504iArDGBvUq8WSeO-50OTZwwXKTk0pNuHpjUD-v1.png)
**Gambar 2:** Pipeline inferensi untuk penyebaran LLM.

Kuantisasi model memetakan nilai presisi tinggi $x$ ke representasi terkuantisasi $q(x)$ dengan presisi terbatas:
$$ q(x) = \text{round}\left(\frac{x - \text{min}}{\text{scale}}\right) \times \text{scale} + \text{min} $$

Pemangkasan (pruning) menghapus elemen bobot $w_i$ jika $|w_i| < \epsilon$, di mana $\epsilon$ adalah ambang batas yang ditentukan. Ini menghasilkan jaringan yang lebih jarang (sparse), mengurangi kebutuhan memori dan waktu komputasi.

Open Neural Network Exchange (ONNX) adalah format sumber terbuka yang dirancang untuk interoperabilitas model AI lintas kerangka kerja. ONNX memungkinkan model yang dikembangkan di PyTorch atau TensorFlow untuk disebarkan di lingkungan produksi yang mendukung ONNX Runtime.

### Contoh Python: Mengevaluasi Model ONNX dengan ONNX Runtime

```python
import onnxruntime as ort
import numpy as np

def simple_eval(model_path):
    # 1. Inisialisasi session ONNX Runtime
    session = ort.InferenceSession(model_path)

    # 2. Mendapatkan informasi input model
    input_info = session.get_inputs()
    inputs = {}
    
    for node in input_info:
        # Membuat tensor dummy nol berdasarkan bentuk dan tipe data model
        shape = [dim if isinstance(dim, int) else 1 for dim in node.shape]
        inputs[node.name] = np.zeros(shape).astype(np.float32)
        print(f"Menyiapkan input '{node.name}' dengan bentuk {shape}")

    # 3. Menjalankan inferensi
    outputs = session.run(None, inputs)
    
    # 4. Mencetak hasil output
    output_info = session.get_outputs()
    for i, node in enumerate(output_info):
        print(f"Output '{node.name}': {outputs[i].shape}")

if __name__ == "__main__":
    # simple_eval("path/to/model.onnx")
    pass
```

### Contoh Python: Pipeline Inferensi SqueezeNet (ONNX)

```python
import onnxruntime as ort
from PIL import Image
import numpy as np

def preprocess_image(image_path):
    img = Image.open(image_path).resize((224, 224))
    img_data = np.array(img).transpose(2, 0, 1).astype(np.float32)
    # Normalisasi ImageNet standar
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    img_data = (img_data / 255.0 - mean) / std
    return img_data[np.newaxis, :] # Menambahkan dimensi batch

def main():
    model_path = "squeezenet1.1.onnx"
    image_path = "test.jpg"
    
    # Load model
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    
    # Preprocess image
    input_data = preprocess_image(image_path)
    
    # Run Inference
    outputs = session.run(None, {input_name: input_data})
    
    # Post-process (Top 5)
    logits = outputs[0]
    # Implementasi softmax sederhana
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / exp_logits.sum()
    
    top5_idx = np.argsort(probs[0])[-5:][::-1]
    print("Hasil Prediksi SqueezeNet:")
    for idx in top5_idx:
        print(f"Kelas ID {idx}: {probs[0][idx]*100:.2f}%")

if __name__ == "__main__":
    # main()
    pass
```

## 13.3. Melayani LLM di Lingkungan Produksi

Penyajian model (model serving) melibatkan penyediaan mekanisme yang terstruktur dan efisien bagi pengguna untuk berinteraksi dengan LLM melalui API, kontainer, dan alat orkestrasi. Kontainer seperti Docker merampingkan penyebaran dengan menciptakan lingkungan yang konsisten, sementara Kubernetes mengelola kontainer ini dalam skala besar, memungkinkan penyeimbangan beban dan toleransi kesalahan.

![Gambar 3: Pipeline Model Serving untuk LLM.](https://lmvr.rantai.dev//images/L6A504iArDGBvUq8WSeO-cZM3B7olXb0jwmbVoTQG-v1.png)
**Gambar 3:** Pipeline Model Serving untuk LLM.

Arsitektur pipeline penyajian model yang kuat mencakup penyeimbangan beban, toleransi kesalahan, dan pertimbangan keamanan. Keamanan dalam pipeline penyajian LLM memerlukan kontrol akses dan enkripsi yang ketat karena interaksi pengguna mungkin melibatkan data sensitif.

Berikut adalah implementasi Python menggunakan FastAPI untuk melayani model Mistral atau model serupa yang tersedia di Hugging Face.

### Contoh Python: Menyajikan Model Mistral via REST API (FastAPI)

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = FastAPI()

# Inisialisasi model dan tokenizer
# Untuk efisiensi, gunakan model terkuantisasi (misal via bitsandbytes)
MODEL_ID = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
# model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map="auto")

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.8

@app.post("/generate")
async def generate_text(request: GenerateRequest):
    try:
        # Simulasi proses generate jika model tidak dimuat untuk menghemat RAM lokal
        # inputs = tokenizer(request.prompt, return_tensors="pt").to("cuda")
        # outputs = model.generate(**inputs, max_new_tokens=request.max_tokens, temperature=request.temperature)
        # response_