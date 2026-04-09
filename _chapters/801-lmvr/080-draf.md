---
slug: lmvr-8
title: lmvr-8
---
# Bab 8: Transformer Multimodal dan Ekstensi

> 💡 **"Masa depan AI terletak pada pembelajaran multimodal, di mana pengintegrasian informasi dari berbagai modalitas akan menghasilkan model yang lebih kaya dan sadar konteks yang dapat memahami dunia seperti halnya manusia." — Fei-Fei Li**

> 📘 **Ringkasan Bab:**
> Bab 8 dari LMVR menyediakan eksplorasi mendalam tentang Transformer multimodal dan ekstensinya, dengan fokus pada integrasi berbagai modalitas data seperti teks, gambar, dan audio. Bab ini dimulai dengan memperkenalkan dasar-dasar pembelajaran multimodal, menyoroti pentingnya menggabungkan sumber data yang beragam untuk menciptakan representasi yang lebih kaya dan kuat. Kemudian, bab ini mendalami adaptasi arsitektur Transformer untuk tugas-tugas multimodal, membahas teknik-teknik utama seperti self-attention, cross-attention, dan berbagai strategi fusi. Bab ini juga mencakup signifikansi pre-training dan fine-tuning model multimodal, bersama dengan ekstensi tingkat lanjut seperti ViLBERT dan UNITER yang mendorong batas kemampuan model-model ini. Terakhir, bab ini membahas tantangan dan arah masa depan dalam pembelajaran multimodal, menekankan potensi inovasi di bidang-bidang seperti layanan kesehatan, mengemudi otonom, dan interaksi manusia-komputer.

---

## 8.1. Pengantar Pembelajaran Multimodal

Pembelajaran multimodal adalah kerangka kerja yang kuat dalam pembelajaran mesin yang bertujuan untuk menyatukan informasi dari berbagai sumber data, seperti teks, gambar, dan audio. Dengan menggabungkan modalitas yang berbeda ini, model dapat menghasilkan representasi yang lebih komprehensif dan sadar konteks. Berbeda dengan pendekatan unimodal yang hanya menangkap satu sumber informasi, pembelajaran multimodal memanfaatkan kualitas unik dari setiap modalitas. Misalnya, data teks sering kali mengodekan informasi sintaksis dan semantik secara berurutan, sementara gambar berisi petunjuk spasial dan visual yang disusun secara kontinu. Integrasi ini memungkinkan model untuk memanfaatkan kekuatan komplementer dari setiap jenis data, yang sangat bermanfaat dalam tugas-tugas seperti pemberian teks pada gambar (*image captioning*), analisis sentimen multimodal, dan menjawab pertanyaan visual (*visual question answering*). Namun, pembelajaran multimodal yang efektif memerlukan penyelesaian tantangan yang kompleks, termasuk penyelarasan dan fusi modalitas, yang harus menjembatani perbedaan inheren dalam struktur data dan semantik lintas modalitas.

Sebagai contoh dalam pembelajaran multimodal berbasis graf, berbagai jenis data seperti gambar, teks, dan dataset ilmiah direpresentasikan sebagai node atau lapisan dalam struktur graf yang terpadu. Pendekatan ini memanfaatkan jaringan saraf graf (*graph neural networks*) untuk menangkap hubungan antar modalitas dengan menghubungkan node melalui tautan lintas modalitas, memungkinkan representasi data kompleks yang holistik dan kaya konteks. Setiap modalitas memberikan kontribusi atribut uniknya—seperti informasi sekuensial dari teks, petunjuk spasial dari gambar, atau data relasional dari bidang ilmiah—memungkinkan peningkatan performa dalam tugas-tugas seperti prediksi, klasifikasi, dan penalaran interdisipliner di berbagai domain yang saling terhubung.

![Figure 1: Ilustrasi paradigma pembelajaran multimodal.](https://lmvr.rantai.dev/images/L6A504iArDGBvUq8WSeO-5KSTrd65BW0ROkAZbFzn-v1.jpeg)

Dalam konteks model bahasa besar, satu teknik canggih dalam pembelajaran multimodal adalah *cross-attention* (atensi silang), yang secara dinamis memprioritaskan informasi relevan dari setiap modalitas. Dalam cross-attention, mekanisme atensi belajar untuk fokus pada fitur utama dari satu modalitas yang bersangkutan dengan modalitas lainnya. Misalnya, model yang dilatih untuk memproses data teks dan gambar mungkin menggunakan cross-attention untuk mengidentifikasi wilayah gambar yang sesuai dengan frasa tertentu dalam teks. Secara matematis, misalkan $X_t$ dan $X_i$ masing-masing merepresentasikan embedding teks dan gambar. Cross-attention memungkinkan model untuk menghitung skor atensi antara kedua modalitas ini, menghasilkan representasi fusi $Z$ yang dipengaruhi oleh konteks dari sumber teks dan gambar. Proses ini direpresentasikan sebagai berikut:

$$ Z = \text{softmax} \left( \frac{Q_t K_i^T}{\sqrt{d_k}} \right) V_i + \text{softmax} \left( \frac{Q_i K_t^T}{\sqrt{d_k}} \right) V_t $$

di mana $Q_t, K_t, V_t$ adalah matriks query, key, dan value untuk modalitas teks, dan $Q_i, K_i, V_i$ adalah matriks query, key, dan value untuk modalitas gambar. Mekanisme cross-attention ini menghasilkan bobot yang menyoroti interaksi lintas modalitas, menghasilkan representasi yang menangkap petunjuk kontekstual timbal balik.

![Figure 2: Ilustrasi cross-attention (Ref: https://arxiv.org/html/2403.03431v1).](https://lmvr.rantai.dev/images/L6A504iArDGBvUq8WSeO-A0uRZjwcfvAAD225vJfy-v1.png)

Dalam pembelajaran multimodal, cross-attention adalah teknik ampuh yang secara dinamis memprioritaskan informasi kunci di berbagai jenis data. Mekanisme ini mempelajari skor atensi antar modalitas, memungkinkan model untuk fokus pada fitur di satu modalitas yang relevan dengan modalitas lainnya. Misalnya, dalam pengemudian otonom, data visual dan sensor dapat difusi, dengan setiap modalitas menginformasikan pemahaman situasional model. Secara formal, cross-attention menghitung representasi fusi, $Z$, menggunakan skor atensi yang diturunkan dari query, key, dan value dari setiap modalitas, menangkap petunjuk kontekstual lintas modalitas dan menghasilkan embedding bersama untuk integrasi yang lebih baik.

Implementasi Python berikut (sebagai padanan untuk contoh Rust) menunjukkan arsitektur transformer multimodal sederhana menggunakan PyTorch. Dalam model ini, setiap modalitas—teks dan gambar—memiliki encodernya sendiri yang memproses data input secara independen. Lapisan cross-attention kemudian menggabungkan representasi ini.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# Definisi struktur blok transformer
class TransformerBlock(nn.Module):
    def __init__(self, dim):
        super(TransformerBlock, self).__init__()
        self.attention = nn.Linear(dim, dim)
        self.feed_forward = nn.Linear(dim, dim)
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x):
        attended = F.softmax(self.attention(x), dim=-1)
        ff_out = self.feed_forward(attended)
        return self.layer_norm(ff_out + x)

# Model Multimodal dengan Cross-Attention
class MultimodalTransformer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MultimodalTransformer, self).__init__()
        self.text_encoder = TransformerBlock(input_dim)
        self.image_encoder = nn.Linear(input_dim, input_dim)
        self.cross_attention = TransformerBlock(input_dim)
        # Lapisan output disetel dua kali lipat input_dim untuk tensor yang digabungkan
        self.output_layer = nn.Linear(2 * input_dim, output_dim)

    def forward(self, text, image):
        # Encode teks dan gambar
        text_encoded = self.text_encoder(text)
        image_encoded = self.image_encoder(image)

        # Cross-attention: fusi representasi teks dan gambar
        text_cross_attended = self.cross_attention(text_encoded + image_encoded)
        image_cross_attended = self.cross_attention(image_encoded + text_encoded)

        # Gabungkan (konkatenasi pada dimensi fitur) dan proyeksikan ke output
        combined = torch.cat([text_cross_attended, image_cross_attended], dim=1)
        
        return self.output_layer(combined)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    input_dim = 512
    output_dim = 128

    # Inisialisasi model
    model = MultimodalTransformer(input_dim, output_dim).to(device)

    # Data dummy untuk demonstrasi
    text_data = torch.randn(1, input_dim).to(device)
    image_data = torch.randn(1, input_dim).to(device)

    # Forward pass
    output = model(text_data, image_data)
    print(f"Output dari transformer multimodal: {output.shape}")

if __name__ == "__main__":
    main()
```

Mengevaluasi transformer multimodal ini terhadap model unimodal biasanya menunjukkan peningkatan performa yang signifikan, terutama dalam tugas-tugas di mana integrasi kontekstual sangat krusial. Kapabilitas fusi ini sangat berharga dalam aplikasi seperti visual question answering, di mana menafsirkan petunjuk visual dalam konteks teks sangatlah penting.

![Figure 3: Ilustrasi arsitektur CLIP-BERT.](https://lmvr.rantai.dev/images/L6A504iArDGBvUq8WSeO-OlgzYreScElFkdDVZ8dY-v1.png)

Berikut adalah contoh Python untuk pencocokan gambar-teks menggunakan model CLIP (padanan untuk contoh Rust menggunakan `candle`):

```python
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

def main():
    # Memuat model dan processor CLIP
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Gambar dan teks contoh
    image_path = "path/to/your/image.jpg" # Ganti dengan path gambar yang ada
    try:
        image = Image.open(image_path)
    except:
        print("Gambar tidak ditemukan, menggunakan data acak.")
        image = Image.new('RGB', (224, 224))

    texts = ["a cycling race", "a photo of two cats", "a robot holding a candle"]

    # Prapemrosesan
    inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)

    # Inference
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Mendapatkan probabilitas kecocokan gambar-teks
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)

    print(f"Hasil pencocokan untuk gambar: {image_path}")
    for i, text in enumerate(texts):
        print(f"Probabilitas: {probs[0][i].item()*100:.2f}% untuk teks: '{text}'")

if __name__ == "__main__":
    main()
```

Sebagai kesimpulan, implementasi Transformer multimodal yang canggih ini menyoroti kekuatan mekanisme cross-attention untuk menyelaraskan dan menyatukan modalitas yang berbeda. Kombinasi kontrol tingkat rendah dan fleksibilitas tingkat tinggi menjadikan lingkungan pengembangan seperti Rust atau Python pilihan yang sangat baik untuk mengimplementasikan sistem AI yang canggih dan sadar konteks.

---

## 8.2. Arsitektur Transformer untuk Pembelajaran Multimodal

Dalam pembelajaran multimodal, arsitektur Transformer telah menjadi pusat perhatian karena kemampuannya untuk mengintegrasikan informasi dari beberapa aliran data. Kekuatan inti dari Transformer terletak pada mekanisme self-attention miliknya. Untuk tugas-tugas multimodal, kemampuan self-attention ini dapat diadaptasi menjadi mekanisme cross-attention yang memungkinkan interaksi antara aliran data yang berbeda.

Dalam Transformer multimodal, lapisan self-attention beroperasi di dalam setiap modalitas untuk menangkap hubungan intra-modal, sementara lapisan cross-attention diperkenalkan untuk mengintegrasikan data lintas modalitas. Diberikan dua modalitas—teks $T$ dan gambar $I$—lapisan cross-attention dapat direpresentasikan sebagai:

$$ A_{TI} = \text{softmax}\left(\frac{Q_T K_I^T}{\sqrt{d_k}}\right) V_I $$

Beberapa model Transformer multimodal terkemuka meliputi ViLBERT, UNITER, dan MMBERT. ViLBERT menggunakan dua aliran atensi terpisah yang dihubungkan oleh cross-attention, sementara UNITER menggunakan arsitektur aliran tunggal untuk mengodekan gambar dan teks secara bersama-sama.

![Figure 4: Ilustrasi arsitektur ViLBERT.](https://lmvr.rantai.dev/images/L6A504iArDGBvUq8WSeO-CBmyqXIKs5pWuLl2CSTb-v1.png)

ViLBERT (*Vision-and-Language BERT*) dirancang untuk mengintegrasikan informasi visual dan tekstual menggunakan arsitektur aliran ganda untuk pengodean independen. Berikut adalah representasi konseptual ViLBERT di Python:

```python
import torch
import torch.nn as nn

class ViLBERT(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super(ViLBERT, self).__init__()
        # Encoder terpisah
        self.image_encoder = nn.Linear(embed_dim, hidden_dim)
        self.text_encoder = nn.Linear(embed_dim, hidden_dim)
        
        # Mekanisme Co-Attention (simulasi sederhana)
        self.co_attention = nn.MultiheadAttention(hidden_dim, num_heads=8)
        self.classifier = nn.Linear(hidden_dim, 2)

    def forward(self, img_feats, text_feats):
        img_h = self.image_encoder(img_feats)
        text_h = self.text_encoder(text_feats)
        
        # Co-attention: Teks memperhatikan Gambar
        # (q, k, v)
        attn_output, _ = self.co_attention(text_h, img_h, img_h)
        
        return self.classifier(attn_output)
```

UNITER (*UNiversal Image-TExt Representation*) menyelaraskan modalitas ini dalam satu encoder Transformer terpadu dengan memproyeksikan wilayah gambar dan token teks ke dalam ruang embedding bersama.

![Figure 5: Ilustrasi arsitektur UNITER.](https://lmvr.rantai.dev/images/L6A504iArDGBvUq8WSeO-4UBTQx3BWB7WUPIYQcs7-v1.jpeg)

```python
# Konsep Sederhana UNITER di Python
class UNITER(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.transformer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8)
        self.classifier = nn.Linear(hidden_dim, 2)

    def forward(self, img_embeds, text_embeds):
        # Gabungkan secara sekuensial
        joint_input = torch.cat([img_embeds, text_embeds], dim=1)
        encoded = self.transformer(joint_input)
        # Ambil representasi rata-rata atau token CLS
        return self.classifier(encoded.mean(dim=1))
```

Secara keseluruhan, mengadaptasi arsitektur Transformer untuk pembelajaran multimodal menawarkan kemampuan yang kuat untuk tugas-tugas yang memerlukan pemahaman terintegrasi di seluruh sumber data. Penggunaan atensi lintas modal dan teknik fusi data tingkat lanjut memungkinkan pengembangan sistem berkinerja tinggi.

---

## 8.3. Teknik Fusi Multimodal

Dalam pembelajaran multimodal, teknik fusi memainkan peran penting dalam menggabungkan informasi dari berbagai sumber data. Strategi fusi ini meliputi *early fusion* (fusi awal), *late fusion* (fusi akhir), dan *hybrid fusion* (fusi hibrida).

![Figure 6: Ilustrasi teknik fusi multimodal.](https://lmvr.rantai.dev/images/L6A504iArDGBvUq8WSeO-QJhPOg69iY7fmjV5m3IY-v1.png)

1.  **Early Fusion:** Menggabungkan data mentah di tingkat input. $X_{combined} = f(X_t, X_i)$. Memungkinkan model mempelajari interaksi lintas modal sejak awal.
2.  **Late Fusion:** Memproses setiap modalitas secara independen dan menggabungkan outputnya di tingkat abstraksi yang lebih tinggi. $F_{final} = \text{Combine}(F_t, F_i)$.
3.  **Hybrid Fusion:** Menggabungkan elemen fusi awal dan akhir untuk menyeimbangkan kekuatan masing-masing.

Berikut adalah implementasi ketiga teknik fusi tersebut di Python:

```python
class FusionModels:
    # Early Fusion: Gabung di awal
    @staticmethod
    def early_fusion(text_emb, img_emb, projection_layer):
        combined = torch.cat([text_emb, img_emb], dim=1)
        return projection_layer(combined)

    # Late Fusion: Gabung di akhir setelah pemrosesan independen
    @staticmethod
    def late_fusion(text_out, img_out, final_layer):
        combined = torch.cat([text_out, img_out], dim=1)
        return final_layer(combined)

    # Hybrid Fusion: Gabung di berbagai tahap
    @staticmethod
    def hybrid_fusion(text, img, early_layer, late_layer):
        # Tahap awal
        e_fused = torch.cat([text, img], dim=1)
        e_out = early_layer(e_fused)
        
        # Tahap akhir (misal gabung lagi dengan input asli)
        l_fused = torch.cat([e_out, text + img], dim=1)
        return late_layer(l_fused)
```

Setiap strategi fusi memiliki implikasi berbeda pada performa model dan kebutuhan sumber daya. Strategi hibrida semakin populer karena kemampuannya menangkap hubungan tingkat rendah dan tingkat tinggi secara efektif.

---

## 8.4. Pre-Training dan Fine-Tuning Transformer Multimodal

Pre-training memungkinkan Transformer multimodal untuk mengembangkan representasi lintas modal dengan memaparkannya pada dataset besar yang berisi berbagai modalitas. Tugas pre-training yang umum meliputi *masked language modeling* (MLM), *image-text matching* (ITM), dan *visual grounding*.

$$ \mathcal{L}_{MLM} = -\log p(X_t^{\text{masked}} | X_t^{\text{context}}, X_i) $$
$$\mathcal{L}_{ITM} = -\log p(\text{match} | X_t, X_i)$$

Setelah pre-training, *fine-tuning* memungkinkan model untuk berspesialisasi dalam tugas hilir tertentu. Tantangan utamanya adalah menghindari *overfitting* pada data multimodal yang kompleks melalui teknik seperti pembekuan lapisan (*selective freezing*) atau augmentasi data.

Berikut adalah contoh logika fine-tuning dengan pembekuan lapisan di Python:

```python
def fine_tune_multimodal(model, train_loader):
    # Membekukan encoder teks untuk menjaga representasi pre-trained
    for param in model.text_encoder.parameters():
        param.requires_grad = False
        
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    
    for epoch in range(10):
        for batch in train_loader:
            text, img, label = batch
            output = model(text, img)
            loss = F.cross_entropy(output, label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
```

---

## 8.5. Ekstensi dan Aplikasi Transformer Multimodal

Ekstensi tingkat lanjut seperti MMBERT, ViLBERT, dan UNITER telah dirancang untuk menangani interaksi rumit antara teks dan gambar.

![Figure 7: Ilustrasi arsitektur multimodal BERT (MMBERT) untuk gambar medis.](https://lmvr.rantai.dev/images/L6A504iArDGBvUq8WSeO-hnorzf3YOOqcMZi7EqNq-v1.jpeg)

Aplikasi nyata meliputi:
- **Layanan Kesehatan:** Menggabungkan rontgen dada dengan laporan medis terkait untuk diagnosis otomatis.
- **Pengemudian Otonom:** Fusi sensor dari LIDAR, radar, dan kamera.
- **Hiburan:** Analisis konten video (frame, audio, subtitle) untuk rekomendasi konten.

Contoh Python untuk MMBERT sederhana (klasifikasi biner teks-gambar):

```python
class MMBERT(nn.Module):
    def __init__(self, text_dim, hidden_dim):
        super().__init__()
        self.text_enc = nn.Linear(text_dim, hidden_dim)
        # CNN sederhana untuk gambar
        self.img_enc = nn.Sequential(
            nn.Conv2d(1, 32, 3), nn.ReLU(), nn.Flatten(),
            nn.Linear(32 * 222 * 222, hidden_dim) # Sesuaikan ukuran output
        )
        self.fusion = nn.Linear(hidden_dim * 2, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 2)

    def forward(self, text, img):
        t_feat = self.text_enc(text)
        i_feat = self.img_enc(img)
        fused = torch.relu(self.fusion(torch.cat([t_feat, i_feat], dim=1)))
        return self.classifier(fused)
```

---

## 8.6. Tantangan dan Arah Masa Depan

Tantangan utama yang dihadapi meliputi kelangkaan data (*data scarcity*), ketidakseimbangan modalitas (*modality imbalance*), dan penyelarasan jenis data heterogen. Tren yang muncul adalah integrasi lebih banyak modalitas (audio, sensor) dan penggunaan pembelajaran mandiri (*self-supervised learning*) untuk memanfaatkan data tak berlabel.

MobileCLIP adalah contoh kemajuan dalam efisiensi, menggunakan encoder gambar berbasis FastViT untuk performa tinggi di perangkat seluler dengan sumber daya terbatas.

```python
# Simulasi seleksi model MobileCLIP di Python
def get_mobileclip_config(variant="S1"):
    if variant == "S1":
        return {"model": "apple/MobileCLIP-S1-OpenCLIP", "image_size": 224}
    # Varian lain...
```

Secara keseluruhan, bidang pembelajaran multimodal yang berkembang pesat ini membuka jalan bagi inovasi dalam berbagai domain, termasuk robotika, *augmented reality* (AR), dan AI yang dipersonalisasi.

---

## 8.7. Kesimpulan

Bab 8 membekali pembaca dengan pemahaman komprehensif tentang Transformer multimodal dan ekstensinya. Dengan menguasai konsep-konsep ini, pembaca akan siap untuk mengembangkan sistem AI tingkat lanjut yang mampu memproses dan mengintegrasikan berbagai modalitas data.

### 8.7.1. Pembelajaran Lebih Lanjut dengan GenAI

- Eksplorasi bagaimana model multimodal mengatasi kebisingan spesifik modalitas.
- Analisis strategi fusi tercanggih untuk data asinkron.
- Selidiki peran mekanisme cross-attention dalam menyoroti fitur paling relevan lintas modalitas.
- Bandingkan fitur unik arsitektur ViLBERT, UNITER, dan MMBERT.
- Diskusikan implikasi etis dari model multimodal (privasi, amplifikasi bias).

### 8.7.2. Praktik Mandiri

- **Latihan 8.1:** Implementasikan teknik *early*, *late*, dan *hybrid fusion* lalu bandingkan akurasinya pada satu dataset.
- **Latihan 8.2:** Lakukan *fine-tuning* pada model ViLBERT untuk tugas pemberian teks pada gambar.
- **Latihan 8.3:** Bereksperimen dengan mekanisme atensi lintas modal dengan memvariasikan jumlah kepala atensi (*attention heads*).
- **Latihan 8.4:** Atasi masalah ketidakseimbangan modalitas menggunakan teknik augmentasi data dalam pipa pelatihan Anda.
- **Latihan 8.5:** Bangun kerangka kerja pembelajaran mandiri untuk melakukan penyelarasan lintas modal tanpa label eksplisit.