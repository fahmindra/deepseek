---
slug: lmvr-8
title: lmvr-8
---
# Bab 8: Multimodal Transformer dan Ekstensi

> **"Masa depan AI terletak pada pembelajaran multimodal, di mana mengintegrasikan informasi dari modalitas yang berbeda akan menghasilkan model yang lebih kaya dan sadar konteks yang dapat memahami dunia seperti cara manusia melakukannya." — Fei-Fei Li**

Bab 8 dari LMVR memberikan eksplorasi mendalam tentang Transformer multimodal dan ekstensinya, dengan fokus pada integrasi modalitas data yang berbeda seperti teks, gambar, dan audio. Bab ini dimulai dengan memperkenalkan dasar-dasar pembelajaran multimodal, menyoroti pentingnya menggabungkan berbagai sumber data untuk menciptakan representasi yang lebih kaya dan kuat. Kemudian dibahas mengenai adaptasi arsitektur Transformer untuk tugas-tugas multimodal, mendiskusikan teknik-teknik utama seperti self-attention, cross-attention, dan berbagai strategi fusi. Bab ini juga mencakup signifikansi pra-pelatihan dan penyetelan halus model multimodal, bersama dengan ekstensi tingkat lanjut seperti ViLBERT dan UNITER yang mendorong batas-batas kemampuan model-model ini. Terakhir, bab ini membahas tantangan dan arah masa depan dalam pembelajaran multimodal, menekankan potensi inovasi di bidang-bidang seperti layanan kesehatan, mengemudi otonom, dan interaksi manusia-komputer.

## 8.1. Pengantar Pembelajaran Multimodal

Pembelajaran multimodal adalah kerangka kerja yang kuat dalam pembelajaran mesin yang bertujuan untuk menyatukan informasi dari berbagai sumber data, seperti teks, gambar, dan audio. Dengan menggabungkan modalitas yang berbeda ini, model dapat menghasilkan representasi yang lebih komprehensif dan sadar konteks. Berbeda dengan pendekatan unimodal yang hanya menangkap satu sumber informasi, pembelajaran multimodal memanfaatkan kualitas unik dari setiap modalitas. Misalnya, data teks sering kali mengkodekan informasi sintaksis dan semantik secara berurutan, sementara gambar berisi petunjuk spasial dan visual yang disusun secara kontinu. Integrasi ini memungkinkan model untuk memanfaatkan kekuatan komplementer dari setiap jenis data, yang sangat bermanfaat dalam tugas-tugas seperti pemberian teks pada gambar (image captioning), analisis sentimen multimodal, dan menjawab pertanyaan visual (visual question answering). Namun, pembelajaran multimodal yang efektif memerlukan penyelesaian tantangan yang kompleks, termasuk penyelarasan dan fusi modalitas, yang harus menjembatani perbedaan inheren dalam struktur data dan semantik antar modalitas.

Sebagai contoh dalam pembelajaran multimodal berbasis graf, berbagai jenis data seperti gambar, teks, dan dataset ilmiah direpresentasikan sebagai node atau lapisan dalam struktur graf yang terpadu. Pendekatan ini memanfaatkan jaringan saraf graf untuk menangkap hubungan antar modalitas dengan menghubungkan node melalui tautan lintas-modal, memungkinkan representasi data kompleks yang holistik dan kaya konteks. Setiap modalitas menyumbangkan atribut uniknya—seperti informasi sekuensial dari teks, petunjuk spasial dari gambar, atau data relasional dari bidang ilmiah—memungkinkan peningkatan kinerja dalam tugas-tugas seperti prediksi, klasifikasi, dan penalaran interdisipliner di berbagai domain yang saling terhubung.

![Gambar 1: Ilustrasi paradigma pembelajaran multimodal.](https://lmvr.rantai.dev//images/L6A504iArDGBvUq8WSeO-5KSTrd65BW0ROkAZbFzn-v1.jpeg)
**Gambar 1:** Ilustrasi paradigma pembelajaran multimodal.

Dalam konteks model bahasa besar, salah satu teknik tingkat lanjut dalam pembelajaran multimodal adalah cross-attention, yang secara dinamis memprioritaskan informasi relevan dari setiap modalitas. Dalam cross-attention, mekanisme atensi belajar untuk fokus pada fitur utama dari satu modalitas yang relevan dengan modalitas lainnya. Misalnya, model yang dilatih untuk memproses data teks dan gambar dapat menggunakan cross-attention untuk mengidentifikasi wilayah gambar yang sesuai dengan frasa tertentu dalam teks. Secara matematis, misalkan $X_t$ dan $X_i$ masing-masing mewakili penyematan teks dan gambar. Cross-attention memungkinkan model untuk menghitung skor atensi antara kedua modalitas ini, menghasilkan representasi fusi $Z$ yang dipengaruhi oleh konteks dari sumber teks dan gambar. Proses ini direpresentasikan sebagai berikut:

$$ Z = \text{softmax} \left( \frac{Q_t K_i^T}{\sqrt{d_k}} \right) V_i + \text{softmax} \left( \frac{Q_i K_t^T}{\sqrt{d_k}} \right) V_t $$

di mana $Q_t$, $K_t$, dan $V_t$ adalah matriks query, key, dan value untuk modalitas teks, dan $Q_i$, $K_i$, dan $V_i$ adalah matriks query, key, dan value untuk modalitas gambar. Mekanisme cross-attention ini menghasilkan bobot yang menonjolkan interaksi lintas-modal, menghasilkan representasi yang menangkap isyarat kontekstual timbal balik.

![Gambar 2: Ilustrasi cross-attention (Ref: https://arxiv.org/html/2403.03431v1).](https://lmvr.rantai.dev//images/L6A504iArDGBvUq8WSeO-A0uRZjwcfvAAD225vJfy-v1.png)
**Gambar 2:** Ilustrasi cross-attention.

Dalam pembelajaran multimodal, cross-attention adalah teknik ampuh yang secara dinamis memprioritaskan informasi kunci di berbagai jenis data. Mekanisme ini mempelajari skor atensi antar modalitas, memungkinkan model untuk fokus pada fitur dalam satu modalitas yang relevan dengan modalitas lainnya. Misalnya, dalam mengemudi otonom, data visual dan sensor dapat difusikan, dengan setiap modalitas menginformasikan pemahaman situasional model. Formally, cross-attention menghitung representasi fusi, $Z$, menggunakan skor atensi yang diturunkan dari query, key, dan value dari setiap modalitas, menangkap isyarat kontekstual di seluruh modalitas dan menghasilkan penyematan bersama untuk integrasi yang lebih baik.

Implementasi Python berikut menunjukkan arsitektur transformer multimodal yang disederhanakan menggunakan PyTorch. Dalam model ini, setiap modalitas—teks dan gambar—memiliki encodernya sendiri yang memproses data input secara independen. Lapisan cross-attention kemudian menggabungkan representasi-representasi ini.

### Contoh Python: Transformer Multimodal Sederhana dengan Cross-Attention

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerBlock(nn.Module):
    def __init__(self, dim):
        super(TransformerBlock, self).__init__()
        self.attention = nn.Linear(dim, dim)
        self.feed_forward = nn.Linear(dim, dim)
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x):
        # Atensi sederhana (simulasi)
        attended = F.softmax(self.attention(x), dim=-1)
        ff_out = self.feed_forward(attended)
        return self.layer_norm(ff_out + x)

class MultimodalTransformer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MultimodalTransformer, self).__init__()
        self.text_encoder = TransformerBlock(input_dim)
        self.image_encoder = nn.Linear(input_dim, input_dim)
        self.cross_attention = TransformerBlock(input_dim)
        # Output layer untuk tensor yang digabungkan (concatenated)
        self.output_layer = nn.Linear(2 * input_dim, output_dim)

    def forward(self, text, image):
        # Encode teks dan gambar secara independen
        text_encoded = self.text_encoder(text)
        image_encoded = self.image_encoder(image)

        # Cross-attention sederhana: fusi representasi
        text_cross = self.cross_attention(text_encoded + image_encoded)
        image_cross = self.cross_attention(image_encoded + text_encoded)

        # Gabungkan dan proyeksikan ke output
        combined = torch.cat([text_cross, image_cross], dim=1)
        return self.output_layer(combined)

# Contoh penggunaan
input_dim = 512
output_dim = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MultimodalTransformer(input_dim, output_dim).to(device)
text_data = torch.randn(1, input_dim).to(device)
image_data = torch.randn(1, input_dim).to(device)

output = model(text_data, image_data)
print("Output Transformer Multimodal:", output.shape)
```

Mengevaluasi transformer multimodal ini terhadap model unimodal biasanya menunjukkan peningkatan kinerja yang signifikan. Kemampuan fusi ini sangat berharga dalam aplikasi seperti menjawab pertanyaan visual.

![Gambar 3: Ilustrasi arsitektur CLIP-BERT.](https://lmvr.rantai.dev//images/L6A504iArDGBvUq8WSeO-OlgzYreScElFkdDVZ8dY-v1.png)
**Gambar 3:** Ilustrasi arsitektur CLIP-BERT.

Kode berikut mengimplementasikan sistem untuk melakukan pencocokan gambar-teks multimodal menggunakan varian model CLIP dengan Python dan library `transformers`.

### Contoh Python: Pencocokan Gambar-Teks dengan CLIP

```python
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

def main():
    # Memuat model dan processor CLIP
    model_id = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_id)
    processor = CLIPProcessor.from_pretrained(model_id)

    # Contoh gambar dan teks
    image_paths = ["path/to/image1.jpg", "path/to/image2.jpg"] # Ganti dengan path asli
    # Untuk demo, kita asumsikan gambar dummy jika file tidak ada
    images = [Image.new('RGB', (224, 224), color='red') for _ in image_paths]
    
    texts = ["a cycling race", "a photo of two cats", "a robot holding a candle"]

    # Preprocessing
    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)

    # Inference
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Mendapatkan skor probabilitas gambar terhadap teks
    logits_per_image = outputs.logits_per_image # Skor kesamaan
    probs = logits_per_image.softmax(dim=1) # Probabilitas untuk setiap teks

    for i, img_path in enumerate(image_paths):
        print(f"\nHasil untuk gambar {i+1}:")
        for j, text in enumerate(texts):
            print(f"Probabilitas: {probs[i][j].item()*100:.2f}% | Teks: {text}")

if __name__ == "__main__":
    main()
```

Sebagai kesimpulan, adaptasi arsitektur Transformer untuk pembelajaran multimodal menawarkan kemampuan yang kuat untuk tugas-tugas yang memerlukan pemahaman terintegrasi di berbagai sumber data.

## 8.2. Arsitektur Transformer untuk Pembelajaran Multimodal

Dalam pembelajaran multimodal, arsitektur Transformer telah menjadi pusat karena kemampuannya untuk mengintegrasikan informasi dari berbagai aliran data. Kekuatan inti Transformer terletak pada mekanisme self-attention, yang secara dinamis menimbang relevansi token dalam suatu urutan. Untuk tugas multimodal, kemampuan self-attention ini dapat diadaptasi menjadi mekanisme cross-attention yang memungkinkan interaksi antara aliran data yang berbeda.

Dalam Transformer multimodal, lapisan self-attention beroperasi di dalam setiap modalitas untuk menangkap hubungan intra-modal, sementara lapisan cross-attention diperkenalkan untuk mengintegrasikan dan menyelaraskan data antar modalitas.

Beberapa model Transformer multimodal mengilustrasikan efektivitas mekanisme ini. ViLBERT (Vision-and-Language BERT) menggunakan dua aliran atensi terpisah yang dihubungkan oleh cross-attention. UNITER (UNiversal Image-TExt Representation) menggunakan arsitektur aliran tunggal untuk mengkodekan gambar dan teks secara bersama-sama.

![Gambar 4: Ilustrasi arsitektur ViLBERT.](https://lmvr.rantai.dev//images/L6A504iArDGBvUq8WSeO-CBmyqXIKs5pWuLl2CSTb-v1.png)
**Gambar 4:** Ilustrasi arsitektur ViLBERT.

Berikut adalah kerangka kerja Python sederhana untuk melatih model multimodal dengan data simulasi.

### Contoh Python: UNITER Sederhana (Pelatihan Simulasi)

```python
import torch
import torch.nn as nn
import torch.optim as optim

class SimpleUNITER(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.img_projection = nn.Linear(embed_dim, hidden_dim)
        self.text_projection = nn.Linear(embed_dim, hidden_dim)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, batch_first=True),
            num_layers=4
        )
        self.classifier = nn.Linear(hidden_dim, 2)

    def forward(self, img_feat, text_feat):
        # Proyeksi ke ruang penyematan bersama
        img_proj = self.img_projection(img_feat)
        text_proj = self.text_projection(text_feat)
        
        # Gabungkan token gambar dan teks (Concatenate pada dimensi sequence)
        joint_input = torch.cat([img_proj, text_proj], dim=1) # (batch, seq_len, hidden_dim)
        
        encoded = self.encoder(joint_input)
        # Ambil representasi rata-rata untuk klasifikasi
        return self.classifier(encoded.mean(dim=1))

# Setup Pelatihan
model = SimpleUNITER(768, 512)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# Simulasi data
img_batch = torch.randn(8, 10, 768) # 10 wilayah gambar
text_batch = torch.randn(8, 20, 768) # 20 token teks
labels = torch.randint(0, 2, (8,))

# Training step
optimizer.zero_grad()
output = model(img_batch, text_batch)
loss = criterion(output, labels)
loss.backward()
optimizer.step()
print(f"Loss pelatihan: {loss.item():.4f}")
```

![Gambar 5: Ilustrasi arsitektur UNITER.](https://lmvr.rantai.dev//images/L6A504iArDGBvUq8WSeO-4UBTQx3BWB7WUPIYQcs7-v1.jpeg)
**Gambar 5:** Ilustrasi arsitektur UNITER.

Penyelarasan lintas-modal sangat penting dalam arsitektur ini, karena memungkinkan Transformer untuk fokus secara selektif pada fitur yang paling relevan di seluruh modalitas. Pra-pelatihan pada dataset multimodal besar dan beragam seperti Conceptual Captions atau Visual Genome sangat penting bagi model untuk mempelajari representasi multimodal yang dapat digeneralisasi.

## 8.3. Teknik Fusi Multimodal

Dalam pembelajaran multimodal, teknik fusi memainkan peran kritis dalam menggabungkan informasi, yang mencakup fusi awal (early fusion), fusi akhir (late fusion), dan fusi hibrida (hybrid fusion).

![Gambar 6: Ilustrasi teknik fusi multimodal.](https://lmvr.rantai.dev//images/L6A504iArDGBvUq8WSeO-QJhPOg69iY7fmjV5m3IY-v1.png)
**Gambar 6:** Ilustrasi teknik fusi multimodal.

1.  **Early Fusion:** Menggabungkan data mentah atau fitur pada tingkat input. Memungkinkan model mempelajari interaksi lintas-modal sejak awal tetapi secara komputasi bisa sangat berat.
2.  **Late Fusion:** Memproses setiap modalitas secara independen melalui encoder masing-masing dan menggabungkan outputnya hanya di tahap akhir. Baik untuk menangkap informasi tingkat tinggi dari setiap sumber.
3.  **Hybrid Fusion:** Menggabungkan elemen fusi awal dan akhir, mengintegrasikan aspek tertentu lebih awal dan fitur lainnya lebih lambat untuk menyeimbangkan kekuatan masing-masing.

### Contoh Python: Perbandingan Model Fusi

```python
import torch
import torch.nn as nn

class EarlyFusion(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim * 2, out_dim)

    def forward(self, text, img):
        combined = torch.cat([text, img], dim=-1)
        return self.fc(combined)

class LateFusion(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.text_net = nn.Linear(in_dim, out_dim)
        self.img_net = nn.Linear(in_dim, out_dim)
        self.final = nn.Linear(out_dim * 2, out_dim)

    def forward(self, text, img):
        t_feat = torch.relu(self.text_net(text))
        i_feat = torch.relu(self.img_net(img))
        return self.final(torch.cat([t_feat, i_feat], dim=-1))

# Dummy inputs
text_emb = torch.randn(1, 128)
img_emb = torch.randn(1, 128)

early_model = EarlyFusion(128, 64)
late_model = LateFusion(128, 64)

print("Output Early Fusion:", early_model(text_emb, img_emb).shape)
print("Output Late Fusion:", late_model(text_emb, img_emb).shape)
```

## 8.4. Pra-Pelatihan dan Penyetelan Halus Multimodal Transformer

Pra-pelatihan memungkinkan Transformer multimodal mengembangkan representasi lintas-modal dengan mengeksposnya pada dataset besar. Tugas pra-pelatihan umum meliputi Masked Language Modeling (MLM) yang diadaptasi untuk multimodal dan Image-Text Matching (ITM).

Penyetelan halus (fine-tuning) menyesuaikan model untuk memenuhi persyaratan aplikasi target. Tantangan utamanya adalah menghindari overfitting melalui teknik seperti pembekuan selektif (*selective freezing*) pada lapisan tertentu atau augmentasi data multimodal.

## 8.5. Ekstensi dan Aplikasi Multimodal Transformer

Transformer multimodal telah menjadi sangat penting dalam memajukan kemampuan AI di dunia nyata. Ekstensi tingkat lanjut seperti Multimodal BERT (MMBERT) diimplementasikan dengan encoder paralel yang berinteraksi melalui cross-attention.

![Gambar 7: Ilustrasi arsitektur multimodal BERT (MMBERT) untuk gambar medis.](https://lmvr.rantai.dev//images/L6A504iArDGBvUq8WSeO-hnorzf3YOOqcMZi7EqNq-v1.jpeg)
**Gambar 7:** Ilustrasi arsitektur multimodal BERT (MMBERT) untuk gambar medis.

Dalam layanan kesehatan, model ini menggabungkan pencitraan medis dengan data pasien berbasis teks untuk diagnostik. Dalam mengemudi otonom, mereka berkontribusi pada fusi sensor (LIDAR, radar, kamera). Dalam hiburan, analisis konten video menggabungkan bingkai video, audio, dan subtitle untuk ringkasan otomatis.

### Contoh Python: MMBERT Sederhana untuk Klasifikasi Medis (Konsep)

```python
class MMBERTMedical(nn.Module):
    def __init__(self, text_dim, hidden_dim):
        super().__init__()
        # CNN sederhana untuk gambar X-ray
        self.img_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 222 * 222, hidden_dim) # Tergantung input size
        )
        self.text_encoder = nn.Linear(text_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim * 2, 2)

    def forward(self, text, x_ray):
        t_feat = self.text_encoder(text)
        i_feat = self.img_encoder(x_ray)
        combined = torch.cat([t_feat, i_feat], dim=-1)
        return self.classifier(combined)
```

## 8.6. Tantangan dan Arah Masa Depan

Terlepas dari kemajuannya, pembelajaran multimodal menghadapi tantangan seperti kelangkaan data (terutama di bidang medis) dan ketidakseimbangan modalitas. Evaluasi memerlukan metrik baru yang melampaui akurasi unimodal sederhana.

Tren yang muncul mengarah pada penggunaan pembelajaran mandiri (self-supervised learning) untuk memanfaatkan data tak berlabel dalam jumlah besar. MobileCLIP adalah contoh kemajuan dalam menciptakan model multimodal yang efisien untuk perangkat seluler dan edge.

![Gambar 8: Perbandingan antara model GAN, VAE, dan Difusi.](https://lmvr.rantai.dev//images/L6A504iArDGBvUq8WSeO-u5mui8fwOw66TBwzmAJD-v1.webp)
**Gambar 8:** Perbandingan antara model GAN, VAE, dan Difusi.

Aplikasi di masa depan akan meluas ke robotika, realitas tertambah (AR), dan AI yang dipersonalisasi, di mana sinkronisasi fitur spasial dan temporal sangatlah kritis.

## 8.7. Kesimpulan

Bab 8 membekali pembaca dengan pemahaman komprehensif tentang Transformer multimodal dan ekstensinya. Dengan menguasai konsep-konsep ini, pembaca akan siap untuk mengembangkan sistem AI canggih yang mampu memproses dan mengintegrasikan berbagai modalitas data, membuka jalan bagi solusi inovatif di berbagai industri.

### 8.7.1. Pembelajaran Lebih Lanjut dengan GenAI

- Jelajahi bagaimana model multimodal mengintegrasikan teks, gambar, dan audio.
- Bedah teknik canggih untuk mengatasi noise spesifik modalitas dan penyelarasan semantik.
- Analisis strategi fusi untuk data asinkron atau tidak terstruktur.
- Selidiki peran mekanisme co-attention dalam model dual-stream.
- Bandingkan fitur unik ViLBERT, UNITER, dan MMBERT.
- Diskusikan pentingnya pra-pelatihan pada dataset multimodal yang masif.
- Jelajahi teknik fine-tuning seperti selective freezing dan adapter layers.
- Analisis implikasi etis (privasi, bias) dalam penyebaran model multimodal di domain sensitif.
- Bagaimana self-supervised learning dapat membantu dalam skenario kelangkaan label?
- Diskusikan pengembangan metrik evaluasi baru untuk kualitas fusi multimodal.

### 8.7.2. Latihan Praktis

#### Latihan Mandiri 8.1: Implementasi Teknik Fusi Multimodal
Tujuan: Membandingkan efektivitas fusi awal, akhir, dan hibrida pada satu tugas klasifikasi tertentu.

#### Latihan Mandiri 8.2: Penyetelan Halus Model Multimodal Pra-Terlatih
Tujuan: Melatih model seperti CLIP atau ViLBERT pada dataset khusus (misal: gambar medis dan laporannya).

#### Latihan Mandiri 8.3: Mekanisme Cross-Modal Attention
Tujuan: Mengimplementasikan lapisan atensi di mana query berasal dari teks dan key/value berasal dari gambar.

#### Latihan Mandiri 8.4: Menangani Ketidakseimbangan Modalitas
Tujuan: Menggunakan teknik augmentasi atau pembobotan khusus untuk kasus di mana satu modalitas memiliki data jauh lebih sedikit.

#### Latihan Mandiri 8.5: Evaluasi Pembelajaran Mandiri Multimodal
Tujuan: Membangun pipeline pelatihan menggunakan fungsi kerugian kontrasif (contrastive loss) tanpa label eksplisit.