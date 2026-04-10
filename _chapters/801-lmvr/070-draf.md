---
slug: lmvr-7
title: lmvr-7
---
# Bab 7: Pembelajaran Multi-tugas: T5 dan Model Terpadu

> **"Pembelajaran multi-tugas adalah paradigma kuat yang memanfaatkan representasi bersama untuk memungkinkan model berkinerja baik di berbagai tugas, sering kali melampaui kinerja model khusus tugas." — Andrew Ng**

Bab 7 dari LMVR menggali ranah pembelajaran multi-tugas, menyoroti arsitektur dan kemampuan model seperti T5 dan model terpadu lainnya. Bab ini dimulai dengan menjelaskan dasar-dasar pembelajaran multi-tugas dan keunggulannya dibandingkan pendekatan tugas tunggal, dengan menekankan pentingnya representasi bersama di seluruh tugas. Kemudian, bab ini mengeksplorasi arsitektur T5, yang membingkai setiap masalah NLP sebagai tugas teks-ke-teks, memberikan pendekatan terpadu untuk menyelesaikan berbagai tugas dalam satu model. Bab ini juga mencakup tantangan dan strategi untuk penyetelan halus model-model ini untuk aplikasi spesifik, metode untuk mengevaluasi model multi-tugas di berbagai tugas, dan teknik untuk penskalaan serta optimalisasi model-model ini untuk penerapan di dunia nyata. Terakhir, bab ini melihat ke masa depan pembelajaran multi-tugas, membahas tren seperti pembelajaran multimodal dan pembelajaran berkelanjutan, serta implikasinya bagi evolusi AI.

## 7.1. Pengantar Pembelajaran Multi-tugas

Pembelajaran multi-tugas (Multitask Learning - MTL) adalah pendekatan dalam pembelajaran mesin di mana sebuah model dilatih untuk melakukan beberapa tugas secara bersamaan, memanfaatkan representasi bersama di seluruh tugas untuk meningkatkan generalisasi dan efisiensi. Berbeda dengan pembelajaran tugas tunggal, di mana model dilatih pada tugas tertentu secara terisolasi, pembelajaran multi-tugas bertujuan untuk meningkatkan kinerja dengan menangkap pola dasar yang berguna di berbagai tugas. Ide kuncinya adalah bahwa tugas-tugas dapat berbagi pengetahuan dan representasi, memungkinkan model untuk menggeneralisasi lebih baik dan menghindari overfitting pada tugas individu dengan menggunakan data dari semua tugas untuk memperhalus pemahamannya.

Dalam pembelajaran multi-tugas, pertimbangkan contoh analisis sentimen dan klasifikasi topik, di mana sebuah model dilatih pada ulasan pelanggan untuk analisis sentimen (positif, negatif, netral) dan pada artikel berita untuk klasifikasi topik (seperti politik, olahraga, dan teknologi). Dengan berbagi parameter pada lapisan pemrosesan teks awal, model mempelajari pola bahasa umum yang menguntungkan kedua tugas tersebut. Analisis sentimen dapat memperoleh manfaat dari konteks topik, karena sentimen sering kali bervariasi di berbagai topik, sementara klasifikasi topik meningkat dengan memahami fitur bahasa yang terkait dengan sentimen. Demikian pula, dalam contoh visi komputer, melatih model pada klasifikasi gambar (di mana setiap gambar memiliki satu label, seperti "kucing" atau "anjing") dan deteksi objek (mengidentifikasi banyak objek, seperti mendeteksi "kucing" dan "anjing" dalam gambar yang sama) memungkinkan lapisan konvolusional bersama untuk menangkap fitur seperti tepi dan bentuk, yang membantu dalam mengidentifikasi objek tunggal maupun mendeteksi banyak objek dalam gambar. Contoh lain dapat melibatkan prediksi harga rumah dan tingkat kejahatan, di mana fitur lingkungan seperti tingkat pendapatan, kedekatan dengan sekolah, dan fasilitas lokal berdampak pada kedua hasil tersebut. Berbagi parameter pada lapisan awal memungkinkan model untuk mempelajari karakteristik lingkungan yang meningkatkan prediksi untuk harga rumah dan tingkat kejahatan. Dalam setiap contoh, parameter bersama memungkinkan model untuk menangkap pola yang bermanfaat di seluruh tugas, mencegah overfitting pada satu tugas saja dan meningkatkan generalisasi, yang merupakan esensi dari pembelajaran multi-tugas.

![Gambar 1: Ilustrasi paradigma Pembelajaran Multi-tugas.](https://lmvr.rantai.dev//images/L6A504iArDGBvUq8WSeO-oc7p9ePlHw2lpWAldyqe-v1.webp)
**Gambar 1:** Ilustrasi paradigma Pembelajaran Multi-tugas.

Konsep Pembelajaran Multi-tugas (MTL) pertama kali diusulkan oleh Rich Caruana pada tahun 1997 sebagai pendekatan inovatif dalam pembelajaran mesin, di mana satu model dilatih untuk melakukan beberapa tugas secara bersamaan dengan berbagi representasi di seluruh tugas. Caruana memperkenalkan MTL dengan ide bahwa berbagi parameter di antara tugas-tugas memungkinkan model untuk menangkap struktur dasar yang umum bagi tugas-tugas tersebut, sehingga meningkatkan generalisasi dan membantu mencegah overfitting. Karya Caruana meletakkan dasar bagi MTL dengan menunjukkan bagaimana mempelajari beberapa tugas terkait secara bersamaan dapat membuat setiap tugas lebih efisien dengan memanfaatkan informasi bersama, sebuah penyimpangan signifikan dari pembelajaran tugas tunggal.

Sejak proposal Caruana, MTL telah berkembang melalui serangkaian kemajuan, terutama di bidang regularisasi, dekomposisi, percabangan, propagasi, optimasi, dan unifikasi. Pada awal 2000-an, para peneliti fokus pada pengembangan teknik regularisasi yang lebih baik untuk mengontrol bagaimana representasi bersama memengaruhi tugas individu, memungkinkan keseimbangan yang lebih baik antara mempelajari pola bersama dan detail khusus tugas. Teknik dekomposisi muncul, memungkinkan model untuk memisahkan pengetahuan umum dan khusus tugas dengan berbagi parameter secara selektif dan secara efektif menguraikan fitur-fitur yang hanya berguna untuk tugas-tugas tertentu. Arsitektur percabangan lebih lanjut menyempurnakan ini dengan menggabungkan lapisan khusus tugas pada tahap akhir dalam jaringan, memungkinkan tugas yang berbeda untuk menyimpang dengan cara yang meningkatkan kinerja pada subtugas khusus sambil tetap menggunakan representasi bersama di lapisan awal.

![Gambar 2: Perjalanan sejarah paradigma Pembelajaran Multi-tugas.](https://lmvr.rantai.dev//images/L6A504iArDGBvUq8WSeO-99ZJORzDR0obSWXELWUA-v1.png)
**Gambar 2:** Perjalanan sejarah paradigma Pembelajaran Multi-tugas.

Pada tahun 2010-an, metode propagasi menjadi bagian integral dari MTL, memungkinkan cara yang lebih efisien untuk berbagi informasi di seluruh tugas. Para peneliti mengeksplorasi teknik seperti jaringan lintas-jahit (cross-stitch networks) dan komputasi kondisional, di mana hanya fitur yang relevan yang dibagikan antara tugas-tugas tertentu, meningkatkan generalisasi dengan memungkinkan pengetahuan untuk merambat secara selektif daripada secara seragam di semua tugas. Strategi optimasi untuk MTL juga sangat krusial dalam periode ini, khususnya untuk merancang metode berbasis gradien yang dapat menangani interaksi kompleks antar tugas. Teknik seperti pembobotan tugas dikembangkan untuk menyesuaikan penekanan pembelajaran pada setiap tugas secara dinamis, memastikan tidak ada tugas yang mendominasi atau menghambat kemajuan tugas lainnya.

Mendekati tahun 2022, penelitian MTL beralih ke arah penyatuan berbagai pendekatan untuk mengembangkan arsitektur fleksibel yang mampu beradaptasi secara dinamis terhadap banyak tugas, bahkan ketika kompleksitas dan keragaman tugas meningkat. Unifikasi melibatkan pencampuran berbagai strategi, seperti menggabungkan teknik regularisasi, dekomposisi, dan propagasi dalam satu kerangka kerja model yang dapat mengakomodasi berbagai tugas dengan penyetelan halus minimal. Era MTL ini membawa kerangka kerja yang memungkinkan model untuk mengalokasikan sumber daya berdasarkan kebutuhan tugas secara dinamis, membuat MTL lebih tangguh dan mudah beradaptasi di berbagai domain, dari pemrosesan bahasa alami hingga visi komputer. Kemajuan ini telah mendefinisikan ulang MTL, membangun visi Caruana dengan menciptakan model yang memanfaatkan informasi bersama untuk meningkatkan efisiensi dan kemampuan beradaptasi dalam lingkungan multi-tugas yang kompleks.

Secara matematis, pembelajaran multi-tugas dapat dibingkai sebagai masalah optimasi di mana model meminimalkan kombinasi berbobot dari fungsi kerugian di beberapa tugas. Untuk sekumpulan tugas $T_1, T_2, \dots, T_n$, dengan fungsi kerugian yang sesuai $\mathcal{L}_1, \mathcal{L}_2, \dots, \mathcal{L}_n$, fungsi kerugian keseluruhan dalam pembelajaran multi-tugas dapat dinyatakan sebagai:

$$ \mathcal{L}_{\text{MTL}} = \sum_{i=1}^{n} \lambda_i \mathcal{L}_i, $$

di mana $\lambda_i$ adalah bobot yang mengontrol kepentingan relatif dari kerugian setiap tugas. Model harus menyeimbangkan meminimalkan kerugian ini dengan cara yang meningkatkan kinerja di semua tugas, sambil juga memastikan bahwa tidak ada tugas tunggal yang mendominasi proses pembelajaran.

Perbedaan antara pembelajaran tugas tunggal dan pembelajaran multi-tugas terletak pada struktur model dan bagaimana informasi dibagikan di seluruh tugas. Dalam MTL, model dibagi menjadi lapisan bersama (shared layers), yang mempelajari representasi umum untuk semua tugas, dan lapisan khusus tugas (task-specific layers). Struktur ini dapat dinyatakan sebagai:

$$ y_i = f_{\text{task}_i}(f_{\text{shared}}(x)), $$

di mana $x$ mewakili input, $f_{\text{shared}}$ mewakili lapisan bersama, dan $f_{\text{task}_i}$ mewakili lapisan khusus tugas untuk tugas $i$.

![Gambar 3: Ilustrasi sistem Tanya Jawab (Q&A) sadar konteks.](https://lmvr.rantai.dev//images/L6A504iArDGBvUq8WSeO-44I42HWGcG6iJYnlehgJ-v1.png)
**Gambar 3:** Ilustrasi sistem Tanya Jawab (Q&A) sadar konteks.

Dalam arsitektur MTL untuk tugas Tanya Jawab (Q&A) dan Pembelajaran Konteks, model dimulai dengan pengkodean independen. Lapisan pengkodean penyelarasan kemudian mengikuti untuk menangkap asosiasi yang menghubungkan kedua tugas. Dalam lapisan atensi-bersama ganda (dual co-attention), model melakukan atensi silang antara pertanyaan dan konteks. Akhirnya, Feed-Forward Network (FFN) dan multi-head attention menggabungkan atensi ini untuk menghasilkan jawaban yang tepat.

![Gambar 4: Contoh arsitektur model MTL berbasis ELECTRA.](https://lmvr.rantai.dev//images/L6A504iArDGBvUq8WSeO-RnIsJrNtNLRKrSAiCzav-v1.png)
**Gambar 4:** Contoh arsitektur model MTL berbasis ELECTRA.

Dalam arsitektur MTL berbasis ELECTRA, representasi token input mengikuti jalur pemrosesan ganda melalui generator, diskriminator, dan lapisan khusus tugas. Diskriminator mengevaluasi kemungkinan setiap token sebagai "asli" atau "palsu" berdasarkan konteks. Representasi yang dihasilkan kemudian dilewatkan ke lapisan model bahasa khusus tugas seperti pemberian skor Part-of-Speech (POS), Dependency Head, Constituent Span, dan Semantic Role.

Berikut adalah contoh penggunaan Python untuk melakukan peringkasan teks menggunakan model T5 melalui library `transformers`.

### Contoh Python: Peringkasan Teks dengan T5

```python
from transformers import T5Tokenizer, T5ForConditionalGeneration

def summarize_text(text):
    # Menggunakan model t5-small untuk efisiensi
    model_name = "t5-small"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    # T5 memerlukan prefix tugas dalam inputnya
    input_text = "summarize: " + text
    
    # Tokenisasi dan pengkodean
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    
    # Menghasilkan ringkasan
    summary_ids = model.generate(
        inputs, 
        max_length=150, 
        min_length=40, 
        length_penalty=2.0, 
        num_beams=4, 
        early_stopping=True
    )
    
    # Mendekode hasil
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Contoh teks (artikel tentang exoplanet)
article = ("In findings published Tuesday in Cornell University's arXiv by a team of scientists "
           "from the University of Montreal... water vapour was confirmed in the atmosphere of K2-18b.")

print("Ringkasan:")
print(summarize_text(article))
```

Berikut adalah contoh implementasi Python untuk tugas terjemahan multibahasa menggunakan T5.

### Contoh Python: Terjemahan Multibahasa dengan T5

```python
from transformers import T5Tokenizer, T5ForConditionalGeneration

def translate_t5(text, target_lang="German"):
    model_name = "t5-base"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    # Format input T5: "translate English to [TargetLang]: [Text]"
    input_text = f"translate English to {target_lang}: {text}"
    
    inputs = tokenizer.encode(input_text, return_tensors="pt")
    outputs = model.generate(inputs, max_length=100)
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

source_sentence = "This sentence will be translated into multiple languages."
print(f"German: {translate_t5(source_sentence, 'German')}")
print(f"French: {translate_t5(source_sentence, 'French')}")
```

Keluarga model T5 di Hugging Face menawarkan berbagai varian mulai dari `t5-small` (60 juta parameter) hingga `t5-11b` (11 miliar parameter). Seri Flan-T5 memperkenalkan pendekatan yang disetel dengan instruksi, membuatnya lebih baik dalam mengikuti perintah pengguna. MT5 adalah versi multibahasa yang mendukung berbagai bahasa secara efektif.

Berikut adalah cara mengunduh berbagai varian model T5 menggunakan library Python `huggingface_hub`.

### Contoh Python: Mengunduh Varian T5 dengan huggingface_hub

```python
from huggingface_hub import snapshot_download
import os

models = ["t5-small", "t5-base"]

for model_id in models:
    print(f"Mengunduh model {model_id}...")
    local_dir = os.path.join("models", model_id)
    # snapshot_download akan mengunduh seluruh repositori model
    snapshot_download(repo_id=model_id, local_dir=local_dir)
    print(f"Model {model_id} disimpan di {local_dir}")
```

## 7.2. Arsitektur T5

Arsitektur T5 (Text-To-Text Transfer Transformer) mewakili inovasi signifikan dalam pembelajaran multi-tugas. T5 membingkai setiap tugas NLP dalam format teks-ke-teks yang sama. Baik input maupun output diperlakukan sebagai string teks, menyederhanakan proses pelatihan karena menggunakan satu arsitektur terpadu.

![Gambar 5: Ilustrasi arsitektur model encoder-decoder T5.](https://lmvr.rantai.dev//images/L6A504iArDGBvUq8WSeO-aZQSQalpuRrifpo7DtTj-v1.png)
**Gambar 5:** Ilustrasi arsitektur model encoder-decoder T5.

Secara matematis, T5 mempelajari distribusi probabilitas bersyarat:
$$ P(Y | X) = \prod_{t=1}^{T} P(y_t | y_1, y_2, \dots, y_{t-1}, X) $$

Pra-pelatihan T5 menggunakan tugas "span corruption" di mana model memprediksi token yang disamarkan dalam sebuah urutan. Loss function pra-pelatihan adalah:
$$ \mathcal{L}_{\text{pretrain}} = - \sum_{i=1}^{N} \log P(y_i | X_{\setminus i}) $$

Berikut adalah implementasi blok Encoder-Decoder T5 yang disederhanakan menggunakan PyTorch.

### Contoh Python: Arsitektur Blok Encoder T5 (PyTorch)

```python
import torch
import torch.nn as nn

class T5EncoderBlock(nn.Module):
    def __init__(self, n_embd, n_heads):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(n_embd, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(n_embd)
        self.ff = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd)
        )
        self.norm2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # Atensi dengan koneksi residual dan normalisasi
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + attn_out)
        
        # Feed-forward dengan koneksi residual dan normalisasi
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x

class T5Encoder(nn.Module):
    def __init__(self, vocab_size, n_embd, n_layers, n_heads):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, n_embd)
        self.blocks = nn.ModuleList([T5EncoderBlock(n_embd, n_heads) for _ in range(n_layers)])

    def forward(self, x):
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x)
        return x

# Contoh Penggunaan
# encoder = T5Encoder(vocab_size=32000, n_embd=512, n_layers=6, n_heads=8)
```

## 7.3. Model Terpadu untuk Pembelajaran Multi-tugas

Model terpadu bertujuan untuk merancang satu arsitektur model yang dapat menangani berbagai tugas tanpa komponen khusus tugas. BART (Bidirectional and Auto-Regressive Transformers) adalah contoh lain yang menggabungkan pengkodean dua arah (seperti BERT) dan decoding autoregresif (seperti GPT).

Salah satu tantangan utama dalam model terpadu adalah interferensi tugas (*task interference*). Jika tugas terlalu berbeda, lapisan bersama mungkin kesulitan untuk mewakili semua tugas secara efektif. Solusinya dapat berupa arsitektur modular di mana komponen bersama digunakan di semua tugas, tetapi modul tertentu (seperti decoder yang berbeda) diaktifkan tergantung pada tugasnya.

Teknik pelatihan yang hemat parameter seperti LoRA (Low-Rank Adaptation) memungkinkan penyetelan halus model besar seperti T5 atau BART secara efisien dengan memfaktorkan matriks pembaruan bobot menjadi matriks peringkat rendah:
$$ W = W_0 + \Delta W = W_0 + A B $$

Berikut adalah implementasi Python untuk peringkasan menggunakan model BART.

### Contoh Python: Peringkasan dengan BART

```python
from transformers import pipeline

def summarize_with_bart(text):
    # Menggunakan model DistilBART untuk peringkasan yang lebih cepat
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    
    summary = summarizer(text, max_length=130, min_length=30, do_sample=False)
    return summary[0]['summary_text']

# Masukkan teks panjang untuk diringkas
text_to_summarize = article # Dari contoh sebelumnya
print(summarize_with_bart(text_to_summarize))
```

Untuk terjemahan banyak-ke-banyak (many-to-many), model MBART-50 sangat mahir dalam menerjemahkan di antara puluhan pasangan bahasa.

### Contoh Python: Terjemahan Multibahasa dengan MBART-50

```python
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

def translate_mbart(text, src_lang="en_XX", tgt_lang="id_ID"):
    model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

    tokenizer.src_lang = src_lang
    encoded_input = tokenizer(text, return_tensors="pt")
    
    generated_tokens = model.generate(
        **encoded_input,
        forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang]
    )
    
    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

text = "This sentence will be translated into multiple languages."
print(f"Indonesian: {translate_mbart(text, 'en_XX', 'id_ID')}")
print(f"French: {translate_mbart(text, 'en_XX', 'fr_XX')}")
```

## 7.4. Penyetelan Halus Model Multi-tugas untuk Aplikasi Spesifik

Proses penyetelan halus (fine-tuning) melibatkan minimalisasi fungsi kerugian pada dataset baru. Masalah utama yang sering muncul adalah lupa katastrofik (*catastrophic forgetting*), di mana model kehilangan kemampuan pada tugas sebelumnya. Teknik seperti Elastic Weight Consolidation (EWC) membantu memitigasi ini dengan menambahkan penalti untuk perubahan pada bobot yang penting bagi tugas lama:

$$ \mathcal{L}_{\text{ewc}} = \mathcal{L}_{\text{new}} + \frac{\lambda}{2} \sum_i F_i (\theta_i - \theta_i^*)^2 $$

Berikut adalah kerangka kerja Python untuk melatih model dengan LoRA menggunakan library `peft`.

### Contoh Python: Penyetelan Halus T5 dengan LoRA

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer
from peft import LoraConfig, get_peft_model, TaskType

def setup_lora_t5():
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    
    # Konfigurasi LoRA
    lora_config = LoraConfig(
        r=8, 
        lora_alpha=32, 
        target_modules=["q", "v"], 
        lora_dropout=0.05, 
        bias="none", 
        task_type=TaskType.SEQ_2_SEQ_LM
    )
    
    # Terapkan LoRA ke model
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()
    return peft_model

# peft_model sekarang siap dilatih menggunakan training loop standar PyTorch
```

## 7.5. Mengevaluasi dan Benchmarking Model MTL

Mengevaluasi model multi-tugas memerlukan metrik gabungan. Kinerja keseluruhan model dapat dinyatakan sebagai:
$$ \mathcal{M}_{\text{overall}} = \sum_{i=1}^{n} w_i \mathcal{M}_i $$

di mana $w_i$ adalah bobot kepentingan tugas dan $\mathcal{M}_i$ adalah metrik tugas spesifik (seperti BLEU untuk terjemahan atau ROUGE untuk peringkasan).

Berikut adalah contoh cara menghitung berbagai metrik evaluasi dalam Python.

### Contoh Python: Menghitung Metrik Evaluasi (Accuracy, BLEU, ROUGE)

```python
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
import numpy as np

def evaluate_performance(predictions, references):
    # Akurasi sederhana (pencocokan string)
    accuracy = np.mean([p == r for p, r in zip(predictions, references)])
    
    # BLEU (untuk terjemahan)
    bleu_scores = [sentence_bleu([r.split()], p.split()) for p, r in zip(predictions, references)]
    avg_bleu = np.mean(bleu_scores)
    
    # ROUGE (untuk peringkasan)
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_scores = [scorer.score(r, p)['rougeL'].fmeasure for p, r in zip(predictions, references)]
    avg_rouge = np.mean(rouge_scores)
    
    return {"accuracy": accuracy, "bleu": avg_bleu, "rougeL": avg_rouge}

preds = ["the cat is on the mat", "hello world"]
refs = ["the cat sat on the mat", "hello world"]
print(evaluate_performance(preds, refs))
```

## 7.6. Penskalaan dan Optimasi Model MTL

Penskalaan model MTL memperkenalkan tantangan biaya komputasi. Teknik seperti kuantisasi (quantization) mengurangi presisi parameter dari FP32 ke INT8 untuk menghemat memori. Di Python, kita bisa menggunakan `bitsandbytes` untuk memuat model dalam mode 8-bit atau 4-bit.

### Contoh Python: Memuat T5 Terkuantisasi (8-bit)

```python
from transformers import T5ForConditionalGeneration, AutoTokenizer

model_id = "google/flan-t5-large"

# Memerlukan library bitsandbytes dan accelerate
# model = T5ForConditionalGeneration.from_pretrained(
#     model_id, 
#     device_map="auto", 
#     load_in_8bit=True
# )

# tokenizer = AutoTokenizer.from_pretrained(model_id)
# print("Model berhasil dimuat dalam mode 8-bit")
```

## 7.7. Arah Masa Depan dalam MTL dan Model Terpadu

Tren saat ini mengarah pada model multimodal yang mengintegrasikan berbagai tipe data (teks, gambar, audio). Pembelajaran berkelanjutan (*continual learning*) juga menjadi fokus agar model dapat beradaptasi dengan data baru tanpa perlu pelatihan ulang dari nol.

Berikut adalah logika dasar implementasi EWC (Elastic Weight Consolidation) untuk mencegah lupa katastrofik saat berpindah dari Tugas A ke Tugas B.

### Contoh Python: Logika EWC (Konsep)

```python
import torch

def calculate_fisher_information(model, task_a_data):
    fisher_info = {}
    model.eval()
    for x, y in task_a_data:
        model.zero_grad()
        output = model(x)
        loss = torch.nn.functional.cross_entropy(output, y)
        loss.backward()
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                # Fisher Info adalah kuadrat dari gradien
                fisher_info[name] = fisher_info.get(name, 0) + param.grad.data.pow(2)
    return fisher_info

def ewc_loss(model, current_loss, fisher_info, optimal_params_a, ewc_lambda=0.4):
    penalty = 0
    for name, param in model.named_parameters():
        if name in fisher_info:
            # Penalti untuk perubahan pada parameter yang penting bagi Tugas A
            penalty += (fisher_info[name] * (param - optimal_params_a[name]).pow(2)).sum()
    return current_loss + (ewc_lambda / 2) * penalty
```

## 7.8. Kesimpulan

Bab 7 memberikan eksplorasi menyeluruh tentang pembelajaran multi-tugas dan model terpadu, menawarkan wawasan tentang bagaimana pendekatan ini dapat meningkatkan kinerja dan efisiensi model NLP. Dengan menguasai konsep-konsep ini, pembaca akan diperlengkapi untuk mengembangkan model canggih yang mampu menangani berbagai tugas secara bersamaan.

### 7.8.1. Pembelajaran Lebih Lanjut dengan GenAI

- Jelaskan konsep dasar pembelajaran multi-tugas dan perbedaannya dengan tugas tunggal.
- Deskripsikan arsitektur T5 dan bagaimana ia membingkai tugas sebagai teks-ke-teks.
- Diskusikan trade-off antara MTL dan spesialisasi tugas.
- Jelajahi peran lapisan bersama dan khusus tugas.
- Analisis tantangan penyetelan halus T5 pada tugas khusus.
- Bandingkan T5 dengan model terpadu lainnya seperti BART.
- Apa strategi terbaik untuk mencegah lupa katastrofik?
- Jelaskan pentingnya pembobotan tugas dan penyeimbangan loss.
- Bagaimana teknik kuantisasi dan pruning membantu skalabilitas MTL?
- Jelajahi potensi pembelajaran multimodal dalam model MTL.

### 7.8.2. Latihan Praktis

#### Latihan Mandiri 7.1: Model MTL dengan Lapisan Bersama
**Tujuan:** Mengimplementasikan model yang menangani klasifikasi teks dan analisis sentimen secara bersamaan menggunakan satu backbone bersama.

#### Latihan Mandiri 7.2: Penyetelan Halus T5 untuk Tugas Khusus
**Tujuan:** Melatih model T5 pada dataset ringkasan dokumen teknis/medis secara spesifik.

#### Latihan Mandiri 7.3: Penyeimbangan Loss Multi-tugas
**Tujuan:** Bereksperimen dengan berbagai bobot $\lambda$ untuk tugas yang memiliki tingkat kesulitan berbeda.

#### Latihan Mandiri 7.4: Penskalaan dengan Pelatihan Terdistribusi
**Tujuan:** Menjalankan pelatihan model MTL besar menggunakan beberapa unit pemrosesan (GPU).

#### Latihan Mandiri 7.5: Desain Metrik Evaluasi Kustom
**Tujuan:** Membuat metrik penilaian baru yang menggabungkan akurasi dan kecepatan inferensi untuk model multi-tugas.