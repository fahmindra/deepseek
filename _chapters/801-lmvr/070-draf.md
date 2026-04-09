---
slug: lmvr-7
title: lmvr-7
---

# Bab 7: Multitask Learning: T5 dan Model Terpadu

> 💡 **"Multitask learning adalah paradigma kuat yang memanfaatkan representasi bersama untuk memungkinkan model berkinerja baik di berbagai tugas, seringkali melampaui performa model khusus tugas." — Andrew Ng**

> 📘 **Ringkasan Bab:**
> Bab 7 dari LMVR menggali ranah multitask learning (pembelajaran multi-tugas), menyoroti arsitektur dan kapabilitas model seperti T5 dan model terpadu lainnya. Bab ini dimulai dengan menjelaskan dasar-dasar multitask learning dan keunggulannya dibandingkan pendekatan tugas tunggal, dengan menekankan pentingnya representasi bersama di seluruh tugas. Kemudian mengeksplorasi arsitektur T5, yang membingkai setiap masalah NLP sebagai tugas teks-ke-teks, menyediakan pendekatan terpadu untuk menyelesaikan berbagai tugas dalam satu model. Bab ini juga mencakup tantangan dan strategi untuk menyempurnakan (*fine-tuning*) model-model ini untuk aplikasi spesifik, metode untuk mengevaluasi model multitask di berbagai tugas, dan teknik untuk menskalakan serta mengoptimalkan model ini untuk penerapan di dunia nyata. Terakhir, bab ini melihat ke depan pada masa depan multitask learning, mendiskusikan tren seperti pembelajaran multimodal dan pembelajaran berkelanjutan (*continual learning*), serta implikasinya bagi evolusi AI.

---

## 7.1. Pengantar Multitask Learning

Multitask learning (MTL) adalah sebuah pendekatan dalam pembelajaran mesin di mana sebuah model dilatih untuk melakukan beberapa tugas secara bersamaan, memanfaatkan representasi bersama di seluruh tugas untuk meningkatkan generalisasi dan efisiensi. Berbeda dengan pembelajaran tugas tunggal (*single-task learning*), di mana sebuah model dilatih pada tugas tertentu secara terisolasi, multitask learning bertujuan untuk meningkatkan performa dengan menangkap pola mendasar yang berguna di berbagai tugas yang berbeda. Ide kuncinya adalah bahwa tugas-tugas dapat berbagi pengetahuan dan representasi, memungkinkan model untuk melakukan generalisasi lebih baik dan menghindari *overfitting* pada tugas individu dengan menggunakan data dari semua tugas untuk mempertajam pemahamannya.

Dalam multitask learning, pertimbangkan contoh analisis sentimen dan klasifikasi topik, di mana sebuah model dilatih pada ulasan pelanggan untuk analisis sentimen (positif, negatif, netral) dan pada artikel berita untuk klasifikasi topik (seperti politik, olahraga, dan teknologi). Dengan berbagi parameter pada lapisan pemrosesan teks awal, model mempelajari pola bahasa umum yang menguntungkan kedua tugas tersebut. Analisis sentimen dapat memperoleh manfaat dari konteks topik, karena sentimen sering kali bervariasi antar topik, sementara klasifikasi topik meningkat dengan memahami fitur bahasa yang terkait dengan sentimen.

Demikian pula dalam contoh visi komputer, melatih model pada klasifikasi gambar (di mana setiap gambar memiliki satu label, seperti "kucing" atau "anjing") dan deteksi objek (mengidentifikasi banyak objek, seperti mendeteksi "kucing" dan "anjing" dalam gambar yang sama) memungkinkan lapisan konvolusional bersama untuk menangkap fitur seperti tepi dan bentuk, yang membantu dalam mengidentifikasi objek tunggal maupun mendeteksi banyak objek dalam gambar. Contoh lain melibatkan prediksi harga rumah dan tingkat kejahatan, di mana fitur lingkungan seperti tingkat pendapatan, kedekatan dengan sekolah, dan fasilitas lokal berdampak pada kedua hasil tersebut. Berbagi parameter pada lapisan awal memungkinkan model untuk mempelajari karakteristik lingkungan yang meningkatkan prediksi untuk harga rumah dan tingkat kejahatan. Dalam setiap contoh, parameter bersama memungkinkan model untuk menangkap pola yang bermanfaat di seluruh tugas, mencegah *overfitting* pada satu tugas saja dan meningkatkan generalisasi, yang merupakan esensi dari multitask learning.

![Figure 1: Ilustrasi paradigma Multitask Learning.](https://lmvr.rantai.dev/images/L6A504iArDGBvUq8WSeO-oc7p9ePlHw2lpWAldyqe-v1.webp)

Konsep Multitask Learning (MTL) pertama kali diusulkan oleh Rich Caruana pada tahun 1997 sebagai pendekatan inovatif dalam pembelajaran mesin. Caruana memperkenalkan MTL dengan gagasan bahwa berbagi parameter antar tugas memungkinkan model untuk menangkap struktur dasar yang umum bagi tugas-tugas tersebut, sehingga meningkatkan generalisasi dan membantu mencegah *overfitting*. Karya Caruana meletakkan dasar bagi MTL dengan menunjukkan bagaimana mempelajari beberapa tugas terkait secara bersama-sama dapat membuat setiap tugas menjadi lebih efisien dengan memanfaatkan informasi bersama.

Sejak usulan Caruana, MTL telah berevolusi melalui serangkaian kemajuan, terutama di bidang regularisasi, dekomposisi, percabangan, propagasi, optimasi, dan unifikasi. Pada awal 2000-an, para peneliti fokus pada pengembangan teknik regularisasi yang lebih baik untuk mengontrol bagaimana representasi bersama memengaruhi tugas individu. Teknik dekomposisi muncul, memungkinkan model untuk memisahkan pengetahuan umum dan khusus tugas dengan berbagi parameter secara selektif. Arsitektur percabangan (*branching*) lebih lanjut menyempurnakan ini dengan menggabungkan lapisan khusus tugas pada tahap akhir jaringan, memungkinkan tugas yang berbeda untuk bercabang dengan cara yang meningkatkan performa pada sub-tugas khusus sambil tetap menggunakan representasi bersama pada lapisan awal.

![Figure 2: Perjalanan sejarah paradigma Multitask Learning.](https://lmvr.rantai.dev/images/L6A504iArDGBvUq8WSeO-99ZJORzDR0obSWXELWUA-v1.png)

Pada tahun 2010-an, metode propagasi menjadi bagian integral dari MTL, memungkinkan cara yang lebih efisien untuk berbagi informasi antar tugas. Para peneliti mengeksplorasi teknik seperti *cross-stitch networks* dan komputasi kondisional, di mana hanya fitur yang relevan yang dibagikan antar tugas tertentu. Strategi optimasi untuk MTL juga sangat krusial pada periode ini, terutama untuk merancang metode berbasis gradien yang dapat menangani interaksi kompleks antar tugas. Teknik seperti pembobotan tugas dikembangkan untuk menyesuaikan penekanan pembelajaran pada setiap tugas secara dinamis.

Mendekati tahun 2022, penelitian MTL beralih ke arah penyatuan berbagai pendekatan untuk mengembangkan arsitektur fleksibel yang mampu beradaptasi secara dinamis terhadap banyak tugas, bahkan ketika kompleksitas dan keragaman tugas meningkat. Unifikasi melibatkan pencampuran berbagai strategi dalam satu kerangka kerja model yang dapat mengakomodasi berbagai tugas dengan penyesuaian minimal. Era MTL ini membawa kerangka kerja yang memungkinkan model untuk mengalokasikan sumber daya berdasarkan persyaratan tugas secara dinamis, membuat MTL lebih tangguh dan mudah beradaptasi di berbagai domain, dari pemrosesan bahasa alami hingga visi komputer.

Secara matematis, multitask learning dapat dibingkai sebagai masalah optimasi di mana model meminimalkan kombinasi tertimbang dari fungsi kerugian (*loss functions*) di beberapa tugas. Untuk sekumpulan tugas $T_1, T_2, \dots, T_n$, dengan fungsi kerugian terkait $\mathcal{L}_1, \mathcal{L}_2, \dots, \mathcal{L}_n$, fungsi kerugian keseluruhan dalam multitask learning dapat dinyatakan sebagai:

$$ \mathcal{L}_{\text{MTL}} = \sum_{i=1}^{n} \lambda_i \mathcal{L}_i, $$

di mana $\lambda_i$ adalah bobot yang mengontrol kepentingan relatif dari kerugian setiap tugas. Model harus menyeimbangkan minimalisasi kerugian ini dengan cara yang meningkatkan performa di semua tugas, sambil juga memastikan bahwa tidak ada satu tugas pun yang mendominasi proses pembelajaran. Salah satu keuntungan utama dari multitask learning adalah memungkinkan model untuk mendapatkan manfaat dari tugas tambahan (*auxiliary tasks*), yang dapat menyediakan data tambahan dan regularisasi yang membantu meningkatkan performa pada tugas utama yang diminati.

Perbedaan antara pembelajaran tugas tunggal dan multitask learning terletak pada struktur model dan bagaimana informasi dibagikan di seluruh tugas. Dalam multitask learning, model dibagi menjadi lapisan bersama (*shared layers*), yang mempelajari representasi umum di semua tugas, dan lapisan khusus tugas (*task-specific layers*), yang fokus pada mempelajari fitur unik untuk setiap tugas. Struktur ini dapat dinyatakan sebagai:

$$ y_i = f_{\text{task}_i}(f_{\text{shared}}(x)), $$

di mana $x$ mewakili masukan, $f_{\text{shared}}$ mewakili lapisan bersama, dan $f_{\text{task}_i}$ mewakili lapisan khusus tugas untuk tugas $i$. Efektivitas multitask learning bergantung pada seberapa baik lapisan bersama melakukan generalisasi di seluruh tugas sementara lapisan khusus tugas menangani nuansa setiap tugas.

![Figure 3: Ilustrasi sistem Q&A sadar konteks.](https://lmvr.rantai.dev/images/L6A504iArDGBvUq8WSeO-44I42HWGcG6iJYnlehgJ-v1.png)

Dalam arsitektur multitask learning (MTL) yang dirancang untuk tugas *Question Answering* (Q&A) dan *Context Learning*, model dimulai dengan pengodean independen, diikuti oleh lapisan pengodean penyelarasan (*alignment encoding layer*), lalu lapisan atensi bersama ganda (*dual co-attention layer*). Dalam lapisan atensi bersama ganda ini, model melakukan atensi silang antar pertanyaan dan konteks. Setelah itu, lapisan *self-attention* lebih lanjut memperbaiki representasi tugas dengan menangkap dependensi internal. Struktur ini secara efektif menyeimbangkan pembelajaran bersama dengan fokus khusus tugas.

Meskipun memiliki keunggulan, multitask learning juga menghadapi tantangan, terutama dalam mengelola interferensi tugas (*task interference*), di mana mengoptimalkan satu tugas dapat berkonflik dengan tujuan tugas lainnya. Penyeimbangan kerugian (*loss balancing*) juga merupakan tantangan kritis. Memilih bobot kerugian tugas ($\lambda_i$) yang tepat sangat penting untuk mencegah satu tugas menutupi tugas lainnya. Teknik seperti pembobotan tugas dinamis telah diperkenalkan untuk mengatasi tantangan ini.

Dalam arsitektur MTL yang terinspirasi oleh model seperti ELECTRA, representasi token masukan mengikuti jalur pemrosesan ganda melalui generator, diskriminator, dan lapisan khusus tugas.

![Figure 4: Contoh arsitektur model MTL berbasis ELECTRA.](https://lmvr.rantai.dev/images/L6A504iArDGBvUq8WSeO-RnIsJrNtNLRKrSAiCzav-v1.png)

Setelah token diproses oleh diskriminator, representasi yang dihasilkan dapat diteruskan ke lapisan model bahasa khusus tugas yang disesuaikan untuk berbagai tugas linguistik, seperti skor *Part-of-Speech* (POS), skor *Dependency Head*, skor *Constituent Span*, dan skor *Semantic Role*.

Dalam hal implementasi praktis, perpustakaan Python `transformers` menyediakan antarmuka yang setara dengan `rust-bert` untuk melakukan tugas-tugas NLP tingkat lanjut. Berikut adalah contoh padanan Python untuk melakukan peringkasan teks menggunakan model T5:

```python
# Contoh Python menggunakan library transformers (padanan rust-bert)
from transformers import pipeline

def main():
    # Menyiapkan pipeline peringkasan dengan model t5-small
    summarizer = pipeline("summarization", model="t5-small")

    # Teks input untuk diringkas
    input_text = """In findings published Tuesday in Cornell University's arXiv by a team of scientists 
    from the University of Montreal and a separate report published Wednesday in Nature Astronomy by a team 
    from University College London (UCL), the presence of water vapour was confirmed in the atmosphere of K2-18b, 
    a planet circling a star in the constellation Leo. This is the first such discovery in a planet in its star's 
    habitable zone — not too hot and not too cold for liquid water to exist..."""

    # Melakukan peringkasan
    summary = summarizer(input_text, min_length=10, max_length=512)

    for s in summary:
        print(s['summary_text'])

if __name__ == "__main__":
    main()
```

Berikut adalah contoh lain penggunaan T5 untuk tugas penerjemahan multibahasa di Python:

```python
from transformers import pipeline

def main():
    # Menggunakan pipeline translasi T5-base
    # T5 menangani tugas berdasarkan prefix prompt
    translator = pipeline("translation_en_to_fr", model="t5-base")
    
    source_sentence = "This sentence will be translated into multiple languages."
    
    # Translate ke Bahasa Prancis
    fr_translation = translator(source_sentence)
    print(f"French: {fr_translation[0]['translation_text']}")

    # Untuk bahasa lain dengan model yang sama (manual prompt)
    from transformers import T5Tokenizer, T5ForConditionalGeneration
    
    model = T5ForConditionalGeneration.from_pretrained("t5-base")
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    
    # Translate ke Bahasa Jerman
    input_ids = tokenizer(f"translate English to German: {source_sentence}", return_tensors="pt").input_ids
    outputs = model.generate(input_ids)
    print(f"German: {tokenizer.decode(outputs[0], skip_special_tokens=True)}")

if __name__ == "__main__":
    main()
```

Keluarga model T5 menawarkan berbagai varian seperti `t5-small`, `t5-base`, `t5-large`, `t5-3b`, dan `t5-11b`. Ada juga seri Flan-T5 yang diinstruksikan (*instruction-tuned*), MT5 (versi multibahasa), dan mT0 (versi multibahasa yang diinstruksikan).

Berikut adalah contoh penggunaan Python untuk mengunduh model dari Hugging Face Hub (padanan untuk krate `hf-hub` di Rust):

```python
from huggingface_hub import hf_hub_download
import os

def main():
    models = ["t5-small", "t5-base"]
    files_to_download = ["config.json", "pytorch_model.bin", "spiece.model"]

    for model_id in models:
        print(f"Mengunduh model: {model_id}")
        local_dir = f"./models/{model_id}"
        os.makedirs(local_dir, exist_ok=True)
        
        for filename in files_to_download:
            path = hf_hub_download(repo_id=model_id, filename=filename, local_dir=local_dir)
            print(f"Selesai mengunduh {filename} ke {path}")

if __name__ == "__main__":
    main()
```

Sebagai kesimpulan, multitask learning meningkatkan generalisasi model dan efisiensi data dengan memanfaatkan representasi bersama di seluruh tugas. Meskipun memperkenalkan tantangan seperti interferensi tugas, MTL telah terbukti efektif dalam berbagai aplikasi dunia nyata.

---

## 7.2. Arsitektur T5

Arsitektur T5 mewakili inovasi signifikan dalam ranah multitask learning. Dikembangkan oleh Google, T5 dirancang sebagai model terpadu yang membingkai setiap tugas NLP dalam format teks-ke-teks yang sama. Pendekatan ini menyederhanakan arsitektur model dan proses pelatihan, karena memperlakukan masukan dan keluaran sebagai string teks, terlepas dari tugasnya.

![Figure 5: Ilustrasi arsitektur model encoder-decoder T5.](https://lmvr.rantai.dev/images/L6A504iArDGBvUq8WSeO-aZQSQalpuRrifpo7DtTj-v1.png)

Pada intinya, arsitektur T5 didasarkan pada struktur encoder-decoder model Transformer. Inovasi utama T5 terletak pada bagaimana ia memformulasikan ulang setiap tugas ke dalam paradigma teks-ke-teks ini. Secara matematis, model mempelajari distribusi probabilitas kondisional:

$$ P(Y | X) = \prod_{t=1}^{T} P(y_t | y_1, y_2, \dots, y_{t-1}, X), $$

Pelatihan awal (*pre-training*) memainkan peran kritis dalam keberhasilan T5. Selama pre-training, model belajar memprediksi token yang disamarkan, sebuah tugas yang dikenal sebagai "korupsi rentang" (*span corruption*). Fungsi kerugian pre-training dapat dinyatakan sebagai:

$$ \mathcal{L}_{\text{pretrain}} = - \sum_{i=1}^{N} \log P(y_i | X_{\setminus i}), $$

Berikut adalah representasi konseptual blok encoder dan decoder T5 di Python menggunakan PyTorch:

```python
import torch
import torch.nn as nn

class T5EncoderBlock(nn.Module):
    def __init__(self, n_embd, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.self_attn = nn.MultiheadAttention(n_embd, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ff = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd)
        )

    def forward(self, x):
        # Attention block
        norm_x = self.ln1(x)
        attn_out, _ = self.self_attn(norm_x, norm_x, norm_x)
        x = x + attn_out
        
        # Feed-forward block
        norm_x = self.ln2(x)
        ff_out = self.ff(norm_x)
        x = x + ff_out
        return x

class T5Model(nn.Module):
    def __init__(self, vocab_size, n_embd, n_layers, n_heads):
        super().__init__()
        self.shared_embedding = nn.Embedding(vocab_size, n_embd)
        self.encoder = nn.ModuleList([T5EncoderBlock(n_embd, n_heads) for _ in range(n_layers)])
        # (Decoder dan logika cross-attention ditambahkan di sini secara serupa)

    def forward(self, src_ids):
        x = self.shared_embedding(src_ids)
        for block in self.encoder:
            x = block(x)
        return x
```

Penerapan industri T5 sangat luas, mencakup penerjemahan, peringkasan, menjawab pertanyaan, hingga pembuatan dialog. Kerangka kerjanya yang terpadu membuatnya sangat menarik bagi perusahaan yang membutuhkan model serbaguna yang mampu menangani banyak tugas NLP dengan modifikasi minimal.

Berikut adalah contoh penggunaan Python untuk menjalankan inferensi T5 menggunakan library `transformers` (padanan untuk contoh Rust `candle`):

```python
from transformers import T5Tokenizer, T5ForConditionalGeneration

def main():
    model_id = "t5-small"
    tokenizer = T5Tokenizer.from_pretrained(model_id)
    model = T5ForConditionalGeneration.from_pretrained(model_id)

    prompt = "Translate English to French: How are you?"
    
    # Tokenisasi
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    
    # Generasi (Decoding)
    outputs = model.generate(
        input_ids, 
        max_length=512, 
        temperature=0.8, 
        do_sample=True,
        repetition_penalty=1.1
    )
    
    decoded_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Generated text: {decoded_text}")

if __name__ == "__main__":
    main()
```

---

## 7.3. Model Terpadu untuk Multitask Learning

Model terpadu dalam multitask learning bertujuan untuk merancang satu arsitektur model tunggal yang dapat menangani berbagai tugas tanpa memerlukan modifikasi atau komponen khusus tugas. Model-model seperti mT5, BART, dan UnifiedQA menunjukkan fleksibilitas arsitektur agnostik-tugas ini.

Tantangan utama adalah menghindari masalah interferensi tugas. Secara matematis, model harus meminimalkan kerugian gabungan:

$$ \mathcal{L}_{\text{unified}} = \sum_{i=1}^{n} \lambda_i \mathcal{L}_i, $$

Berikut adalah contoh penggunaan model BART untuk peringkasan teks di Python:

```python
from transformers import pipeline

def main():
    # Menggunakan model DistilBART
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-6-6")

    input_text = """In findings published Tuesday in Cornell University's arXiv... (teks panjang)"""

    summary = summarizer(
        input_text, 
        num_beams=1, 
        min_length=56, 
        max_length=142
    )
    
    print(summary[0]['summary_text'])

if __name__ == "__main__":
    main()
```

Berikut adalah contoh penggunaan MBART-50 untuk penerjemahan banyak-ke-banyak (*many-to-many*) di Python:

```python
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

def main():
    model_name = "facebook/mbart-large-50-many-to-many-mmt"
    model = MBartForConditionalGeneration.from_pretrained(model_name)
    tokenizer = MBart50TokenizerFast.from_pretrained(model_name)

    source_sentence = "This sentence will be translated in multiple languages."
    
    # Set source language ke English
    tokenizer.src_lang = "en_XX"
    encoded_input = tokenizer(source_sentence, return_tensors="pt")

    # Translate ke French
    generated_tokens = model.generate(
        **encoded_input, 
        forced_bos_token_id=tokenizer.lang_code_to_id["fr_XX"]
    )
    print(f"French: {tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]}")

    # Translate ke Indonesian
    generated_tokens = model.generate(
        **encoded_input, 
        forced_bos_token_id=tokenizer.lang_code_to_id["id_ID"]
    )
    print(f"Indonesian: {tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]}")

if __name__ == "__main__":
    main()
```

---

## 7.4. Fine-Tuning Model Multitask untuk Aplikasi Spesifik

*Fine-tuning* melibatkan kelanjutan minimalisasi fungsi kerugian pada dataset baru yang khusus untuk tugas tersebut. Tujuan barunya seringkali diformulasikan sebagai:

$$ \mathcal{L}_{\text{fine-tune}} = \lambda_1 \mathcal{L}_{\text{new}} + \lambda_2 \mathcal{L}_{\text{pretrain}}, $$

Untuk mencegah *catastrophic forgetting* (lupa fatal), teknik seperti *Elastic Weight Consolidation* (EWC) dapat digunakan:

$$ \mathcal{L}_{\text{ewc}} = \mathcal{L}_{\text{new}} + \frac{\lambda}{2} \sum_i F_i (\theta_i - \theta_i^*)^2, $$

Berikut adalah struktur loop pelatihan dasar untuk *fine-tuning* di Python:

```python
import torch
import torch.optim as optim

def train_one_epoch(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    for batch in dataloader:
        inputs, labels = batch
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

# Inisialisasi model, optimizer, dan jalankan loop
```

Tren terbaru adalah penggunaan LoRA (*Low-Rank Adaptation*) untuk *fine-tuning* yang efisien parameter:

$$ W = W_0 + \Delta W = W_0 + A B, $$

Berikut adalah implementasi konseptual LoRA di Python:

```python
import torch.nn as nn

class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank=4):
        super().__init__()
        self.A = nn.Parameter(torch.randn(in_dim, rank))
        self.B = nn.Parameter(torch.zeros(rank, out_dim))

    def forward(self, x):
        # x @ A @ B
        return (x @ self.A) @ self.B

# Lapisan ini ditambahkan secara paralel ke lapisan linear yang ada
```

---

## 7.5. Mengevaluasi dan Melakukan Benchmark Model Multitask

Evaluasi keseluruhan model MTL dapat ditangkap sebagai agregat dari metrik individu:

$$ \mathcal{M}_{\text{overall}} = \sum_{i=1}^{n} w_i \mathcal{M}_i, $$

Berikut adalah contoh cara menghitung metrik evaluasi (seperti akurasi) untuk berbagai tugas di Python:

```python
import torch

def evaluate_multitask(model, test_datasets):
    # test_datasets adalah dict: { "task_name": dataloader }
    results = {}
    model.eval()
    with torch.no_grad():
        for task_name, loader in test_datasets.items():
            correct = 0
            total = 0
            for inputs, targets in loader:
                outputs = model(inputs, task=task_name)
                predictions = torch.argmax(outputs, dim=-1)
                correct += (predictions == targets).sum().item()
                total += targets.size(0)
            results[task_name] = correct / total
    return results

# Bobot kepentingan tugas
weights = {"translation": 0.4, "summarization": 0.3, "classification": 0.3}
```

---

## 7.6. Menskalakan dan Mengoptimalkan Model Multitask Learning

Menskalakan model MTL menghadirkan tantangan biaya komputasi. Teknik optimasi meliputi:
1.  **Kuantisasi:** Mengurangi presisi (misal FP32 ke INT8).
2.  **Pruning:** Menghapus parameter yang tidak penting.
3.  **Distributed Training:** Membagi beban kerja di beberapa GPU.

Berikut adalah simulasi sederhana untuk memuat model yang dikuantisasi di Python (menggunakan library `bitsandbytes` atau fitur bawaan PyTorch):

```python
from transformers import T5ForConditionalGeneration
import torch

# Contoh konseptual pemuatan model dengan kuantisasi 8-bit
def load_quantized_t5(model_id):
    # Membutuhkan library bitsandbytes
    # model = T5ForConditionalGeneration.from_pretrained(model_id, load_in_8bit=True, device_map="auto")
    print(f"Memuat model {model_id} dengan simulasi kuantisasi.")

if __name__ == "__main__":
    load_quantized_t5("t5-small")
```

---

## 7.7. Arah Masa Depan dalam Multitask Learning dan Model Terpadu

Tren saat ini mengarah pada:
- **Multimodal Learning:** Integrasi teks, gambar, dan audio.
- **Continual Learning:** Beradaptasi dengan tugas baru tanpa retrain dari nol.

Berikut adalah contoh implementasi logis EWC di Python untuk mencegah *catastrophic forgetting*:

```python
def ewc_loss(model, current_loss, fisher_matrices, opt_params, ewc_lambda):
    # fisher_matrices: Pentingnya setiap parameter dari tugas sebelumnya
    # opt_params: Nilai parameter optimal dari tugas sebelumnya
    
    extra_loss = 0
    for name, param in model.named_parameters():
        if name in fisher_matrices:
            fisher = fisher_matrices[name]
            opt_p = opt_params[name]
            # Penalty: (param - opt_p)^2 * fisher
            extra_loss += (fisher * (param - opt_p)**2).sum()
            
    return current_loss + ewc_lambda * extra_loss
```

---

## 7.8. Kesimpulan

Bab 7 memberikan eksplorasi menyeluruh tentang multitask learning dan model terpadu, menawarkan wawasan tentang bagaimana pendekatan ini dapat meningkatkan performa dan efisiensi model NLP.

### 7.8.1. Pembelajaran Lebih Lanjut dengan GenAI

- Jelaskan perbedaan mendasar antara multitask learning dan transfer learning.
- Deskripsikan arsitektur T5 dan bagaimana ia menangani beragam tugas NLP dalam satu format.
- Diskusikan strategi untuk meminimalkan interferensi tugas dalam model terpadu.
- Bagaimana teknik LoRA membantu dalam *fine-tuning* model besar dengan sumber daya terbatas?
- Analisis implikasi etis dari model terpadu yang memproses data dari berbagai domain sensitif.

### 7.8.2. Praktik Mandiri

- **Latihan 7.1:** Implementasikan model MTL sederhana dengan satu lapisan bersama dan dua kepala khusus tugas untuk klasifikasi teks.
- **Latihan 7.2:** Lakukan *fine-tuning* pada model Flan-T5 untuk tugas klasifikasi ulasan film.
- **Latihan 7.3:** Eksperimen dengan pembobotan kerugian ($\lambda$) yang berbeda pada model MTL dan amati pengaruhnya terhadap setiap tugas.
- **Latihan 7.4:** Gunakan pelatihan terdistribusi (simulasi) untuk mempercepat pelatihan model T5 pada dataset besar.
- **Latihan 7.5:** Rancang metrik evaluasi kustom yang menggabungkan skor akurasi dan latensi inferensi untuk model multitask.