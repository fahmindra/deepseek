---
slug: modul-6-5
title: modul-6-5
---
# 2.5 Varian CPT: Full, Partial, dan Adapter-Based (LoRA)

---

Di subtopik sebelumnya, kita membahas teknik agar CPT "aman". Sekarang, kita akan membahas **arsitektur pelatihannya**.

Pertanyaan utama seorang AI Engineer saat melakukan CPT adalah: **"Apakah saya harus memperbarui semua parameter model?"**

Bayangkan Anda ingin merenovasi rumah (Model).
1. Apakah Anda merobohkan semua tembok dan membangun ulang? (**Full CPT**)
2. Apakah Anda hanya mengecat ulang kamar tertentu? (**Partial CPT**)
3. Atau apakah Anda hanya menambahkan furnitur baru tanpa menyentuh bangunan asli? (**Adapter-Based**)

Pilihan varian ini akan sangat menentukan kecepatan training, kebutuhan memori (VRAM), dan seberapa besar risiko model Anda "lupa" ingatan lamanya. Di subtopik ini, kita akan membedah tiga pendekatan tersebut dan memperkenalkan standar industri saat ini: **LoRA**.

---

## 2.5.1 Full CPT (Full-Parameter Training)

Ini adalah pendekatan tradisional dan paling "brutal".

**Apa itu Full CPT sebenarnya?**
**Full CPT adalah** proses di mana **seluruh bobot (*weights*)** model mulai dari *embedding*, semua lapisan *Self-Attention*, semua lapisan FFN, hingga *output head* dibiarkan cair (*unfrozen*) dan diperbarui selama proses *backpropagation*.

*   **Kelebihan:** Potensi adaptasi maksimal. Karena semua neuron bisa berubah, model bisa mengubah struktur pemahaman dasarnya secara total. Ini wajib dilakukan jika domain baru sangat asing (misal: melatih Llama Inggris menjadi fasih Bahasa Batak).
*   **Kekurangan:**
    *   **Biaya VRAM Ekstrem:** Untuk model 7B (7 Miliar parameter), Anda butuh memuat 7B bobot + 7B gradien + 14B state optimizer. Butuh GPU kelas data center (A100).
    *   **Risiko Tinggi:** Karena semua bobot berubah, risiko *Catastrophic Forgetting* sangat tinggi jika *hyperparameter* tidak diatur sempurna.

---

## 2.5.2 Partial CPT (Layer Freezing)

Pendekatan ini mencoba mencari jalan tengah dengan membekukan sebagian "otak" model.

**Apa itu Partial CPT (Layer Freezing)?**
Partial CPT adalah teknik di mana kita membekukan (*freeze*) sebagian besar lapisan awal model dan hanya melatih N lapisan terakhir (teratas).

Dalam Deep Learning, lapisan awal (bawah) biasanya menangkap fitur-fitur dasar (sintaksis, tata bahasa), sedangkan lapisan akhir (atas) menangkap konsep abstrak dan semantik tingkat tinggi.

Dengan membekukan lapisan bawah, kita menjaga kemampuan bahasa dasar model tetap utuh, sambil membiarkan model menyesuaikan pemahaman konsep tingkat tingginya terhadap domain baru.

*   **Kelebihan:** Menghemat memori (gradien tidak dihitung untuk lapisan beku) dan lebih aman dari *forgetting*.
*   **Kekurangan:** Seringkali kurang efektif. Perubahan pengetahuan seringkali membutuhkan penyesuaian di seluruh kedalaman jaringan, bukan hanya di permukaan.

---

## 2.5.3 Adapter-Based CPT (PEFT & LoRA)

Ini adalah revolusi modern dalam *fine-tuning* dan CPT. Pendekatan ini masuk dalam kategori **PEFT (Parameter-Efficient Fine-Tuning).**

**Definisi: Adapter-Based CPT (LoRA)**
Alih-alih mengubah bobot model asli $(W)$, teknik seperti LoRA (Low-Rank Adaptation) membekukan model asli sepenuhnya. Ia kemudian menyisipkan matriks kecil tambahan (Adapter) di samping lapisan asli yang bertugas mempelajari "perubahan".

Saat inferensi, output adalah: $W_{asli} + \text{Adapter}$

**Analogi: Kacamata Pintar**
*   Bayangkan model adalah mata manusia.
*   **Full CPT:** Melakukan operasi lasik pada mata. Permanen, mahal, berisiko.
*   **LoRA:** Memakaikan kacamata khusus. Mata asli tidak disentuh (beku), tapi penglihatannya dimodifikasi oleh lensa (adapter) agar bisa melihat "domain medis" dengan jelas. Jika kacamata dilepas, ia kembali normal.

**Implementasi Kode Praktis (Menggunakan peft):**
Berikut adalah cara mengubah skrip CPT kita sebelumnya menjadi CPT berbasis LoRA yang hemat memori.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType


model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)


# 1. Konfigurasi LoRA
# Kita menentukan lapisan mana yang akan ditempeli adapter
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,            # Rank: Ukuran matriks adapter (makin kecil makin hemat)
    lora_alpha=32,  # Scaling factor
    lora_dropout=0.1
)


# 2. Bungkus Model dengan PEFT
# Model asli dibekukan, adapter ditambahkan
model = get_peft_model(model, peft_config)


# Cek penghematan parameter
model.print_trainable_parameters()
```

**Output:**
```text
trainable params: 294,912 || all params: 124,734,720 || trainable%: 0.236
```

Lihat pada output di atas, Kita hanya melatih **0.2%** parameter. Ini memungkinkan CPT dilakukan pada GPU konsumen (seperti RTX 3060/4090) dengan risiko *forgetting* yang sangat minim karena "memori asli" model dibekukan total.

---

## Mini Quiz

[Konten Interaktif: Multiple Choice Quiz]
*(Akses kuis melalui platform LMS: https://lms.sdmdigital.id/h5p/embed.php?url=https%3A%2F%2Flms.sdmdigital.id%2Fpluginfile.php%2F635431%2Fmod_page%2Fcontent%2F13%2Fmultiple-choice-188.h5p)*

---

> **Refleksi Singkat**
>
> Teknik LoRA (2.5.3) memungkinkan kita melakukan CPT dengan hanya melatih <1% parameter.
>
> Bayangkan Anda adalah seorang AI Engineer di sebuah startup dengan anggaran terbatas yang ingin membuat model spesialis hukum Indonesia.
>
> Mengapa LoRA menjadi pilihan yang jauh lebih masuk akal secara bisnis dan teknis dibandingkan Full CPT, dan apa implikasinya terhadap infrastruktur serving (deployment) Anda nanti jika Anda ingin punya 5 model spesialis berbeda?