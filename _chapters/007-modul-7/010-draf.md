---
slug: modul-7-1
title: modul-7-1
---
# 3.1 PEFT (Parameter-Efficient Fine-Tuning)

---

Metode tradisional untuk mengadaptasi model adalah **Full Fine-Tuning (FFT)**. Dalam FFT, kita memuat semua parameter model (misal: 7 Miliar) dan memperbarui *setiap* parameter tersebut melalui *backpropagation*.

Ini mungkin terdengar standar, tetapi untuk LLM, ini adalah mimpi buruk logistik. Solusinya adalah **PEFT (Parameter-Efficient Fine-Tuning)**.

![Parameter-Efficient Fine Tuning (PEFT)](https://lms.sdmdigital.id/pluginfile.php/635441/mod_page/content/11/PEFT.jpg)
*Gambar 1. Parameter-Efficient Fine Tuning (PEFT)*

**Apa itu PEFT?**

**PEFT adalah** serangkaian teknik yang **membekukan (freeze)** sebagian besar parameter model *pre-trained* dan hanya melatih sejumlah kecil parameter (bisa berupa parameter tambahan atau subset parameter asli). Hasilnya adalah Kinerja yang sebanding dengan *Full Fine-Tuning*, tetapi dengan kebutuhan komputasi dan penyimpanan yang jauh lebih rendah.

Di subtopik ini, kita akan membedah mengapa FFT tidak lagi relevan untuk sebagian besar kasus, dan melihat peta strategi PEFT.

---

## 3.1.1 Tantangan Skalabilitas pada Full Fine-Tuning (FFT)

Mengapa kita sangat ingin menghindari FFT? Ada dua tembok besar yang menghalangi:

1.  **Batasan Memori atau OOM (Out Of Memory, bisa juga disebut VRAM Overhead):** Melatih model tidak sama dengan menjalankan model. Untuk melatih model 7B (yang bobotnya 14GB), Anda tidak hanya butuh 14GB VRAM. Anda butuh memori untuk:
    *   **Gradients:** Menyimpan arah perubahan untuk setiap parameter.
    *   **Optimizer States:** (Misal AdamW) Menyimpan *momentum* dan *variance* untuk setiap parameter (ini memakan memori 2x lipat dari model itu sendiri!).
    *   **Activations:** Menyimpan data yang mengalir antar layer.
    *   **Total:** FFT untuk model 7B bisa membutuhkan **>80GB VRAM** (Setara GPU A100 yang sangat mahal).
2.  **Tembok Penyimpanan (Storage Bottleneck):** Bayangkan Anda ingin membuat 100 model spesial (satu untuk Hukum, satu untuk Medis, satu untuk Koding, dst.).
    *   Jika pakai FFT: Anda punya 100 file model raksasa (100 x 14GB = 1.4 Terabyte).
    *   Ini tidak praktis untuk *deployment* dan manajemen versi.

![Perbedaan Full Finetuning dan PEFT](https://lms.sdmdigital.id/pluginfile.php/635441/mod_page/content/11/perbedaan%20FineTuning%20dan%20PEFT.jpg)
*Gambar 2. Perbedaan Full Finetuning dan PEFT*

Gambar di atas mengilustrasikan perbedaan fundamental dalam strategi *fine-tuning* model besar.

*   **Sisi Kiri (Full Fine-Tuning):** Model dilambangkan sebagai tumpukan blok Merah/Oranye (Trainable). Ini menunjukkan bahwa 100% dari total bobot model diperbarui (*backpropagation* mengalir ke mana-mana). Hal ini memicu kebutuhan VRAM dan risiko *Catastrophic Forgetting* yang sangat tinggi.
*   **Sisi Kanan (PEFT):** Model utama (Backbone) dilambangkan sebagai tumpukan blok Biru/Abu-abu dengan ikon gembok (Frozen). Ini menunjukkan bobot utama *tidak disentuh*. Perubahan bobot terjadi hanya pada *adapter* kecil (blok tipis berwarna Merah), yang bekerja secara paralel dan menyalurkan hasil perubahan ke output.

Perbedaan visual antara 100% vs kurang dari 1% bobot yang dilatih inilah yang membuat PEFT menjadi solusi skalabilitas dan efisiensi.

Kita telah melihat bahwa PEFT adalah sebuah kebutuhan, bukan kemewahan. Namun, PEFT sendiri bukanlah satu teknik tunggal. Ada beberapa pendekatan arsitektural yang berbeda untuk mencapai efisiensi ini, mulai dari menambah lapisan kecil hingga mengubah cara representasi bobot. Pendekatan ini dapat diklasifikasikan ke dalam tiga keluarga besar, tergantung pada *di mana* perubahan parameter dilakukan.

---

## 3.1.2 Tiga Keluarga Utama PEFT

PEFT bukan hanya satu teknik. Ini adalah "keluarga besar" strategi efisiensi. Secara umum, ada tiga pendekatan untuk melakukan PEFT:

1.  **Additive Methods (Menambah Komponen):**
    *   **Ide:** Kita tidak menyentuh model asli sama sekali. Kita **menyisipkan** lapisan baru yang kecil di antara lapisan model asli. Hanya lapisan sisipan ini yang dilatih.
    *   **Contoh:** *Adapters*, *Soft Prompting*.
    *   **Kelebihan:** Aman, model asli 100% utuh.
    *   **Kekurangan:** Menambah lapisan berarti menambah waktu komputasi saat inferensi (sedikit lebih lambat).
2.  **Selective Methods (Memilih Pemenang):**
    *   **Ide:** Kita tidak menambah apa-apa. Kita hanya memilih sebagian kecil parameter asli (misal: hanya *bias* atau hanya lapisan *top*) untuk dilatih, sisanya dibekukan.
    *   **Contoh:** *BitFit*.
    *   **Kekurangan:** Seringkali sulit mencapai performa maksimal dibandingkan metode lain.
3.  **Reparameterization Methods (Mengubah Representasi):**
    *   **Ide:** Kita menggunakan trik matematika untuk merepresentasikan perubahan bobot ($\Delta W$) dengan matriks yang jauh lebih kecil (Rank rendah).
    *   **Contoh: LoRA (Low-Rank Adaptation)**.
    *   **Kelebihan Utama:** Tidak menambah latensi inferensi (karena matriks bisa digabung ulang ke model asli) dan sangat hemat memori. Inilah yang akan menjadi fokus utama kita di 2.3.2.

Kita telah memetakan tiga keluarga besar PEFT (2.3.1.2). Untuk memvalidasi klaim efisiensi yang masif ini, seorang *engineer* harus dapat **mengukur secara langsung** berapa persen parameter model yang sebenarnya diperbarui dan bagaimana cara mengubahnya. Langkah pertama sebelum memulai pelatihan adalah melakukan verifikasi kode ini.

---

## 3.1.3 Implementasi Kode: Mengecek Parameter yang Dilatih

Sebelum masuk ke LoRA, penting bagi seorang *engineer* untuk bisa memverifikasi: "Berapa persen parameter yang sebenarnya saya latih?"

Library `peft` dari Hugging Face memiliki fungsi utilitas yang sangat berguna untuk ini: `print_trainable_parameters`.

Berikut adalah simulasi kode untuk melihat perbedaannya:

```python
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model


# 1. Memuat Model Dasar (Base Model)
model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)


def print_trainable_parameters(model):
    """
    Fungsi helper untuk menghitung jumlah parameter yang dilatih vs total parameter
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
   
    print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}%")


print("--- Status Awal (Full Fine-Tuning) ---")
print_trainable_parameters(model)


print("\n--- Status Setelah Aplikasi PEFT (Simulasi LoRA) ---")


# 2. Menerapkan Konfigurasi PEFT (LoRA)
config = LoraConfig(r=8)
peft_model = get_peft_model(model, config)


# 3. Cek status setelah aplikasi PEFT
peft_model.print_trainable_parameters()
```

**Penjelasan Kode:**

*   **Fungsi Helper (`print_trainable_parameters`):** Fungsi ini melakukan iterasi (loop) ke seluruh parameter dalam model. Ia membedah parameter berdasarkan atribut `.requires_grad`. Jika `True`, berarti parameter tersebut akan diperbarui (dilatih) saat *backpropagation*. Jika `False`, parameter tersebut "beku".
*   **`get_peft_model`:** Ini adalah fungsi inti dari library `peft`. Fungsi ini melakukan dua hal secara otomatis:
    *   Membekukan (*Freeze*) seluruh parameter model asli (GPT-2).
    *   Menyisipkan dan mengaktifkan modul *adapter* tambahan sesuai konfigurasi (`config`) yang diberikan.

**Output yang akan dihasilkan:**

```text
--- Status Awal (Full Fine-Tuning) ---
trainable params: 124439808 || all params: 124439808 || trainable%: 100.00%


--- Status Setelah Aplikasi PEFT (Simulasi LoRA) ---
trainable params: 294912 || all params: 124734720 || trainable%: 0.24%
```

**Penjelasan Output:**

*   **Pada Status Awal**, 100% parameter (sekitar 124 Juta) siap dilatih. Ini berarti memori GPU harus menampung gradien untuk 124 juta angka tersebut.
*   **Pada Status PEFT**, jumlah parameter yang dilatih anjlok drastis menjadi hanya **294 Ribu** (sekitar 0,24%). Ini berarti beban memori untuk pelatihan berkurang lebih dari 99%. Model asli tetap ada sebagai fondasi (backbone), tetapi tidak membebani memori komputasi (*gradient memory*).

---

### Mini Quiz

[Mini Quiz Embed - app.lumi.education]

---

> **Refleksi Singkat**
>
> Kita telah membuktikan melalui kode bahwa PEFT hanya melatih <1% parameter. Bayangkan Anda adalah CTO sebuah startup. Anda ingin melatih model khusus untuk 100 klien perusahaan yang berbeda.
>
> *   **Skelario A (Full Fine-Tuning):** Anda butuh menyimpan 100 model raksasa (Total 1.4 Terabyte).
> *   **Skelario B (PEFT/Adapter):** Anda cukup menyimpan 1 model dasar dan 100 adapter kecil (Total ~15GB).
>
> Selain penghematan storage yang gila-gilaan ini, bagaimana hal ini mengubah arsitektur serving (deployment) Anda? Apakah Anda perlu menyalakan 100 server GPU yang berbeda, atau bisakah satu server melayani semua klien dengan teknik hot-swapping adapter?