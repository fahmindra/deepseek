---
slug: modul-7-2
title: modul-7-2
---

# 3.2 Konsep Update Decomposition pada LoRA dan QLora

Di Subtopik 3.1, kita melihat bukti bahwa PEFT hanya melatih <1% parameter. Di Subtopik 3.2 ini, kita akan menjawab pertanyaan "BAGAIMANA?".

Bagaimana mungkin kita bisa mengubah perilaku model raksasa hanya dengan menyentuh bagian yang sangat kecil darinya? Apakah itu sihir? Bukan, itu adalah Aljabar Linear.

Kita akan membedah dua inovasi teknis paling berpengaruh dalam fine-tuning modern: LoRA (Low-Rank Adaptation) yang menghemat memori gradien, dan QLoRA (Quantized LoRA) yang menghemat memori model dasar.

Bayangkan Anda ingin pergi dari Jakarta ke Bandung. Anda tidak perlu mengubah posisi seluruh pulau Jawa (memindahkan triliunan atom tanah). Anda hanya perlu mengubah posisi diri Anda di atas peta (koordinat kecil).

Peneliti (Aghajanyan et al., 2020) ditemukan fenomena serupa pada LLM: Meskipun model memiliki miliaran parameter (dimensi sangat tinggi), perubahan bobot yang diperlukan untuk beradaptasi ke tugas baru (misal: belajar gaya bahasa medis) sebenarnya memiliki "dimensi intrinsik yang rendah".

Artinya, perubahan besar pada perilaku model sebenarnya bisa direpresentasikan oleh sekumpulan angka yang sangat sedikit, asalkan angka-angka tersebut diletakkan di tempat yang tepat secara matematis.

Inilah dasar dari LoRA.

---

## 3.2.1 Matematika LoRA: Matriks A dan B

Dalam fine-tuning standar, kita ingin memperbarui matriks bobot $W$ (misal ukuran $1000 \times 1000$). Perubahannya disebut $\Delta W$.

$$W_{baru} = W_{lama} + \Delta W$$

Masalahnya, $\Delta W$ sama besarnya dengan $W_{lama}$ (1 juta parameter). Solusinya adalah LoRA.

LoRA memecah (mendekomposisi) matriks perubahan $\Delta W$ menjadi perkalian dua matriks yang jauh lebih kecil, $A$ dan $B$.

$$\Delta W = B \times A$$

*   **Matriks B:** Ukuran $1000 \times r$ (Tinggi, Kurus)
*   **Matriks A:** Ukuran $r \times 1000$ (Pendek, Lebar)
*   **Rank ($r$):** Angka yang sangat kecil (misal 8 atau 16).

**Penghematan Masif:** Jika $r=8$ maka:
*   Parameter di $A: 8 \times 1000 = 8000$
*   Parameter di $B: 1000 \times 8 = 8000$
*   Total = **16.000** parameter.

Bandingkan dengan $\Delta W$ asli yang butuh **1.000.000** parameter. Penghematannya luar biasa, namun hasil perkalian $BxA$ tetap menghasilkan matriks $1000 \times 1000$ yang bisa dijumlahkan ke model asli.

![Arsitektur LoRA Adapter](https://lms.sdmdigital.id/pluginfile.php/635442/mod_page/content/26/LoRA%20adapter.png)
*Gambar 3. Arsitektur LoRA Adapter*

**Gambar di atas** memvisualisasikan mekanisme "jalur samping" (bypass) yang menjadi inti dari LoRA.

1.  **Jalur Kiri (Frozen Backbone):** Kotak besar berwarna biru merepresentasikan bobot asli model ($W$). Jalur ini **dibekukan** (ikon gembok), artinya bobot ini tidak berubah sama sekali selama pelatihan. Tugasnya adalah mempertahankan pengetahuan lama model.
2.  **Jalur Kanan (Trainable Adapter):** Input yang sama disalin ke jalur kanan, melewati dua matriks kecil berwarna merah:
    *   **Matriks A:** Mengompresi input ke dimensi rendah ($r$).
    *   **Matriks B:** Mengembalikan dimensi ke ukuran asli.
    Kedua matriks inilah yang dilatih untuk mempelajari "perubahan" atau adaptasi baru.
3.  **Penjumlahan (+):** Di akhir, hasil dari jalur asli dan jalur adapter dijumlahkan. Ini berarti *output* model adalah kombinasi dari pengetahuan lama ditambah adaptasi baru.

Meskipun LoRA berhasil memangkas kebutuhan memori untuk *gradient* dan *optimizer* secara drastis (karena parameter yang dilatih sangat sedikit), kita masih menghadapi satu rintangan besar: Model Asli $W$ itu sendiri.

Untuk menjalankan proses pelatihan, kita tetap harus memuat seluruh bobot model dasar (Jalur Kiri pada Gambar 2) ke dalam VRAM GPU. Untuk model 7B dalam presisi standar (16-bit), ini saja sudah memakan sekitar 14-15 GB VRAM, nyaris memenuhi kapasitas GPU gratisan (T4 16GB) bahkan sebelum pelatihan dimulai.

Bagaimana jika kita bisa mengecilkan ukuran model dasar tersebut tanpa kehilangan kinerjanya secara signifikan? Di sinilah inovasi selanjutnya, **QLoRA**, mengambil peran.

---

## 3.2.2 QLoRA: Menembus Batas Memori dengan Kuantisasi

LoRA menghemat memori untuk *gradien* (proses belajar). Tapi, untuk memulai, kita tetap harus memuat model dasar atau $W_{lama}$ ke VRAM.

*   Model 7B dalam 16-bit (float16) = **~14 GB VRAM**.
*   Ini seringkali masih terlalu besar untuk GPU gratis (T4 16GB) jika ditambah overhead sistem.

Di sinilah **QLoRA (Quantized LoRA)** berperan.

**Apa itu QLoRA?**

QLoRA adalah teknik yang menggabungkan LoRA dengan Kuantisasi 4-bit yang ekstrem pada model dasar. Model dimuat dalam presisi 4-bit (sangat padat), sementara adapter LoRA tetap dilatih dalam presisi tinggi.

**Inovasi Kunci (Dettmers et al., 2023):**

1.  **4-bit NormalFloat (NF4):** Tipe data khusus yang dioptimalkan untuk distribusi bobot neural network. Ini jauh lebih akurat daripada 4-bit integer biasa.
2.  **Double Quantization:** Mengompresi parameter kuantisasi itu sendiri untuk menghemat ekstra memori.
3.  **Paged Optimizers:** Menggunakan RAM CPU (bukan VRAM GPU) untuk menyimpan *state optimizer* saat terjadi lonjakan memori, mencegah *crash* (OOM).

**Hasilnya:** Model 7B bisa dimuat hanya dengan **~4 GB VRAM**. Sisa VRAM yang banyak bisa digunakan untuk *batch size* yang lebih besar atau *sequence length* yang lebih panjang.

---

### Instalasi Library

```bash
!pip install -U bitsandbytes transformers peft accelerate
```
Perintah ini akan menginstall library yang kita butuhkan untuk kuantisasi model menggunakan QLoRA.

---

### Kode QLoRA

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
import gc


# Fungsi Helper untuk Cek VRAM
def print_gpu_memory():
    if not torch.cuda.is_available():
        print("GPU tidak terdeteksi!")
        return
   
    # Bersihkan cache agar angka akurat
    torch.cuda.empty_cache()
    gc.collect()
   
    # Hitung memori
    allocated = torch.cuda.memory_allocated() / (1024 ** 3) # Konversi ke GB
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)
    print(f"VRAM Terpakai (Allocated): {allocated:.2f} GB")
    print(f"VRAM Dipesan  (Reserved) : {reserved:.2f} GB")


print("--- Status Awal (Sebelum Load Model) ---")
print_gpu_memory()


# --- Konfigurasi ---
model_id = "Qwen/Qwen2.5-7B" # Model 7B (Ukuran asli FP16 biasanya ~15GB)


# Konfigurasi 4-bit (QLoRA)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",      
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)


print(f"\n--- Memuat Model {model_id} dalam 4-bit... ---")
try:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto"
    )
    print("Model berhasil dimuat!")
except Exception as e:
    print(f"Gagal memuat model: {e}")


print("\n--- Status VRAM (Setelah Load 4-bit) ---")
print_gpu_memory()


# --- Konfigurasi LoRA ---
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=16,            
    lora_alpha=32,  
    lora_dropout=0.05,
    # "all-linear" akan menempelkan adapter ke semua lapisan Linear (Q,K,V,O,FFN)
    # Ini standar modern untuk performa maksimal
    target_modules="all-linear"
)


# Terapkan LoRA
model = get_peft_model(model, peft_config)
print("\n--- Statistik Parameter LoRA ---")
model.print_trainable_parameters()
```

---

## 1. Bedah Kode: Konfigurasi Kunci

*   **BitsAndBytesConfig (Kuantisasi):**
    *   `load_in_4bit=True`: Ini instruksi utama. Alih-alih memuat bobot model sebagai angka 16-bit (2 byte), kita memaksanya menjadi 4-bit (0.5 byte). Ini secara teoritis mengecilkan ukuran model hingga **4x lipat**.
    *   `bnb_4bit_quant_type="nf4"`: Kita menggunakan tipe data *NormalFloat 4-bit*. Ini jauh lebih akurat daripada integer biasa karena distribusi bobot *neural network* biasanya mengikuti kurva lonceng (normal), dan NF4 didesain khusus untuk menampung itu.
    *   `bnb_4bit_compute_dtype=torch.float16`: **Trik Sulap**. Model *disimpan* dalam 4-bit (hemat memori), tetapi saat melakukan perhitungan (komputasi), bobot tersebut diubah sementara (*dequantized*) menjadi 16-bit agar akurasinya tetap tinggi.
*   **LoraConfig (Adaptasi):**
    *   `target_modules="all-linear"`: Kita menempelkan *adapter* LoRA pada **semua** lapisan linear di model (Q, K, V, O, Gate, Up, Down). Ini memberikan performa adaptasi terbaik, meskipun jumlah parameternya sedikit lebih banyak (0.5%) dibanding hanya menempel di Q dan V (0.1%).

---

## 2. Analisis Output (Bukti Efisiensi)

**A. Penghematan VRAM yang Masif**

```text
--- Status Awal (Sebelum Load Model) ---
VRAM Terpakai (Allocated): 0.00 GB


--- Status VRAM (Setelah Load 4-bit) ---
VRAM Terpakai (Allocated): 5.18 GB


--- Status VRAM (Jika tidak memakai QLoRA) ---
VRAM Terpakai (Allocated): 15.2 GB
```

**Penjelasan:**
*   **Teori (Tanpa QLoRA):** Model Qwen2.5-7B memiliki sekitar 7.6 Miliar parameter. Jika dimuat standar (FP16 / 2 byte per parameter):
    $$7.6 \times 10^9 \times 2 \text{ bytes} \approx 15.2 \text{ GB}$$
    Ditambah overhead sistem, ini akan langsung memenuhi GPU 16GB.
*   **Praktik (Dengan QLoRA):** Output Anda menunjukkan hanya **5.18 GB**.
    *   Ini adalah penghematan sekitar **66% memori**.
    *   **Kesimpulan:** Kita berhasil memuat model kelas industri (7B) ke dalam GPU yang relatif kecil, menyisakan ruang yang cukup banyak (~10GB di T4) untuk menampung data *batch* dan proses pelatihan.

**B. Efisiensi Parameter (LoRA)**

```text
--- Statistik Parameter LoRA ---
trainable params: 40,370,176 || all params: 7,655,986,688 || trainable%: 0.5273%
```

**Penjelasan:**
*   `all params: 7,655,986,688`: Ini adalah ukuran total "otak" Qwen.
*   `trainable params: 40,370,176`: Ini adalah ukuran "adapter" (lensa tambahan) yang akan kita latih. Hanya sekitar **40 Juta** parameter.
*   `trainable%: 0.5273%`:
    *   Kita hanya perlu menghitung gradien untuk **0.5%** bagian model.
    *   99.5% sisanya (7.6 Miliar parameter) dibekukan (*frozen*). Ini berarti Anda tidak perlu menyimpan *optimizer state* untuk 99.5% model tersebut, yang merupakan sumber utama habisnya memori saat *fine-tuning*.

**Kesimpulan Akhir:** Kombinasi angka **VRAM 5.18 GB** dan **Trainable 0.5%** inilah yang memungkinkan siapa pun dengan GPU *gaming* menengah atau Google Colab gratis sekarang bisa melatih LLM canggih mereka sendiri.

---

### Mini Quiz
*(Iframe: Mini Quiz Placeholder)*

---

> **Refleksi Singkat**
>
> Kita baru saja membuktikan melalui kode bahwa QLoRA dapat memuat model 7B hanya dengan ~5GB VRAM. Sebagai AI Engineer, implikasi apa yang dibawa fakta ini terhadap aksesibilitas pengembangan AI? Jika sebelumnya fine-tuning dimonopoli oleh perusahaan besar dengan server farm mahal, bagaimana QLoRA mengubah peta persaingan bagi developer individu atau startup kecil?