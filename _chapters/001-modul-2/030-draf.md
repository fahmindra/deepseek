---
slug: modul-2-3
title: modul-2-3
---
### 1.3. Pengenalan Pytorch
---

#### **1.3.1 Apa itu Pytorch?**

Pytorch adalah sebuah *library* Python yang menyediakan berbagai macam tipe data dan operasi untuk melakukan operasi pada tensor, mirip seperti Numpy. Pytorch banyak digunakan dalam *Deep Learning* untuk melakukan implementasi model dan melakukan proses *training*, baik oleh akademisi maupun praktisi dalam industri, dan sudah memiliki ekosistem yang lengkap untuk melakukan training hingga skala besar yang memerlukan *distributed computing*, optimasi model untuk *inference*, dan banyak keperluan lainnya dalam mengembangkan model *Deep Learning.*

Pytorch memiliki dua fitur utama, yaitu:
*   Melakukan operasi tensor dengan berbagai macam pilihan *hardware acceleration*, termasuk GPU, dan
*   Sistem *automatic differentiation* untuk membangun dan melatih *Deep Neural Networks*.

*Automatic differentiation* sangat penting dalam *Deep Learning* karena memudahkan proses *backpropagation*. Mirip seperti `ndarray` pada *library Numpy*, Pytorch menggunakan tensor sebagai tipe data utama yang diproses. Pytorch menyediakan berbagai macam operasi seperti *slicing*, *indexing*, operasi matematika, aljabar linier, operasi reduksi, yang semuanya dapat berjalan di CPU maupun GPU.

Pytorch terdiri dari banyak komponen. Beberapa di antaranya yang paling banyak digunakan dijelaskan pada tabel di bawah ini.

**Tabel 2. Beberapa modul utama dalam Pytorch.**

| Komponen | Deskripsi |
| :--- | :--- |
| `torch` | Library tensor yang mirip dengan Numpy yang dilengkapi dengan automatic differentiation dan ekosistem yang lengkap untuk mengembangkan model Deep Learning. |
| `torch.autograd` | Library Pytorch yang berisikan implementasi tape-based autograd system yang memudahkan penghitungan gradien pada operasi pytorch secara otomatis. |
| `torch.nn` | Berisi berbagai macam operasi Deep Learning yang sudah terintegrasi dengan autograd untuk memudahkan membangun sebuah model. |
| `torch.utils` | Berisi fungsi utilitas seperti dataloader yang bertujuan memudahkan pengembangan model. |

**Catatan Notasi dalam Modul ini:**

Ketika memberikan contoh kode python, ada dua jenis format yang akan digunakan, yaitu dalam bentuk kode yang bisa dijalankan maupun melalui interpreter interaktif. Bentuk kode cukup mudah dimengerti, kode yang diberikan bisa dijalankan sendiri setelah melakukan setup untuk instalasi library yang diperlukan dan bisa diakses melalui link github yang diberikan pada unjuk kerja sebagai bagian dari hands-on. Selain menggunakan kode, untuk interaksi yang lebih interaktif, akan digunakan interpreter. Interpreter interaktif bisa diakses dengan menjalankan python di terminal, yang akan memberikan tampilan berikut:

![Tampilan interpreter Python](https://lms.sdmdigital.id/pluginfile.php/635351/mod_page/content/52/image20.png?time=1764658671654)
**Gambar 18.** Tampilan interpreter Python.

Setiap baris perintah akan didahului `>>>` untuk memberikan informasi bahwa baris tersebut berisi perintah yang telah dijalankan. Kode python dapat dimasukkan baris per baris. Untuk melihat nilai dari suatu variabel, bisa menggunakan perintah `print` atau juga bisa dengan langsung mengetik nama variabelnya. Nilai dari variabel akan muncul di bawah perintah sebelumnya.

![Ilustrasi Tensor 2D](https://lms.sdmdigital.id/pluginfile.php/635351/mod_page/content/52/image5.png)
**Gambar 19.** Mengakses nilai variabel pada interpreter interaktif.

Dalam konteks ini, tensor direpresentasikan sebagai matriks dua dimensi berukuran 3x3 yang berisi nilai-nilai seperti terlihat pada ilustrasi di atas. Contoh kode yang ditunjukkan melalui interpreter bisa dijalankan sebagai sebuah script dengan cara melakukan copy setiap baris perintah yang dijalankan. Perlu ingat bahwa berbeda dengan interpreter interaktif, untuk melihat nilai sebuah variabel perlu menggunakan metode `print`.

**Langkah Pertama Menggunakan PyTorch**

Untuk menggunakan Pytorch, sebelumnya perlu di-install dulu librarynya melalui package manager Python yang digunakan.

```bash
// menggunakan uv
uv add torch 
// menggunakan pip
pip install torch
```

Lalu, ketik dan jalankan kode berikut:

```python
import torch
x = torch.tensor([[1, 2], [3, 4]])
print(x)
print(x.dtype, x.device, x.shape)
```

Akan muncul hasil seperti pada gambar berikut, yang menandakan bahwa proses instalasi dasar telah berhasil.

![Output dari program yang dijalankan](https://lms.sdmdigital.id/pluginfile.php/635351/mod_page/content/52/image29.png)
**Gambar 20.** Output dari program yang dijalankan.

**Setup CUDA untuk operasi menggunakan GPU (opsional, hanya untuk yang memiliki GPU Nvidia)**

Untuk menggunakan GPU, perlu dilakukan langkah setup ekstra yaitu melakukan instalasi CUDA. Singkatnya, CUDA merupakan sebuah platform yang menyediakan library untuk melakukan komputasi paralel pada GPU Nvidia. Berikut adalah langkah-langkah untuk melakukan instalasi CUDA:

1.  Pastikan GPU yang dimiliki kompatibel dengan CUDA. Sebagai rule of thumb, GPU yang cukup modern (rilis dalam 5 tahun terakhir) seharusnya mendukung CUDA.
2.  Download versi CUDA yang kompatibel dengan Pytorch dari website resmi untuk developer dari NVIDIA, sesuaikan dengan versi OS yang digunakan. Untuk melihat versi CUDA yang perlu diinstall, silahkan kunjungi [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) untuk melihat versi CUDA yang didukung.
3.  Install CUDA.

Setelah menginstall CUDA, install versi Pytorch yang kompatibel dengan versi CUDA yang diinstall. Sebagai contoh, untuk menginstall pytorch yang kompatibel dengan CUDA 12.6, gunakan perintah:

```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

Setelah selesai, jalankan kode berikut:

```python
import torch
print(torch.cuda.is_available())
```

Apabila instalasi CUDA berhasil dan valid, program akan memberikan output `True`.

---

#### **1.3.2 Implementasi Berbagai Operasi dalam Pytorch**

**Atribut dari Sebuah Tensor**

Setiap tipe objek tensor di Pytorch memiliki tiga atribut utama: **shape**, **device** dan **dtype**.

*   **Shape** adalah atribut yang menunjukkan dimensi dari sebuah tensor.
*   **Device** merupakan atribut yang menunjukkan apakah tensor tersebut disimpan di CPU atau di GPU (`cpu` atau `cuda`). Tensor bisa dipindahkan menggunakan fungsi `.to()`.
*   **Dtype** menunjukkan tipe data numerik (int32, float32, dsb).

```python
import torch

print(torch.cuda.is_available())
# Output: False/True tergantung setup

t1 = torch.Tensor([[1, 2, 3, 4],
                   [5, 6, 7, 8],
                   [9, 10, 11, 12]])

print(t1)
# Output:
# tensor([[ 1.,  2.,  3.,  4.],
#         [ 5.,  6.,  7.,  8.],
#         [ 9., 10., 11., 12.]])

print(t1.shape)
# Output: torch.Size([3, 4])

print(t1.device)
# Output: device(type='cpu')

if torch.cuda.is_available():
    t1_cuda = t1.to('cuda')
    print(t1_cuda.device)
    # Output: device(type='cuda', index=0)

print(t1.dtype)
# Output: torch.float32
```

**Dimensi dan Broadcasting**

Menambahkan dua Tensor hanya bisa dilakukan jika keduanya memiliki shape yang sama. Untuk perkalian matriks ($I \times J$ dikali $K \times L$), $J$ harus sama dengan $K$. Simbol `@` digunakan untuk perkalian matriks.

```python
>>> result = torch.randn((2,3)) + torch.randn((2,3))
>>> result = torch.randn((2,3)) + torch.randn((3,3)) # error (shape mismatch)

>>> result = torch.randn((2,3)) @ torch.randn((3,4))
>>> result.shape
torch.Size([2, 4])
```

**Broadcasting** memungkinkan operasi pada tensor dengan shape berbeda dengan ketentuan tertentu: dimensinya sama, salah satunya adalah 1, atau salah satunya tidak ada.

```python
>>> a = torch.randn((3,4,5)) * torch.randn((2,3,1,5))
>>> a.shape
torch.Size([2, 3, 4, 5])
```

**Inisialisasi tensor**

*   `torch.from_numpy`: Dari array numpy.
*   `torch.zeros`: Tensor nol.
*   `torch.ones_like`: Tensor satu dengan shape yang sama dengan tensor lain.
*   `torch.arange`: Seperti range python.
*   `torch.randn`: Nilai random distribusi normal.

**Indexing dan slicing tensor**

Pytorch menggunakan cara yang sama dengan numpy untuk indexing dan slicing.

```python
>>> tensor = torch.randn((3,4))
>>> tensor[0] # Baris pertama
>>> tensor[:, :1] # Kolom pertama
```

**Menggabungkan tensor**

Gunakan `torch.concat` atau `torch.stack`.

**Manipulasi dimensi tensor**

*   **Squeeze**: Menghilangkan dimensi bernilai 1.
*   **Unsqueeze**: Menambah dimensi bernilai 1.
*   **Flatten**: Meratakan menjadi 1D.
*   **Reshape**: Mengubah shape (total elemen tetap).
*   **Transpose / Permute**: Menukar urutan dimensi.

**Operasi tensor**

*   **Pointwise**: Dilakukan pada setiap elemen (sin, cos, exp, dsb).
*   **Reduction**: Reduksi dimensi (max, mean, std, sum).

Contoh implementasi manual Logistic Regression:

```python
import torch

W = torch.randn(size=(5, 2), dtype=torch.float32)
B = torch.zeros(size=(1, 2), dtype=torch.float32)
X = torch.randn(size=(8, 5), dtype=torch.float32)

y = torch.matmul(X, W) + B
y = 1 / (1 + torch.exp(-1 * y)) # Sigmoid pointwise
y = y.max(dim=1) # Reduksi
```

Bisa juga menggunakan `torch.nn`:

```python
from torch import nn
linear_layer = nn.Linear(5, 2, bias=True)
sigmoid = nn.Sigmoid()
y = sigmoid(linear_layer(X))
```

**Random seeding**
```python
seed = 11563
torch.manual_seed(seed)
```

---

#### **1.3.3 Implementasi Model Sederhana dalam Pytorch**

Sebuah modul dalam Pytorch adalah subclass dari `torch.nn.Module`. Metode `__init__` mendefinisikan komponen, dan `forward()` mendefinisikan alur komputasi.

Contoh Inception Module:
```python
class InceptionModule(nn.Module):
    def __init__(self, n_input_features, n_channels, n_reduces):
        super().__init__()
        self._1x1_conv = nn.Conv2d(n_input_features, n_channels["1x1"], kernel_size=1)
        # ... (komponen lainnya)

    def forward(self, input_tensor):
        # ... (logika forward)
        return torch.concat([res1, res2, res3, res4], dim=1)
```

Untuk urutan sekuensial, gunakan `torch.nn.Sequential`.

---

#### **1.3.4 Implementasi Training dengan Pytorch**

**Data loaders**

Dataloader membantu memuat data secara efisien menggunakan fitur *Prefetching* dan *Parallel loading* untuk mencegah GPU idle.

![Ilustrasi proses loading data](https://lms.sdmdigital.id/pluginfile.php/635351/mod_page/content/52/image25.jpg?time=1768357942631)
**Gambar 21.** Ilustrasi proses loading data yang sekuensial (CPU vs GPU).

Implementasi Dataset memerlukan: `__init__`, `__len__`, dan `__getitem__`.

**Loss Function**

Tersedia di `torch.nn` (MSELoss, CrossEntropyLoss, dsb) atau bisa dibuat secara custom.

**Backpropagation dan Autograd**

Training meminimalkan loss dengan menghitung gradien. Panggil `.backward()` pada tensor loss untuk memicu kalkulasi gradien otomatis melalui *dynamic computation graph*. Hanya tensor dengan `requires_grad=True` yang gradiennya dihitung.

![Ilustrasi graf komputasi](https://lms.sdmdigital.id/pluginfile.php/635351/mod_page/content/52/image3.png)
**Gambar 22.** Ilustrasi graf komputasi.

**Optimizer**

Gunakan `torch.optim` (SGD, Adam, dsb).
Langkah umum training:
1.  Forward pass.
2.  `optimizer.zero_grad()`.
3.  `loss.backward()`.
4.  `optimizer.step()`.

**Metrics**

Gunakan library seperti `scikit-learn` atau `TorchMetrics`. Gunakan `.detach().numpy()` untuk mengubah tensor ke array numpy.

**Serialisasi (Menyimpan Model)**

Gunakan `torch.save` dan `torch.load` pada `state_dict()` model.

```python
torch.save(model.state_dict(), "model.pt")
# Memuat kembali
model.load_state_dict(torch.load("model.pt"))
```

---

#### **1.3.5 Konsep Lanjutan Pytorch**

*   **Distributed Training**: Menggunakan `torch.distributed` untuk Data Parallelism atau Model Parallelism di banyak GPU.
*   **torch.compile**: Mengoptimalkan graf komputasi menjadi kernel low-level yang lebih cepat.
*   **torch.export**: Mengonversi model menjadi *Intermediate Representation* (IR) untuk deployment (misal ke format ONNX).
*   **Profiling**: Menggunakan Pytorch Profiler untuk memeriksa latency dan penggunaan memori.

---

#### **Mini Quiz**

*Silakan akses quiz interaktif melalui platform LMS:*
1. [Quiz - Dasar Pytorch](https://app.lumi.education/api/v1/run/_XbKyo/embed)
2. [Quiz - Operasi & Training](https://app.lumi.education/api/v1/run/G7d-uE/embed)

---

> **Refleksi / Bahan Diskusi Pribadi**
> *   Ada banyak library *Deep Learning* selain Pytorch seperti Tensorflow, Keras, JAX dan lain sebagainya. Apa yang membedakan masing-masing library ini?
> *   Untuk lebih mengapresiasi kemudahan yang diberikan library *Deep Learning*, coba pelajari bagaimana caranya menggunakan *device* seperti GPU untuk melakukan komputasi secara manual (tanpa library). Bagaimana caranya library *deep learning* mengintegrasikan berbagai macam *device* ini?