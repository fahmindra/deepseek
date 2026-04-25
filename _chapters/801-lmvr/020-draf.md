---
slug: lmvr-2
title: lmvr-2
---

# Bab 2: Landasan Matematika untuk LLM

> **"Masa depan AI bergantung pada perpaduan algoritma yang kuat dan pemahaman matematika yang mendasarinya." — Andrew Ng**

Bab 2 dari LMVR meletakkan dasar matematika yang esensial untuk memahami dan mengimplementasikan model bahasa besar (LLM). Bab ini menggali area kritis aljabar linear, probabilitas dan statistik, kalkulus dan optimasi, teori informasi, transformasi linear, dan matematika diskrit. Ini mencakup konsep-konsep fundamental seperti vektor, matriks, distribusi probabilitas, gradien, dan entropi, yang berkembang ke topik yang lebih lanjut seperti dekomposisi eigen, analisis komponen utama (PCA), dan teori graf. Prinsip-prinsip matematika ini dikontekstualisasikan dalam Rust, menekankan pada implementasi praktis algoritma, teknik optimasi, dan struktur data yang esensial untuk pengembangan LLM yang efisien. Dengan mengintegrasikan teori dengan praktik pengkodean langsung, bab ini membekali pembaca dengan pemahaman yang kuat, komprehensif, dan tepat tentang landasan matematika yang diperlukan untuk LLM.

## 2.1. Aljabar Linear dan Ruang Vektor

Aljabar linear memainkan peran sentral dalam desain dan fungsi Model Bahasa Besar (LLM), yang mendasari matematika tentang bagaimana data direpresentasikan dan ditransformasikan di dalam model-model ini. Inti dari fondasi ini adalah skalar, vektor, matriks, dan tensor, yang merupakan blok pembangun manipulasi data. Skalar adalah nilai numerik tunggal, sedangkan vektor adalah array angka satu dimensi. Ketika Anda memperluas konsep ini ke dua dimensi, Anda memiliki matriks, dan tensor menggeneralisasi ini ke dimensi yang lebih tinggi lagi. Dalam konteks LLM, tensor sangat penting karena mewakili data multi-dimensi yang diperlukan untuk memodelkan bahasa, di mana setiap dimensi mungkin mengkodekan properti yang berbeda, seperti penyematan kata (word embeddings), struktur kalimat, atau bahkan seluruh urutan teks.

Dalam bahasa aljabar linear, ruang vektor menyediakan kerangka kerja formal untuk memahami bagaimana titik data (vektor) disusun dan ditransformasikan. Sebuah ruang vektor terdiri dari kumpulan vektor yang dapat diskalakan dan ditambahkan bersama untuk membentuk vektor baru. Di dalamnya, subruang adalah ruang vektor yang lebih kecil yang terkandung dalam ruang yang lebih besar, dan konsep ini penting saat mengurangi dimensionalitas dari kumpulan data yang besar. Misalnya, saat melatih LLM, subruang dari kata-kata yang sering digunakan mungkin lebih kritis untuk mekanisme atensi daripada seluruh ruang vektor. Basis dari ruang vektor adalah set minimal vektor yang darinya semua vektor dalam ruang tersebut dapat dibentuk melalui kombinasi linear. Dimensi dari ruang vektor adalah jumlah vektor dalam basis ini, dan untuk LLM, ini bisa berhubungan dengan jumlah token unik (kata atau subkata) yang diproses oleh model.

![Figure 1: Ilustrasi ruang vektor untuk NLP.](https://lmvr.rantai.dev//images/L6A504iArDGBvUq8WSeO-jwJbCh3P4pGx2LB9YHme-v1.jpeg)
**Gambar 1:** Ilustrasi ruang vektor untuk NLP.

Properti penting dari ruang vektor, seperti independensi linear dan rentang (span), secara langsung memengaruhi seberapa baik LLM dapat mempelajari hubungan yang bermakna antar kata. Sekumpulan vektor bersifat independen secara linear jika tidak ada vektor dalam set tersebut yang dapat ditulis sebagai kombinasi dari yang lain, yang berarti setiap vektor menambahkan informasi unik. Ini adalah properti yang diinginkan dalam penyematan kata, di mana vektor setiap kata harus menangkap informasi semantik yang berbeda. Rentang dari sekumpulan vektor mewakili semua vektor yang mungkin terbentuk dengan menggabungkan set tersebut secara linear. Dalam aplikasi praktis, konsep ini esensial saat memodelkan korpus teks yang besar, di mana rentang penyematan kata harus mencakup hubungan semantik antar kata dalam bahasa tersebut.

Operasi matriks adalah manipulasi matematika yang menggerakkan komputasi LLM. Penjumlahan dan perkalian matriks bersifat fundamental saat mentransformasikan data antar lapisan dalam sebuah model. Misalnya, diberikan dua matriks $A$ dan $B$, penjumlahan matriks menghasilkan matriks baru $C$ di mana setiap elemen $C_{ij} = A_{ij} + B_{ij}$, dan perkalian diberikan oleh $C_{ij} = \sum_k A_{ik} B_{kj}$, yang digunakan secara luas dalam mentransformasikan data masukan melalui berbagai lapisan dalam jaringan saraf. Operasi seperti transpose matriks, di mana baris dan kolom ditukar, sangat penting untuk operasi seperti atensi dot-product dalam model Transformer. Memahami bagaimana matriks berinteraksi sangat penting untuk memahami bagaimana data mengalir melalui model, terutama saat bekerja dengan arsitektur kompleks seperti Transformer dalam LLM.

Matriks juga memiliki properti seperti invers dan determinan, yang fundamental dalam menyelesaikan sistem persamaan linear, meskipun penggunaannya dalam LLM lebih abstrak. Invers dari matriks $A$ adalah matriks lain $A^{-1}$ sedemikian rupa sehingga $A A^{-1} = I$ (di mana $I$ adalah matriks identitas), dan determinan memberikan wawasan tentang properti matriks, seperti apakah matriks tersebut dapat diinversi. Yang lebih penting bagi LLM, konsep nilai eigen (eigenvalues) dan vektor eigen (eigenvectors) menyediakan basis matematika untuk teknik reduksi dimensionalitas, seperti Principal Component Analysis (PCA), yang digunakan dalam langkah pemrosesan awal untuk mengurangi kompleksitas penyematan kata atau fitur lainnya. Secara formal, jika matriks $A$ bekerja pada vektor $v$, vektor tersebut hanya akan diskalakan oleh skalar $\lambda$, yang berarti $A v = \lambda v$. Skalar $\lambda$ adalah nilai eigen, dan vektor $v$ adalah vektor eigen. Prinsip ini membantu dalam mengekstraksi arah (atau komponen) yang paling signifikan dalam kumpulan data, yang secara langsung membantu kinerja LLM dengan berfokus pada fitur yang paling penting.

Konsep penting lainnya dalam LLM adalah ortogonalisasi dan basis ortonormal. Dua vektor dikatakan ortogonal jika hasil kali titik (dot product) mereka adalah nol, dan basis ortogonal memungkinkan transformasi dan penyederhanaan yang mudah dalam komputasi matriks. Basis ortonormal adalah sekumpulan vektor ortogonal yang juga memiliki panjang unit. Dalam praktiknya, banyak algoritma seperti Gram-Schmidt digunakan untuk mengortogonalkan vektor, menyederhanakan transformasi dalam komputasi LLM, terutama dalam metode seperti dekomposisi nilai singular (SVD), yang digunakan dalam reduksi dimensionalitas.

Operasi matriks sangat mendasar bagi implementasi model bahasa besar (LLM), di mana tugas-tugas seperti mekanisme atensi, penyematan token, dan backpropagation sangat bergantung pada penanganan array multi-dimensi yang efisien.

### Contoh Python: Operasi Aljabar Linear dengan NumPy

```python
import numpy as np

# Contoh perkalian Matriks 2x2 dan Vektor 2x1
a = np.array([[1.0, 2.0], [3.0, 4.0]])
b = np.array([5.0, 6.0])
result = np.dot(a, b)
print(f"Hasil perkalian matriks-vektor: {result}")

# Contoh perkalian Matriks 3x3 dan Matriks 3x2
c = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
d = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
result_matrix = np.dot(c, d)
print(f"Hasil perkalian matriks-matriks (3x3 * 3x2):\n{result_matrix}")

# Contoh transpose matriks
transpose = c.T
print(f"Matriks transpose:\n{transpose}")

# Perkalian elemen-demi-elemen (Hadamard product)
e = np.array([[1.0, 2.0], [3.0, 4.0]])
f = np.array([[0.5, 0.2], [0.3, 0.1]])
hadamard_product = e * f
print(f"Perkalian elemen-demi-elemen (Hadamard product):\n{hadamard_product}")

# Contoh invers matriks
try:
    inverse = np.linalg.inv(e)
    print(f"Invers dari matriks e:\n{inverse}")
except np.linalg.LinAlgError:
    print("Matriks e tidak dapat diinversi.")

# Contoh dot product vektor
g = np.array([1.0, 2.0])
h = np.array([3.0, 4.0])
dot_product = np.dot(g, h)
print(f"Dot product vektor g dan h: {dot_product}")

# Perkalian matriks dinamis besar
matrix1 = np.arange(1, 17).reshape(4, 4).astype(float)
matrix2 = np.eye(4)
large_result = np.dot(matrix1, matrix2)
print(f"Hasil perkalian matriks dinamis besar:\n{large_result}")
```

Kode di atas mendemonstrasikan berbagai operasi matriks menggunakan pustaka `numpy`. Operasi ini membentuk tulang punggung komputasi dalam arsitektur LLM seperti Transformer, yang memungkinkan model untuk memproses dan belajar dari kumpulan data teks yang sangat besar.

Salah satu tantangan terbesar dalam bekerja dengan model besar adalah manajemen memori yang efisien. Saat menangani matriks dan tensor dengan ukuran yang sangat besar, seperti yang ditemui dalam LLM, penting untuk mengoptimalkan bagaimana memori dialokasikan dan didealokasikan. Kemampuan Rust untuk mengelola memori tanpa garbage collector berarti operasi matriks besar dapat dilakukan tanpa penundaan yang tidak terduga, menjadikannya pilihan tepat untuk mengimplementasikan LLM di mana performa sangat kritis.

Sebagai kesimpulan, aljabar linear bukan sekadar landasan teoretis tetapi alat praktis yang menggerakkan operasi inti LLM. Dari ruang vektor hingga operasi matriks, fitur tingkat sistem dan pustaka yang kuat memungkinkan pengembang untuk secara efisien mengimplementasikan dan mengoptimalkan operasi ini untuk aplikasi dunia nyata dalam LLM.

## 2.2. Probabilitas dan Statistik

Dalam dunia LLM, probabilitas dan statistik memainkan peran esensial dalam memahami dan memodelkan ketidakpastian serta keacakan yang melekat dalam bahasa alami. Landasan ini dimulai dengan teori probabilitas, yang berkaitan dengan studi variabel acak dan kemungkinan hasil yang berbeda. Variabel acak adalah variabel yang nilainya merupakan hasil dari fenomena acak, dan dapat bersifat diskrit (mengambil nilai tertentu seperti hasil lemparan dadu) atau kontinu (mengambil nilai apa pun dalam rentang tertentu, seperti suhu). Distribusi probabilitas menjelaskan seberapa besar kemungkinan variabel acak mengambil nilai tertentu. Konsep penting seperti ekspektasi (nilai rata-rata atau yang diharapkan dari variabel acak) dan varians (ukuran seberapa tersebar nilai-nilai variabel acak) sangat penting untuk memahami bagaimana model memproses dan menghasilkan bahasa dalam dunia yang tidak pasti.

![Figure 2: Peran probabilitas dan statistik dalam LLM.](https://lmvr.rantai.dev//images/L6A504iArDGBvUq8WSeO-9SuVOjzqbJWbcs1GsWHq-v1.png)
**Gambar 2:** Peran probabilitas dan statistik dalam LLM.

Beberapa distribusi probabilitas penting yang umum digunakan dalam pengembangan LLM meliputi Distribusi Gaussian (distribusi normal) untuk memodelkan variabel kontinu, distribusi Bernoulli untuk hasil biner, dan distribusi Binomial untuk beberapa percobaan. Demikian pula, distribusi Multinomial memperluas kasus Binomial ke skenario dengan lebih dari dua hasil yang mungkin, yang sangat berguna dalam model bahasa saat berurusan dengan prediksi kata. Terakhir, distribusi Poisson digunakan untuk memodelkan jumlah peristiwa yang terjadi dalam interval waktu atau ruang yang tetap, yang relevan dalam tugas NLP seperti memodelkan frekuensi kemunculan kata.

Inferensi Bayesian adalah konsep kunci lainnya dalam LLM, yang menyediakan kerangka kerja untuk memperbarui probabilitas hipotesis seiring tersedianya lebih banyak bukti. Secara matematis, ini direpresentasikan menggunakan Teorema Bayes:

$$ P(A | B) = \frac{P(B | A) \cdot P(A)}{P(B)} $$

di mana $P(A | B)$ adalah probabilitas posterior, $P(B | A)$ adalah likelihood, $P(A)$ adalah probabilitas prior, dan $P(B)$ adalah likelihood marginal. Dalam LLM, inferensi Bayesian digunakan untuk meningkatkan prediksi model dengan memasukkan pengetahuan awal atau ketidakpastian ke dalam proses pengambilan keputusan model.

Probabilitas bersyarat $P(A|B)$ mewakili probabilitas terjadinya peristiwa A mengingat peristiwa B telah terjadi. Memahami ketergantungan antar kata dalam sebuah kalimat dapat dibingkai dalam probabilitas bersyarat. Pengujian hipotesis digunakan untuk menentukan apakah suatu hasil signifikan secara statistik, yang relevan saat membandingkan model yang berbeda atau menilai kinerja di bawah kondisi eksperimental yang berbeda.

Dalam LLM, konsep probabilitas adalah kunci untuk tugas-tugas seperti menghasilkan teks, pengambilan sampel dari logit, dan regularisasi model dengan noise. Kode Python berikut menggunakan `numpy` dan `scipy` untuk mensimulasikan berbagai distribusi dan operasi probabilitas.

### Contoh Python: Distribusi Probabilitas dan Sampling

```python
import numpy as np
from scipy.special import softmax as scipy_softmax

# Implementasi fungsi Softmax
def manual_softmax(logits):
    exps = np.exp(logits - np.max(logits)) # Stabilitas numerik
    return exps / np.sum(exps)

def main():
    # Contoh 1: Distribusi Gaussian (Normal)
    mean = 0.0
    std_dev = 1.0
    gaussian_samples = np.random.normal(mean, std_dev, 10)
    print("Sampel Distribusi Gaussian:")
    print(gaussian_samples)

    # Contoh 2: Distribusi Bernoulli
    p_success = 0.7
    bernoulli_samples = np.random.binomial(1, p_success, 10)
    print(f"\nSampel Distribusi Bernoulli (p = {p_success}):")
    print(bernoulli_samples)

    # Contoh 3: Distribusi Multinomial (untuk prediksi kata)
    probabilities = [0.2, 0.5, 0.3]
    multinomial_sample = np.random.multinomial(1, probabilities, 10)
    print("\nSampel Distribusi Multinomial (untuk probabilitas prediksi kata):")
    print(multinomial_sample)

    # Contoh 4: Sampling berbasis Softmax untuk LLM
    logits = np.array([2.0, 1.0, 0.1])
    probs = manual_softmax(logits)
    print(f"\nProbabilitas Softmax dari Logit {logits}:")
    print(probs)
    
    # Mengambil sampel berdasarkan probabilitas softmax
    word_indices = np.arange(len(logits))
    sampled_indices = [np.random.choice(word_indices, p=probs) for _ in range(5)]
    print(f"Indeks yang disampel berdasarkan probabilitas softmax: {sampled_indices}")

    # Contoh 5: Estimasi Monte Carlo menggunakan distribusi Gaussian
    num_samples = 1000
    mc_samples = np.random.normal(0, 1, num_samples)
    mc_estimate = np.mean(mc_samples)
    print(f"\nEstimasi Monte Carlo dari Ekspektasi (Gaussian): {mc_estimate}")

if __name__ == "__main__":
    main()
```

Kode di atas mendemonstrasikan berbagai distribusi probabilitas dan teknik pengambilan sampel yang relevan dengan LLM. Ini sangat penting untuk memahami ketidakpastian, melakukan prediksi, dan mengambil sampel secara efisien dalam tugas-tugas NLP.

Tren terbaru dalam penalaran probabilistik dalam LLM mencakup integrasi estimasi ketidakpastian dalam prediksi model. Alat-alat seperti Optimasi Bayesian semakin banyak digunakan dalam menyetel hyperparameter LLM, yang memungkinkan model dioptimalkan secara lebih efisien.

## 2.3. Kalkulus dan Optimasi

Dalam pengembangan dan pelatihan Model Bahasa Besar (LLM), kalkulus sangat mendasar, terutama dalam konteks optimasi. Diferensiasi dan integrasi berfungsi sebagai tulang punggung matematika tentang bagaimana model belajar dengan menyesuaikan parameter mereka untuk meminimalkan fungsi kerugian (loss function). Dalam pembelajaran mesin, kita terutama berurusan dengan diferensiasi, yang membantu memahami bagaimana perubahan kecil dalam parameter model memengaruhi output.

![Figure 3: Siklus optimasi dalam melatih LLM.](https://lmvr.rantai.dev//images/L6A504iArDGBvUq8WSeO-HAkqlPnDmnvkCgALfKs7-v1.png)
**Gambar 3:** Siklus optimasi dalam melatih LLM.

Salah satu konsep terpenting adalah turunan parsial, yang merupakan turunan terhadap satu variabel sementara menjaga variabel lainnya tetap konstan. Gradien, yang merupakan vektor dari turunan parsial, mewakili arah pendakian tercuram untuk fungsi multivariabel. Aturan rantai (chain rule) sangat penting dalam model yang dalam, memungkinkan kita menghitung turunan dari fungsi komposit dalam algoritma backpropagation:

$$\frac{d}{dx} f(g(x)) = f'(g(x)) \cdot g'(x)$$

Optimasi adalah jantung dari pelatihan LLM, dan metode yang paling penting adalah penurunan gradien (gradient descent). Dalam algoritma ini, kita menggunakan gradien dari fungsi kerugian untuk memperbarui parameter model secara iteratif:

$$ \theta := \theta - \eta \cdot \nabla_\theta L(\theta) $$

di mana $\eta$ adalah laju pembelajaran (learning rate). Varian seperti Adam (Adaptive Moment Estimation) sering digunakan dalam LLM karena kemampuannya menangani lanskap optimasi non-konveks yang kompleks. Pengali Lagrange (Lagrange multipliers) menyediakan cara untuk melakukan optimasi dengan kendala, sementara Jacobian dan Hessian adalah representasi matriks dari turunan pertama dan kedua yang memandu algoritma optimasi.

![Figure 4: Animasi 5 metode penurunan gradien: SGD (cyan), Momentum (magenta), AdaGrad (putih), RMSProp (hijau), Adam (biru). Sumur kiri adalah minimum global; sumur kanan adalah minimum lokal.](https://lmvr.rantai.dev//images/L6A504iArDGBvUq8WSeO-UA416PCEldbMPUIr2kvE-v1.gif)
**Gambar 4:** Animasi 5 metode penurunan gradien pada sebuah permukaan.

Berikut adalah simulasi sederhana dari berbagai algoritma optimasi menggunakan Python.

### Contoh Python: Benchmarking Optimizers

```python
import numpy as np
import time

class Optimizers:
    @staticmethod
    def sgd(params, grad, lr):
        return params - lr * grad

    @staticmethod
    def momentum(params, grad, lr, velocity, beta=0.9):
        velocity = beta * velocity + (1 - beta) * grad
        params = params - lr * velocity
        return params, velocity

    @staticmethod
    def adagrad(params, grad, lr, accum, eps=1e-8):
        accum += grad**2
        params = params - (lr / (np.sqrt(accum) + eps)) * grad
        return params, accum

    @staticmethod
    def rmsprop(params, grad, lr, accum, beta=0.9, eps=1e-8):
        accum = beta * accum + (1 - beta) * (grad**2)
        params = params - (lr / (np.sqrt(accum) + eps)) * grad
        return params, accum

    @staticmethod
    def adam(params, grad, lr, m, v, t, beta1=0.9, beta2=0.999, eps=1e-8):
        t += 1
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad**2)
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        params = params - (lr / (np.sqrt(v_hat) + eps)) * m_hat
        return params, m, v, t

def loss_grad(params):
    return params - 0.5

def benchmark_optimizers():
    param_size = 1000000
    num_iterations = 100
    lr = 0.01
    initial_params = np.ones(param_size)

    # SGD
    params = initial_params.copy()
    start = time.time()
    for _ in range(num_iterations):
        grad = loss_grad(params)
        params = Optimizers.sgd(params, grad, lr)
    print(f"SGD duration: {time.time() - start:.4f}s")

    # Momentum
    params = initial_params.copy()
    velocity = np.zeros(param_size)
    start = time.time()
    for _ in range(num_iterations):
        grad = loss_grad(params)
        params, velocity = Optimizers.momentum(params, grad, lr, velocity)
    print(f"Momentum duration: {time.time() - start:.4f}s")

    # Adam
    params = initial_params.copy()
    m = np.zeros(param_size)
    v = np.zeros(param_size)
    t = 0
    start = time.time()
    for _ in range(num_iterations):
        grad = loss_grad(params)
        params, m, v, t = Optimizers.adam(params, grad, lr, m, v, t)
    print(f"Adam duration: {time.time() - start:.4f}s")

if __name__ == "__main__":
    benchmark_optimizers()
```

Masing-masing optimizer ini memiliki kelebihan dan kekurangannya sendiri. Adam adalah yang paling banyak digunakan dalam LLM skala besar karena kemampuannya menangani lanskap optimasi non-konveks yang kompleks secara efisien.

## 2.4. Teori Informasi

Teori informasi menyediakan kerangka kerja matematika kritis untuk memahami dan mengukur ketidakpastian, kompresi, dan transmisi informasi. Konsep entropi adalah pusat dari teori informasi, yang mengukur ketidakpastian atau keacakan dalam variabel acak:

$$ H(X) = -\sum_{x \in X} p(x) \log p(x) $$

Dalam pelatihan model bahasa, entropi silang (cross-entropy) adalah salah satu fungsi kerugian yang paling banyak digunakan. Ini mengukur perbedaan antara dua distribusi probabilitas: distribusi sebenarnya (label) dan distribusi yang diprediksi (output model):

$$ H(p, q) = -\sum_{i} p(y_i) \log q(y_i) $$

Divergensi Kullback-Leibler (KL) adalah konsep penting lainnya yang mengukur bagaimana satu distribusi probabilitas menyimpang dari distribusi probabilitas referensi:

$$ D_{KL}(P || Q) = \sum_{x} P(x) \log \frac{P(x)}{Q(x)} $$

Divergensi KL sering digunakan dalam Variational Autoencoders (VAEs) dan dalam penyetelan halus LLM dengan Reinforcement Learning from Human Feedback (RLHF) untuk memastikan bahwa model yang diperbarui tidak menyimpang terlalu jauh dari model dasar.

### Contoh Python: Cross-Entropy dan KL Divergence

```python
import numpy as np

def log_softmax(x):
    e_x = np.exp(x - np.max(x))
    return np.log(e_x / e_x.sum())

def cross_entropy_loss(y_true, y_pred_logits):
    """
    y_true: One-hot encoded (n_samples, n_classes)
    y_pred_logits: Raw logits (n_samples, n_classes)
    """
    loss = 0.0
    for i in range(len(y_true)):
        log_probs = log_softmax(y_pred_logits[i])
        loss += -np.dot(y_true[i], log_probs)
    return loss / len(y_true)

def kl_divergence(p, q):
    """
    P dan Q adalah distribusi probabilitas
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

def main():
    # Contoh Cross-entropy
    y_true = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    y_pred_logits = np.array([[2.0, 0.5, 0.1], [1.0, 3.0, 0.2]])
    
    loss = cross_entropy_loss(y_true, y_pred_logits)
    print(f"Cross-entropy loss: {loss:.4f}")

    # Contoh KL Divergence
    p = [0.6, 0.3, 0.1]
    q = [0.5, 0.4, 0.1]
    kl_div = kl_divergence(p, q)
    print(f"KL divergence: {kl_div:.4f}")

if __name__ == "__main__":
    main()
```

Teori informasi juga mengarah pada teknik seperti regularisasi entropi untuk mendorong eksplorasi dalam prediksi model, mencegah model menjadi terlalu percaya diri secara palsu.

## 2.5. Transformasi Linear dan Dekomposisi Eigen

Transformasi linear memainkan peran penting dalam bagaimana data input dimanipulasi di dalam berbagai lapisan model. Dalam jaringan saraf, bobot di setiap lapisan bertindak sebagai transformasi linear yang diterapkan pada vektor input.

Dekomposisi matriks seperti LU, QR, dan Cholesky sangat penting untuk efisiensi operasi matriks skala besar. Dekomposisi nilai eigen (Eigen decomposition) membantu dalam tugas-tugas seperti reduksi dimensionalitas, menangkap arah varians maksimal dalam data. Dekomposisi Nilai Singular (SVD) adalah teknik yang lebih kuat yang menggeneralisasi dekomposisi eigen ke semua matriks: $A = U \Sigma V^T$. SVD sangat penting dalam Analisis Semantik Laten (LSA) dan Principal Component Analysis (PCA).

![Figure 5: Ilustrasi metode PCA untuk reduksi dimensionalitas.](https://lmvr.rantai.dev//images/L6A504iArDGBvUq8WSeO-28fwUEkC1vJukNX5t4aW-v1.png)
**Gambar 5:** Ilustrasi metode PCA untuk reduksi dimensionalitas.

### Contoh Python: PCA menggunakan Eigen Decomposition

```python
import numpy as np

def pca(data, n_components):
    # Centering data
    mean = np.mean(data, axis=0)
    centered_data = data - mean
    
    # Matriks kovarians
    covariance_matrix = np.cov(centered_data, rowvar=False)
    
    # Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    
    # Urutkan eigenvalues dan eigenvectors dari besar ke kecil
    sorted_index = np.argsort(eigenvalues)[::-1]
    sorted_eigenvectors = eigenvectors[:, sorted_index]
    
    # Pilih n komponen teratas
    eigenvector_subset = sorted_eigenvectors[:, :n_components]
    
    # Proyeksi data
    reduced_data = np.dot(centered_data, eigenvector_subset)
    return reduced_data, eigenvalues[sorted_index]

def main():
    # Data contoh: 4 sampel dengan 3 fitur
    data = np.array([
        [4.0, 2.0, 1.0],
        [3.0, 1.0, 2.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 3.0]
    ])

    reduced_data, eigenvalues = pca(data, 2)
    print("Data yang direduksi (2 komponen):")
    print(reduced_data)
    print("\nNilai Eigen:")
    print(eigenvalues)

if __name__ == "__main__":
    main()
```

Pemahaman tentang struktur nilai eigen dari matriks Hessian dapat membantu meningkatkan konvergensi dan mencegah ketidakstabilan selama pelatihan, terutama dalam masalah optimasi non-konveks yang umum terjadi pada LLM.

## 2.6. Matematika Diskrit

Matematika diskrit berurusan dengan nilai-nilai yang terpisah dan berbeda, menyediakan alat untuk memahami struktur seperti graf, set, dan penalaran logika.

Teori graf digunakan untuk memodelkan hubungan antar entitas, misalnya dalam graf pengetahuan (knowledge graphs). Graf pengetahuan membantu LLM dalam menjawab pertanyaan dan memberikan output yang lebih akurat secara kontekstual. Teori set dan aljabar Boolean menjadi dasar untuk pengambilan keputusan logis dalam sistem berbasis aturan.

Kombinatorika memainkan peran penting dalam tugas-tugas seperti tokenisasi. Byte Pair Encoding (BPE) adalah metode tokenisasi subkata yang menggunakan teknik kombinatorial untuk menangani kosakata besar secara efisien. BPE dimulai dengan memecah kata menjadi karakter individu dan secara iteratif menggabungkan pasangan karakter atau subkata yang paling sering muncul.

### Contoh Python: Byte Pair Encoding (BPE) Sederhana

```python
import re
from collections import Counter, defaultdict

def get_stats(vocab):
    pairs = defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[symbols[i], symbols[i+1]] += freq
    return pairs

def merge_vocab(pair, v_in):
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out

def main():
    # Korpus contoh
    corpus = ["machine", "learning", "machinelearning"]
    
    # Inisialisasi kosakata dengan spasi antar karakter
    vocab = Counter([" ".join(list(word)) for word in corpus])
    
    num_merges = 10
    for i in range(num_merges):
        pairs = get_stats(vocab)
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        vocab = merge_vocab(best, vocab)
        print(f"Merge #{i+1}: {best}")

    print("\nKosakata BPE Akhir:")
    for word in vocab:
        print(word)

if __name__ == "__main__":
    main()
```

Algoritma berbasis graf seperti PageRank juga dapat diadaptasi untuk memberi peringkat pada pentingnya kata atau konsep dalam dokumen. Berikut adalah visualisasi graf hubungan kata menggunakan `networkx`.

### Contoh Python: Visualisasi Graf Hubungan

```python
import networkx as nx
import matplotlib.pyplot as plt

def main():
    # Membuat graf berarah
    G = nx.DiGraph()

    # Menambahkan node dan edge berbobot
    edges = [
        ("word1", "word2", 3),
        ("word2", "word3", 2),
        ("word1", "word3", 6),
        ("word3", "word4", 4),
        ("word4", "word5", 1)
    ]
    
    for u, v, w in edges:
        G.add_edge(u, v, weight=w)

    # Mencari jalur terpendek menggunakan Dijkstra
    path = nx.dijkstra_path(G, "word1", "word5", weight='weight')
    print(f"Jalur terpendek dari word1 ke word5: {path}")

    # Visualisasi
    pos = nx.circular_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=2000, font_size=10, font_weight='bold')
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    
    plt.title("Visualisasi Graf Hubungan Kata")
    plt.savefig("simple_graph.png")
    plt.show()

if __name__ == "__main__":
    main()
```

Matematika diskrit juga mencakup aritmatika modular yang kritis dalam aplikasi kriptografi, yang semakin relevan untuk keamanan dan privasi data dalam LLM.

## 2.7. Kesimpulan

Bab 2 memberikan landasan matematika yang kuat yang dipadukan dengan implementasi praktis. Konsep yang dibahas memastikan bahwa pembaca siap untuk memahami, mengembangkan, dan mengoptimalkan LLM, menjembatani kesenjangan antara pengetahuan teoretis dan aplikasi dunia nyata.

### 2.7.1. Pembelajaran Lebih Lanjut dengan GenAI

- Jelaskan peran ruang vektor dalam merepresentasikan word embeddings.
- Diskusikan pentingnya matematis dari nilai eigen dan vektor eigen dalam PCA.
- Analisis tantangan praktis penerapan perkalian matriks untuk data berdimensi tinggi.
- Deskripsikan proses dan signifikansi inferensi Bayesian dalam meningkatkan akurasi prediksi.
- Bagaimana Teorema Limit Pusat berlaku untuk pelatihan dan generalisasi LLM?
- Jelaskan signifikansi penurunan gradien dan variannya (SGD, Adam).
- Diskusikan penggunaan cross-entropy loss sebagai fungsi tujuan utama.
- Analisis konsep informasi bersama (mutual information) dalam memahami ketergantungan variabel.
- Deskripsikan proses SVD dalam meningkatkan kinerja LLM melalui faktorisasi matriks.
- Jelaskan bagaimana teori graf dapat diterapkan untuk memodelkan hubungan dalam LLM.
- Apa pertimbangan utama saat mengimplementasikan algoritma optimasi kombinatorial?
- Bagaimana konsep entropi berhubungan dengan ketidakpastian model?
- Deskripsikan implementasi pengali Lagrange dalam mengoptimalkan pelatihan LLM dengan kendala.
- Diskusikan tantangan mengimplementasikan struktur matematika diskrit (aljabar Boolean, teori set).
- Bagaimana metode Monte Carlo dapat diterapkan untuk memperkirakan model probabilistik?
- Jelaskan penggunaan diferensiasi otomatis dalam mengoptimalkan pelatihan.
- Jelajahi aplikasi pengelompokan spektral (spectral clustering) dalam LLM.
- Diskusikan penggunaan praktis aritmatika modular dalam aplikasi kriptografi di LLM.

### 2.7.2. Latihan Praktis

#### Latihan Mandiri 2.1: Mengimplementasikan dan Mengoptimalkan Perkalian Matriks
**Tujuan:** Mengembangkan pemahaman mendalam tentang perkalian matriks dalam konteks pemrosesan data skala besar.

#### Latihan Mandiri 2.2: Inferensi Bayesian dan Pemodelan Probabilistik
**Tujuan:** Menerapkan inferensi Bayesian dalam konteks model bahasa untuk memperbarui keyakinan berdasarkan bukti baru.

#### Latihan Mandiri 2.3: Analisis Komponen Utama untuk Reduksi Dimensionalitas
**Tujuan:** Mendapatkan pengalaman langsung dengan PCA dan aplikasinya dalam mengurangi dimensionalitas kumpulan data besar.

#### Latihan Mandiri 2.4: Diferensiasi Otomatis untuk Melatih LLM
**Tujuan:** Memahami dan mengimplementasikan mesin diferensiasi otomatis untuk menghitung gradien secara efisien.

#### Latihan Mandiri 2.5: Pengelompokan Spektral untuk Pengelompokan Data
**Tujuan:** Mengimplementasikan algoritma pengelompokan spektral untuk mengelompokkan titik data serupa (seperti word embeddings) dalam LLM.