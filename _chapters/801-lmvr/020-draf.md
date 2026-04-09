---
slug: lmvr-2
title: lmvr-2
---

# Bab 2: Fondasi Matematika untuk LLM

> 💡 **"Masa depan AI bergantung pada perpaduan algoritma yang kuat dan pemahaman matematika yang mendasarinya." — Andrew Ng**

> 📘 **Ringkasan Bab:**
> Bab 2 dari LMVR meletakkan dasar matematika yang esensial untuk memahami dan mengimplementasikan model bahasa besar (LLM). Bab ini mendalami bidang-bidang kritis seperti aljabar linear, probabilitas dan statistik, kalkulus dan optimasi, teori informasi, transformasi linear, dan matematika diskret. Ini mencakup konsep-konsep dasar seperti vektor, matriks, distribusi probabilitas, gradien, dan entropi, berkembang ke topik yang lebih maju seperti dekomposisi eigen, Principal Component Analysis (PCA), dan teori graf. Prinsip-prinsip matematika ini dikontekstualisasikan dalam Rust, menekankan implementasi praktis algoritma, teknik optimasi, dan struktur data yang esensial untuk pengembangan LLM yang efisien. Dengan mengintegrasikan teori dengan praktik pengkodean langsung di Rust, bab ini membekali pembaca dengan pemahaman yang kuat, komprehensif, dan tepat tentang dasar-dasar matematika yang diperlukan untuk LLM.

---

## 2.1. Aljabar Linear dan Ruang Vektor

Aljabar linear memainkan peran sentral dalam desain dan fungsi Model Bahasa Besar (LLM), mendasari matematika tentang bagaimana data direpresentasikan dan ditransformasikan di dalam model-model ini. Inti dari fondasi ini adalah skalar, vektor, matriks, dan tensor, yang mewakili blok bangunan manipulasi data. Skalar adalah nilai numerik tunggal, sedangkan vektor adalah array angka satu dimensi. Ketika Anda memperluas konsep ini ke dua dimensi, Anda memiliki matriks, dan tensor menggeneralisasi ini ke dimensi yang lebih tinggi lagi. Dalam konteks LLM, tensor sangat krusial karena mereka mewakili data multi-dimensi yang diperlukan untuk memodelkan bahasa, di mana setiap dimensi mungkin mengkodekan properti yang berbeda, seperti word embeddings, struktur kalimat, atau bahkan seluruh urutan teks.

Dalam bahasa aljabar linear, ruang vektor menyediakan kerangka kerja formal untuk memahami bagaimana titik data (vektor) disusun dan ditransformasikan. Sebuah ruang vektor terdiri dari kumpulan vektor yang dapat diskalakan dan dijumlahkan bersama untuk membentuk vektor baru. Di dalamnya, subruang adalah ruang vektor yang lebih kecil yang terkandung dalam ruang yang lebih besar, dan konsep ini penting saat mengurangi dimensionalitas dataset besar. Misalnya, saat melatih LLM, subruang dari kata-kata yang sering digunakan mungkin lebih kritis untuk mekanisme atensi daripada seluruh ruang vektor. Basis dari sebuah ruang vektor adalah set minimal vektor dari mana semua vektor dalam ruang tersebut dapat dibentuk melalui kombinasi linear. Dimensi dari ruang vektor adalah jumlah vektor dalam basis ini, dan untuk LLM, ini bisa berhubungan dengan jumlah token unik (kata atau subkata) yang diproses oleh model.

![Figure 1: Ilustrasi ruang vektor untuk NLP.](https://lmvr.rantai.dev/images/L6A504iArDGBvUq8WSeO-jwJbCh3P4pGx2LB9YHme-v1.jpeg)

Properti penting dari ruang vektor, seperti independensi linear dan rentang (*span*), secara langsung memengaruhi seberapa baik LLM dapat mempelajari hubungan yang bermakna antar kata. Sekumpulan vektor dikatakan independen secara linear jika tidak ada vektor dalam kumpulan tersebut yang dapat ditulis sebagai kombinasi dari yang lain, yang berarti setiap vektor menambahkan informasi unik. Ini adalah properti yang diinginkan dalam word embeddings, di mana vektor setiap kata harus menangkap informasi semantik yang berbeda. Rentang dari sekumpulan vektor mewakili semua vektor yang mungkin dibentuk dengan menggabungkan kumpulan tersebut secara linear. Dalam aplikasi praktis, konsep ini esensial saat memodelkan korpus teks yang besar, di mana rentang dari word embeddings harus mencakup hubungan semantik antar kata dalam bahasa tersebut.

Operasi matriks adalah manipulasi matematika yang menggerakkan komputasi LLM. Penjumlahan dan perkalian matriks bersifat fundamental saat mentransformasikan data antar lapisan dalam model. Sebagai contoh, diberikan dua matriks $A$ dan $B$, penjumlahan matriks menghasilkan matriks baru $C$ di mana setiap elemen $C_{ij} = A_{ij} + B_{ij}$, dan perkalian diberikan oleh $C_{ij} = \sum_k A_{ik} B_{kj}$, yang digunakan secara ekstensif dalam mentransformasikan data masukan melalui berbagai lapisan dalam jaringan saraf. Operasi seperti transpose matriks, di mana baris dan kolom ditukar, sangat kritis untuk operasi seperti dot-product attention dalam model Transformer. Memahami bagaimana matriks berinteraksi sangat penting untuk memahami bagaimana data mengalir melalui model, terutama saat bekerja dengan arsitektur kompleks seperti Transformer dalam LLM.

Matriks juga memiliki properti seperti invers dan determinan, yang fundamental dalam memecahkan sistem persamaan linear, meskipun penggunaannya dalam LLM lebih abstrak. Invers dari matriks $A$ adalah matriks lain $A^{-1}$ sedemikian rupa sehingga $A A^{-1} = I$ (di mana $I$ adalah matriks identitas), dan determinan memberikan wawasan tentang properti matriks, seperti apakah matriks tersebut dapat dibalik. Yang lebih penting untuk LLM, konsep eigenvalue dan eigenvector menyediakan basis matematika untuk teknik reduksi dimensionalitas, seperti Principal Component Analysis (PCA), yang digunakan dalam langkah pra-pemrosesan untuk mengurangi kompleksitas word embeddings atau fitur lainnya. Secara formal, jika matriks $A$ bekerja pada vektor $v$, vektor tersebut hanya akan diskalakan oleh skalar $\lambda$, artinya $A v = \lambda v$. Skalar $\lambda$ adalah eigenvalue, dan vektor $v$ adalah eigenvector. Prinsip ini membantu dalam mengekstraksi arah (atau komponen) paling signifikan dalam sebuah dataset, yang secara langsung membantu performa LLM dengan berfokus pada fitur yang paling penting.

Konsep penting lainnya dalam LLM adalah ortogonalisasi dan basis ortonormal. Dua vektor dikatakan ortogonal jika hasil kali titik (*dot product*) mereka adalah nol, dan basis ortogonal memungkinkan transformasi dan penyederhanaan yang mudah dalam komputasi matriks. Basis ortonormal adalah sekumpulan vektor ortogonal yang juga memiliki panjang unit. Dalam praktiknya, banyak algoritma seperti Gram-Schmidt digunakan untuk mengortogonalisasi vektor, menyederhanakan transformasi dalam komputasi LLM, terutama dalam metode seperti singular value decomposition (SVD), yang digunakan dalam reduksi dimensionalitas.

Operasi matriks adalah dasar bagi implementasi model bahasa besar (LLM), di mana tugas-tugas seperti mekanisme atensi, token embeddings, dan backpropagation sangat bergantung pada penanganan array multi-dimensi yang efisien. Berikut adalah implementasi operasi matriks menggunakan Python (sebagai padanan krate `nalgebra` di Rust):

```python
import numpy as np

def main():
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

    # Perkalian elemen-per-elemen (Hadamard product)
    e = np.array([[1.0, 2.0], [3.0, 4.0]])
    f = np.array([[0.5, 0.2], [0.3, 0.1]])
    hadamard_product = e * f
    print(f"Perkalian elemen-per-elemen (Hadamard product):\n{hadamard_product}")

    # Contoh invers matriks (untuk matriks 2x2)
    try:
        inverse = np.linalg.inv(e)
        print(f"Invers dari matriks e:\n{inverse}")
    except np.linalg.LinAlgError:
        print("Matriks e tidak dapat diinvers.")

    # Contoh dot product
    g = np.array([1.0, 2.0])
    h = np.array([3.0, 4.0])
    dot_product = np.dot(g, h)
    print(f"Dot product dari vektor g dan h: {dot_product}")

    # Perkalian matriks dinamis besar
    matrix1 = np.arange(1, 17).reshape(4, 4).astype(float)
    matrix2 = np.eye(4)
    large_result = np.dot(matrix1, matrix2)
    print(f"Hasil perkalian matriks dinamis besar:\n{large_result}")

if __name__ == "__main__":
    main()
```

Kode di atas mendemonstrasikan berbagai operasi matriks menggunakan NumPy, yang berfokus pada tugas-tugas yang relevan bagi LLM. Pertama, perkalian matriks-vektor dilakukan, diikuti oleh perkalian matriks-matriks untuk menunjukkan bagaimana data masukan yang lebih besar dapat diproses. Transpose digunakan untuk mengubah orientasi matriks, kebutuhan umum selama penghitungan gradien dalam backpropagation. Perkalian elemen-per-elemen (Hadamard product) disorot untuk mengilustrasikan bagaimana elemen-elemen matriks yang sesuai digabungkan dalam lapisan jaringan saraf tertentu. Selain itu, invers matriks disertakan untuk menyelesaikan sistem linear. Operasi-operasi ini membentuk tulang punggung komputasi dalam arsitektur LLM seperti Transformer, yang memungkinkan model untuk memproses dan belajar dari dataset teks yang sangat besar.

Salah satu tantangan terbesar dalam bekerja dengan model besar adalah manajemen memori yang efisien. Saat berhadapan dengan matriks dan tensor berukuran sangat besar, seperti yang ditemui dalam LLM, penting untuk mengoptimalkan bagaimana memori dialokasikan dan didealokasikan. Model kepemilikan dan peminjaman Rust, dikombinasikan dengan library seperti `nalgebra` dan `tch`, memastikan bahwa memori ditangani dengan aman dan efisien. Kemampuan Rust untuk mengelola memori tanpa *garbage collector* berarti operasi matriks besar dapat dilakukan tanpa penundaan tak terduga yang disebabkan oleh manajemen memori otomatis, menjadikannya pilihan tepat untuk mengimplementasikan LLM, di mana kinerja sangat kritis.

Sebagai kesimpulan, aljabar linear bukan sekadar fondasi teoretis tetapi alat praktis yang menggerakkan operasi inti LLM. Dari ruang vektor hingga operasi matriks, fitur tingkat sistem Rust dan library yang kuat seperti nalgebra memungkinkan pengembang untuk mengimplementasikan dan mengoptimalkan operasi ini secara efisien untuk aplikasi dunia nyata dalam LLM. Penggunaan prinsip-prinsip ini dalam tugas-tugas seperti word embeddings dan mekanisme self-attention menyoroti peran sentral yang dimainkan aljabar linear dalam model NLP modern.

---

## 2.2. Probabilitas dan Statistik

Dalam dunia LLM, probabilitas dan statistik memainkan peran esensial dalam memahami dan memodelkan ketidakpastian serta keacakan yang melekat dalam bahasa alami. Fondasi ini dimulai dengan teori probabilitas, yang mempelajari variabel acak dan kemungkinan dari berbagai hasil. Variabel acak adalah variabel yang nilainya merupakan hasil dari fenomena acak, dan bisa berupa diskret (mengambil nilai tertentu seperti hasil lemparan dadu) atau kontinu (mengambil nilai apa pun dalam rentang tertentu, seperti suhu). Distribusi probabilitas menjelaskan seberapa besar kemungkinan variabel acak mengambil nilai tertentu. Konsep penting seperti ekspektasi (nilai rata-rata atau yang diharapkan dari variabel acak) dan varians (ukuran seberapa tersebar nilai- variabel acak tersebut) sangat krusial untuk memahami bagaimana model memproses dan menghasilkan bahasa di dunia yang tidak pasti.

![Figure 2: Peran probabilitas dan statistik dalam LLM.](https://lmvr.rantai.dev/images/L6A504iArDGBvUq8WSeO-9SuVOjzqbJWbcs1GsWHq-v1.png)

Beberapa distribusi probabilitas penting yang umum digunakan dalam pengembangan LLM meliputi: distribusi Gaussian (juga dikenal sebagai distribusi normal) yang sentral dalam memodelkan variabel kontinu; distribusi Bernoulli yang memodelkan hasil biner (seperti respons ya/tidak); distribusi Binomial yang menggeneralisasi ini ke beberapa percobaan; distribusi Multinomial yang memperluas kasus Binomial ke skenario dengan lebih dari dua kemungkinan hasil (sangat berguna dalam prediksi kata); dan distribusi Poisson yang digunakan untuk memodelkan jumlah peristiwa yang terjadi dalam interval waktu atau ruang yang tetap.

Inferensi Bayesian adalah konsep kunci lainnya dalam LLM, menyediakan kerangka kerja untuk memperbarui probabilitas suatu hipotesis seiring tersedianya lebih banyak bukti. Dalam penalaran Bayesian, ide kuncinya adalah memperbarui keyakinan awal (*prior probability*) dengan bukti baru (*likelihood*) untuk mendapatkan keyakinan yang direvisi (*posterior probability*). Secara matematis, ini direpresentasikan menggunakan Teorema Bayes:

$$ P(A | B) = \frac{P(B | A) \cdot P(A)}{P(B)} $$

Dalam LLM, inferensi Bayesian digunakan untuk meningkatkan prediksi model dengan memasukkan pengetahuan awal atau ketidakpastian ke dalam proses pengambilan keputusan model. Klasifikasi Naive Bayes, misalnya, didasarkan pada penerapan Teorema Bayes dengan asumsi independensi antar fitur. Meskipun sederhana, klasifikasi Naive Bayes sering digunakan dalam tugas klasifikasi teks karena efisiensi komputasi dan interpretabilitasnya.

Probabilitas bersyarat dan independensi juga kritis dalam LLM. Probabilitas bersyarat $P(A|B)$ mewakili probabilitas peristiwa A terjadi jika peristiwa B sudah terjadi. Memahami ketergantungan dan hubungan antar kata dalam sebuah kalimat dapat dibingkai dalam istilah probabilitas bersyarat. Independensi terjadi ketika probabilitas satu peristiwa tidak memengaruhi probabilitas peristiwa lainnya.

Pengujian hipotesis digunakan untuk menentukan apakah suatu hasil signifikan secara statistik. Dalam LLM, ini relevan saat membandingkan model yang berbeda atau menilai performa di bawah kondisi eksperimental yang berbeda. Alat umum dalam pengujian hipotesis adalah *p-value* dan interval kepercayaan (*confidence intervals*).

Berikut adalah implementasi konsep probabilitas dan pengambilan sampel menggunakan Python (padanan contoh Rust):

```python
import numpy as np
import scipy.stats as stats

# Implementasi fungsi Softmax
def softmax(logits):
    exp_values = np.exp(logits - np.max(logits)) # Stabilitas numerik
    return exp_values / exp_values.sum()

def main():
    # Contoh 1: Distribusi Gaussian (Normal)
    mean, std = 0.0, 1.0
    gaussian_samples = np.random.normal(mean, std, 10)
    print("Sampel Distribusi Gaussian:")
    print(gaussian_samples)

    # Contoh 2: Distribusi Bernoulli
    p_success = 0.7
    bernoulli_samples = np.random.binomial(1, p_success, 10)
    print(f"\nSampel Distribusi Bernoulli (p = {p_success}):")
    print(bernoulli_samples)

    # Contoh 3: Distribusi Multinomial
    probabilities = [0.2, 0.5, 0.3]
    multinomial_sample = np.random.multinomial(1, probabilities, 10)
    print("\nSampel Distribusi Multinomial (untuk prediksi kata):")
    print(multinomial_sample)

    # Contoh 4: Sampling Softmax untuk LLM
    logits = np.array([2.0, 1.0, 0.1])
    softmax_probs = softmax(logits)
    # Mengambil sampel berdasarkan probabilitas softmax
    indices = np.random.choice(len(softmax_probs), size=5, p=softmax_probs)
    print(f"\nSampling berbasis Softmax (Logits: {logits}):")
    print(f"Indeks terpilih: {indices}")

    # Contoh 5: Estimasi Monte Carlo menggunakan distribusi Gaussian
    num_samples = 1000
    mc_samples = np.random.normal(0, 1, num_samples)
    mc_estimate = np.mean(mc_samples)
    print(f"\nEstimasi Monte Carlo dari Ekspektasi (Gaussian): {mc_estimate}")

if __name__ == "__main__":
    main()
```

Kode Python ini mendemonstrasikan berbagai distribusi probabilitas dan teknik pengambilan sampel yang relevan bagi LLM. Dimulai dengan pengambilan sampel dari distribusi Gaussian, diikuti oleh distribusi Bernoulli untuk tugas klasifikasi biner, dan distribusi Multinomial yang esensial untuk menghasilkan prediksi kategoris seperti pemilihan kata. Kode ini juga mencakup contoh pengambilan sampel softmax, di mana logit diubah menjadi probabilitas untuk menghasilkan prediksi kata. Terakhir, diimplementasikan estimasi Monte Carlo untuk menghitung nilai yang diharapkan.

Integrasi penalaran probabilistik dalam pipa LLM menjadi lebih lazim dengan munculnya teknik seperti Bayesian Neural Networks (BNNs). BNN menggabungkan kekuatan deep learning dengan inferensi Bayesian, memungkinkan LLM menghasilkan prediksi dengan interval kepercayaan, yang sangat berharga di bidang seperti perawatan kesehatan atau keuangan.

Sebagai kesimpulan, probabilitas dan statistik sangat diperlukan dalam pengembangan dan aplikasi LLM. Dari fondasi teoretis distribusi probabilitas dan Teorema Bayes hingga implementasi praktis, integrasi penalaran probabilistik dalam LLM sangat penting untuk memodelkan bahasa, memahami ketidakpastian, dan meningkatkan kemampuan pengambilan keputusan.

---

## 2.3. Kalkulus dan Optimasi

Dalam pengembangan dan pelatihan Model Bahasa Besar (LLM), kalkulus bersifat fundamental, terutama dalam konteks optimasi. Diferensiasi dan integrasi berfungsi sebagai tulang punggung matematika tentang bagaimana model belajar dengan menyesuaikan parameternya untuk meminimalkan fungsi kerugian (*loss function*). Dalam pembelajaran mesin, kita terutama berurusan dengan diferensiasi, yang membantu memahami bagaimana perubahan kecil dalam parameter model memengaruhi output. Gradien, yang merupakan vektor dari turunan parsial, mewakili arah pendakian tercuram untuk fungsi multivariabel, dan mereka adalah kunci untuk mengoptimalkan parameter selama pelatihan.

![Figure 3: Siklus optimasi dalam melatih LLM.](https://lmvr.rantai.dev/images/L6A504iArDGBvUq8WSeO-HAkqlPnDmnvkCgALfKs7-v1.png)

Aturan rantai (*chain rule*) sangat penting saat bekerja dengan model yang dalam, di mana masukan diproses melalui berbagai lapisan fungsi. Aturan rantai memungkinkan kita menghitung turunan dari fungsi komposit, yang merupakan kasus dalam backpropagation. Secara formal, jika kita memiliki fungsi komposit $f(g(x))$, aturan rantai menyatakan:

$$\frac{d}{dx} f(g(x)) = f'(g(x)) \cdot g'(x)$$

Optimasi adalah jantung dari pelatihan LLM, dan salah satu metode yang paling penting adalah *gradient descent*. Dalam algoritma ini, kita menggunakan gradien dari fungsi kerugian untuk memperbarui parameter model secara berulang. Secara matematis, jika $\theta$ mewakili parameter model dan $L(\theta)$ mewakili fungsi kerugian, *gradient descent* memperbarui parameter sebagai berikut:

$$ \theta := \theta - \eta \cdot \nabla_\theta L(\theta) $$

di mana $\eta$ adalah tingkat pembelajaran (*learning rate*), dan $\nabla_\theta L(\theta)$ adalah gradien dari fungsi kerugian terhadap parameter.

![Figure 4: Animasi 5 metode gradient descent pada sebuah permukaan: SGD (sian), Momentum (magenta), AdaGrad (putih), RMSProp (hijau), Adam (biru). Sumur kiri adalah minimum global; sumur kanan adalah minimum lokal.](https://lmvr.rantai.dev/images/L6A504iArDGBvUq8WSeO-UA416PCEldbMPUIr2kvE-v1.gif)

Dalam konteks melatih LLM, ada beberapa optimizer kunci:
1.  **SGD**: Paling sederhana, memperbarui parameter dengan mengambil langkah ke arah negatif gradien.
2.  **Momentum**: Memasukkan istilah "kecepatan" untuk membantu meredam osilasi.
3.  **AdaGrad**: Menyesuaikan tingkat pembelajaran secara adaptif berdasarkan informasi gradien historis.
4.  **RMSProp**: Memperbaiki AdaGrad dengan memperkenalkan rata-rata bergerak dari kuadrat gradien.
5.  **Adam**: Menggabungkan manfaat momentum dan tingkat pembelajaran adaptif, menjadi pilihan utama untuk LLM modern seperti GPT dan BERT.

Berikut adalah simulasi perbandingan performa optimizer tersebut dalam Python:

```python
import numpy as np
import time

class Optimizers:
    @staticmethod
    def sgd(params, grad, lr):
        return params - lr * grad

    @staticmethod
    def momentum(params, grad, lr, v, beta=0.9):
        v = beta * v + (1 - beta) * grad
        params = params - lr * v
        return params, v

    @staticmethod
    def adagrad(params, grad, lr, accum, eps=1e-8):
        accum += grad**2
        params = params - lr * grad / (np.sqrt(accum) + eps)
        return params, accum

    @staticmethod
    def rmsprop(params, grad, lr, accum, beta=0.9, eps=1e-8):
        accum = beta * accum + (1 - beta) * grad**2
        params = params - lr * grad / (np.sqrt(accum) + eps)
        return params, accum

    @staticmethod
    def adam(params, grad, lr, m, v, t, beta1=0.9, beta2=0.999, eps=1e-8):
        t += 1
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad**2
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        params = params - lr * m_hat / (np.sqrt(v_hat) + eps)
        return params, m, v, t

def benchmark_optimizers():
    param_size = 3
    num_iterations = 10000
    lr = 0.01
    
    # Fungsi gradien kerugian sederhana: (p - 0.5)
    def loss_grad(params):
        return params - 0.5

    initial_params = np.array([1.0, 1.0, 1.0])

    # SGD
    p = initial_params.copy()
    start = time.time()
    for _ in range(num_iterations):
        p = Optimizers.sgd(p, loss_grad(p), lr)
    print(f"SGD duration: {time.time() - start:.6f}s")

    # Momentum
    p = initial_params.copy()
    v = np.zeros(param_size)
    start = time.time()
    for _ in range(num_iterations):
        p, v = Optimizers.momentum(p, loss_grad(p), lr, v)
    print(f"Momentum duration: {time.time() - start:.6f}s")

    # AdaGrad
    p = initial_params.copy()
    accum = np.zeros(param_size)
    start = time.time()
    for _ in range(num_iterations):
        p, accum = Optimizers.adagrad(p, loss_grad(p), lr, accum)
    print(f"AdaGrad duration: {time.time() - start:.6f}s")

    # Adam
    p = initial_params.copy()
    m = np.zeros(param_size)
    v = np.zeros(param_size)
    t = 0
    start = time.time()
    for _ in range(num_iterations):
        p, m, v, t = Optimizers.adam(p, loss_grad(p), lr, m, v, t)
    print(f"Adam duration: {time.time() - start:.6f}s")

if __name__ == "__main__":
    benchmark_optimizers()
```

Kode di atas mengimplementasikan setiap optimizer sebagai fungsi terpisah dan melakukan benchmark kecepatan eksekusi dengan mensimulasikan loop pelatihan sederhana. Hasilnya melaporkan waktu yang dibutuhkan oleh setiap optimizer untuk memproses tugas yang sama.

Sebagai kesimpulan, kalkulus dan optimasi sangat vital bagi pelatihan dan fungsi LLM. Konsep seperti diferensiasi, *gradient descent*, dan optimasi konveks membentuk tulang punggung bagaimana LLM belajar dari data. Mengimplementasikan metode ini dalam Rust memberikan beberapa keuntungan, termasuk manajemen memori yang efisien dan kemampuan pemrosesan paralel yang aman.

---

## 2.4. Teori Informasi

Teori informasi menyediakan kerangka kerja matematika kritis untuk memahami dan mengukur ketidakpastian, kompresi, dan transmisi informasi. Dalam konteks LLM, teori informasi membantu mengukur jumlah ketidakpastian dalam prediksi, kesamaan antar distribusi probabilitas, dan efisiensi komunikasi antar komponen model. Konsep entropi adalah pusat dari teori informasi, mengukur ketidakpastian atau keacakan dalam variabel acak $X$:

$$ H(X) = -\sum_{x \in X} p(x) \log p(x) $$

Dalam pelatihan model bahasa, *cross-entropy* adalah salah satu fungsi kerugian yang paling banyak digunakan. Ia mengukur perbedaan antara dua distribusi probabilitas: distribusi sebenarnya (label) dan distribusi prediksi (output model):

$$ H(p, q) = -\sum_{i} p(y_i) \log q(y_i) $$

*Kullback-Leibler (KL) divergence* adalah konsep penting lainnya yang mengukur bagaimana satu distribusi probabilitas menyimpang dari distribusi referensi kedua:

$$ D_{KL}(P || Q) = \sum_{x} P(x) \log \frac{P(x)}{Q(x)} $$

Berikut adalah implementasi ukuran teori informasi dalam Python:

```python
import numpy as np

def log_softmax(x):
    # Log-softmax yang stabil secara numerik
    e_x = np.exp(x - np.max(x))
    return np.log(e_x / e_x.sum())

def cross_entropy_loss(y_true, y_pred_logits):
    # y_true: One-hot encoded, y_pred_logits: Raw scores
    total_loss = 0.0
    for i in range(len(y_true)):
        log_probs = log_softmax(y_pred_logits[i])
        total_loss += -np.dot(y_true[i], log_probs)
    return total_loss / len(y_true)

def kl_divergence(p, q):
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    # Menghindari pembagian dengan nol
    q = np.where(q == 0, 1e-10, q)
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

def main():
    # Contoh Cross-entropy batch
    y_true = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    y_pred_logits = np.array([[0.7, 0.2, 0.1], [0.4, 0.5, 0.1]])
    
    loss = cross_entropy_loss(y_true, y_pred_logits)
    print(f"Cross-entropy loss (batch): {loss}")

    # Contoh KL Divergence
    p = [0.6, 0.3, 0.1]
    q = [0.5, 0.4, 0.1]
    kl_div = kl_divergence(p, q)
    print(f"KL divergence: {kl_div}")

if __name__ == "__main__":
    main()
```

Kode Python tersebut mengimplementasikan versi lanjutan dari *cross-entropy loss* dan *KL divergence*. Dalam fungsi *cross-entropy*, operasi softmax distabilkan dengan mengurangkan nilai maksimum dari prediksi sebelum menerapkan logaritma, mencegah masalah *overflow* dan *underflow*.

---

## 2.5. Transformasi Linear dan Dekomposisi Eigen

Dalam konteks LLM, transformasi linear memainkan peran krusial dalam bagaimana data masukan dimanipulasi dan direpresentasikan dalam berbagai lapisan model. Transformasi linear dalam bentuk matriks direpresentasikan dengan mengalikan matriks dengan vektor. Bobot di setiap lapisan jaringan saraf bertindak sebagai transformasi linear yang diterapkan pada vektor masukan, memproyeksikannya ke ruang fitur baru.

Dekomposisi matriks adalah konsep vital lainnya, melibatkan pemecahan matriks menjadi komponen yang lebih sederhana. Teknik seperti dekomposisi LU, QR, dan Cholesky umum digunakan dalam aljabar linear numerik.

Eigenvalue dan eigenvector sangat penting untuk memahami perilaku transformasi linear. Diberikan matriks $A$, jika ada skalar $\lambda$ dan vektor non-nol $v$ sedemikian rupa sehingga: $A v = \lambda v$, maka $\lambda$ disebut eigenvalue dan $v$ disebut eigenvector-nya.

Salah satu teknik dekomposisi matriks yang paling kuat adalah *Singular Value Decomposition* (SVD), yang memecah matriks $A$ menjadi tiga matriks $A = U \Sigma V^T$. SVD sangat krusial dalam tugas seperti *Latent Semantic Analysis* (LSA) dan *Principal Component Analysis* (PCA).

PCA adalah teknik yang banyak digunakan untuk reduksi dimensionalitas. PCA bekerja dengan memproyeksikan data berdimensi tinggi ke ruang berdimensi lebih rendah sambil mempertahankan jumlah varians maksimum dalam data.

![Figure 5: Ilustrasi metode PCA untuk reduksi dimensionalitas.](https://lmvr.rantai.dev/images/L6A504iArDGBvUq8WSeO-28fwUEkC1vJukNX5t4aW-v1.png)

Berikut adalah implementasi PCA menggunakan Python:

```python
import numpy as np

def pca(matrix, n_components):
    # 1. Centering data (mengurangi rata-rata)
    mean = np.mean(matrix, axis=0)
    centered_matrix = matrix - mean
    
    # 2. Menghitung matriks kovarians
    covariance_matrix = np.cov(centered_matrix, rowvar=False)
    
    # 3. Dekomposisi Eigen
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    
    # 4. Mengurutkan eigenvalue dan eigenvector dari yang terbesar
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]
    
    # 5. Memilih n_components teratas
    components = eigenvectors[:, :n_components]
    
    # 6. Proyeksi data
    return np.dot(centered_matrix, components)

def main():
    # Contoh data: 4 sampel dengan 3 fitur
    data = np.array([[4.0, 2.0, 1.0],
                     [3.0, 1.0, 2.0],
                     [1.0, 0.0, 1.0],
                     [0.0, 1.0, 3.0]])

    # Melakukan PCA untuk reduksi dimensionalitas menjadi 2 komponen
    reduced_data = pca(data, 2)
    print("Data yang direduksi:\n", reduced_data)

    # Analisis Dekomposisi Eigen dari (X^T * X)
    cov = np.dot(data.T, data)
    evals, evecs = np.linalg.eigh(cov)
    print("\nEigenvalues:\n", evals)
    print("Eigenvectors:\n", evecs)

if __name__ == "__main__":
    main()
```

Kode ini mendemonstrasikan aplikasi PCA menggunakan dekomposisi eigen. Dalam LLM, penanganan matriks besar yang efisien sangat krusial, dan mengurangi dimensionalitas fitur masukan atau embedding dapat meningkatkan performa model secara signifikan.

---

## 2.6. Matematika Diskret

Matematika diskret berurusan dengan nilai yang berbeda dan terpisah, memberikan alat dan fondasi teoretis untuk memahami struktur seperti graf, himpunan, dan penalaran logika.

Satu area kunci adalah teori graf. Pengetahuan graf (*knowledge graphs*) mewakili hubungan semantik antar konsep atau entitas. Dalam LLM, pengetahuan graf dapat digunakan untuk meningkatkan kemampuan model dalam menjawab pertanyaan dan memberikan output yang lebih akurat.

Kombinatorika memainkan peran penting dalam tugas seperti tokenisasi dan pemodelan urutan. Salah satu contoh utamanya adalah *Byte Pair Encoding* (BPE), metode tokenisasi subkata yang banyak digunakan dalam model seperti BERT dan GPT.

Berikut adalah implementasi sederhana BPE dalam Python:

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
    # Inisialisasi vocab: karakter dipisahkan spasi, diakhiri penanda kata
    vocab = Counter([' '.join(list(word)) for word in corpus])
    
    num_merges = 10
    for i in range(num_merges):
        pairs = get_stats(vocab)
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        vocab = merge_vocab(best, vocab)
        print(f"Merge #{i+1}: {best}")

    print("\nFinal BPE Vocabulary Structure:")
    for word, freq in vocab.items():
        print(f"{word}: {freq}")

if __name__ == "__main__":
    main()
```

Algoritma ini menghitung frekuensi pasangan karakter yang berdekatan dan secara iteratif menggabungkan pasangan yang paling sering menjadi token baru. Teknik kombinatorial ini secara efisien mengurangi ukuran kosakata sambil tetap mempertahankan pola umum dalam korpus.

Aplikasi matematika diskret lainnya adalah visualisasi graf. Berikut adalah contoh cara membuat visualisasi graf sederhana menggunakan Python (padanan contoh Rust dengan `petgraph` dan `plotters`):

```python
import networkx as nx
import matplotlib.pyplot as plt

def main():
    # Membuat graf berarah
    G = nx.DiGraph()

    # Menambahkan node
    nodes = ["word1", "word2", "word3", "word4", "word5"]
    G.add_nodes_from(nodes)

    # Menambahkan edge dengan bobot
    edges = [
        ("word1", "word2", 3),
        ("word2", "word3", 2),
        ("word1", "word3", 6),
        ("word3", "word4", 4),
        ("word4", "word5", 1)
    ]
    for u, v, w in edges:
        G.add_edge(u, v, weight=w)

    # Mencari jalur terpendek menggunakan algoritma Dijkstra
    path = nx.dijkstra_path(G, "word1", "word5", weight='weight')
    length = nx.dijkstra_path_length(G, "word1", "word5", weight='weight')
    
    print(f"Jalur terpendek dari word1 ke word5: {path}")
    print(f"Total bobot: {length}")

    # Visualisasi
    pos = nx.circular_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='red', node_size=2000, font_size=10)
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    
    plt.title("Visualisasi Graf Sederhana")
    plt.savefig("simple_graph_py.png")
    plt.show()

if __name__ == "__main__":
    main()
```

Kode ini membangun graf berarah, menambahkan hubungan antar kata, dan menerapkan algoritma Dijkstra untuk menghitung jalur terpendek. Visualisasi dihasilkan menggunakan tata letak melingkar dengan tanda panah yang menunjukkan arah hubungan.

Sebagai kesimpulan, matematika diskret memainkan peran penting dalam pengembangan LLM, baik melalui teori graf dalam pengetahuan graf, teori himpunan dalam pengambilan keputusan logis, maupun optimasi kombinatorial dalam tokenisasi.

---

## 2.7. Kesimpulan

Bab 2 menetapkan panggung untuk menguasai model bahasa besar dengan menyediakan fondasi matematika yang kuat yang dipadukan dengan implementasi praktis. Konsep yang dibahas memastikan bahwa pembaca dibekali dengan baik untuk memahami, mengembangkan, dan mengoptimalkan LLM, menjembatani kesenjangan antara pengetahuan teoretis dan aplikasi dunia nyata.

### 2.7.1. Pembelajaran Lebih Lanjut dengan GenAI

Setiap prompt adalah ajakan untuk memperluas pemahaman Anda:

- Jelaskan peran ruang vektor dalam merepresentasikan word embeddings. Bagaimana konsep independensi linear dan ortogonalitas memengaruhi kualitas interpretasi embedding?
- Diskusikan pentingnya matematis dari eigenvalue dan eigenvector dalam konteks reduksi dimensionalitas seperti PCA pada LLM.
- Apa tantangan praktis dalam mengimplementasikan perkalian matriks untuk data berdimensi tinggi di Rust? Analisis perbedaan antara matriks padat (*dense*) dan jarang (*sparse*).
- Deskripsikan proses inferensi Bayesian dalam meningkatkan akurasi prediksi dan kuantifikasi ketidakpastian pada LLM.
- Bagaimana Teorema Limit Pusat berlaku pada pelatihan dan generalisasi LLM?
- Jelaskan signifikansi *gradient descent* dan variannya (SGD, Adam, dll.) dalam melatih LLM.
- Diskusikan penggunaan *cross-entropy loss* sebagai fungsi objektif utama dalam pelatihan LLM.
- Analisis konsep informasi bersama (*mutual information*) dalam memahami ketergantungan antar variabel pada LLM.
- Jelaskan peran *Singular Value Decomposition* (SVD) dalam meningkatkan performa LLM melalui teknik faktorisasi matriks.
- Deskripsikan bagaimana teori graf dapat diterapkan untuk memodelkan hubungan dalam LLM, seperti pada konstruksi pengetahuan graf.

### 2.7.2. Praktik Langsung

#### **Latihan Mandiri 2.1: Mengimplementasikan dan Mengoptimalkan Perkalian Matriks**
**Objektif:** Mengembangkan pemahaman mendalam tentang perkalian matriks dalam konteks pemrosesan data skala besar dan mengimplementasikan versi yang dioptimalkan.

#### **Latihan Mandiri 2.2: Inferensi Bayesian dan Pemodelan Probabilistik**
**Objektif:** Menerapkan inferensi Bayesian dalam konteks model bahasa, menggunakan Rust/Python untuk mengimplementasikan dan menganalisis model probabilistik.

#### **Latihan Mandiri 2.3: Principal Component Analysis untuk Reduksi Dimensionalitas**
**Objektif:** Mendapatkan pengalaman langsung dengan PCA dan aplikasinya dalam mengurangi dimensionalitas dataset besar.

#### **Latihan Mandiri 2.4: Diferensiasi Otomatis untuk Melatih LLM**
**Objektif:** Memahami dan mengimplementasikan mesin diferensiasi otomatis (*automatic differentiation*) sederhana untuk menghitung gradien.

#### **Latihan Mandiri 2.5: Spectral Clustering untuk Pengelompokan Data**
**Objektif:** Mengimplementasikan algoritma *spectral clustering* dari awal dan mengeksplorasi aplikasinya dalam mengelompokkan titik data serupa di dalam LLM.