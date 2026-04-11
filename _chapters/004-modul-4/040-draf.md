---
slug: modul-4-4
title: modul-4-4
---
# 4.4 Distillation: Teknik Kompresi Model yang Ramah Performa
---

Model yang besar (jumlah parameternya banyak) ideal untuk *training*, tetapi mungkin kurang ideal untuk *inference*. Idealnya, diinginkan model yang optimal untuk *inference* tapi memiliki kemampuan yang mirip dengan model dengan kapasitas yang lebih besar. Dalam *Machine Learning*, terdapat konsep yang disebut sebagai *knowledge distillation*, yaitu proses untuk menggunakan model dengan kapasitas yang besar untuk melatih model lainnya dengan kapasitas yang lebih kecil untuk mencapai tingkatan performa yang mendekati model yang lebih besar. Teknik ini termasuk salah satu teknis yang umum dilakukan untuk melakukan *model compression*, dan memiliki efektivitas yang cukup terbukti mampu menghasilkan model yang efisien secara *compute* tapi memiliki tingkat performa yang cukup baik.

---

### 4.4.1 Memahami knowledge distillation

Dalam *knowledge distillation*, sebuah model yang disebut sebagai *teacher model* digunakan untuk melatih model lainnya yang disebut *student model* untuk melakukan “transfer ilmu” dari model *teacher* ke model *student*. Menurut *universal approximation theorem*, sebuah *neural network* dengan 1 *hidden layer* bisa digunakan untuk melakukan aproksimasi fungsi apapun. Akan tetapi pada praktisnya, metode yang umumnya digunakan untuk melatih model *Deep Learning* mengalami kesulitan untuk mencapai tingkatan performa yang tinggi untuk model dengan kapasitas yang lebih rendah. Hal ini menjadi motivasi di balik penelitian berjudul [“Distilling the Knowledge in a Neural Network”](https://arxiv.org/abs/1503.02531) oleh Hinton dan timnya yang mencetuskan ide di balik *knowledge distillation*.

Mereka mendemonstrasikan proses yang disebut sebagai *knowledge distillation* atau distilasi pengetahuan yang dilakukan dengan cara melatih model *student* menggunakan *soft targets* berupa distribusi kelas pada *output layer* sebuah model *teacher*. Mereka menemukan bahwa model *student* dengan jumlah parameter yang lebih sedikit yang dilatih menggunakan cara ini mampu mencapai tingkat performa yang mendekati model *teacher*, bahkan tanpa memerlukan dataset berlabel sekalipun. Temuan ini mereka generalisasi dimana tidak hanya distribusi output pada *layer terakhir*, tapi distribusi fitur pada *intermediate layer* juga bisa digunakan sebagai *soft target* untuk melakukan proses distilasi. Mereka menyebut pengetahuan “tersembunyi” yang bisa dipelajari oleh model *student* yang terkandung dalam distribusi kelas maupun *intermediate representation* dari suatu *layer* sebagai dengan istilah “*Dark Knowledge*”.

![Ilustrasi proses knowledge distillation](https://lms.sdmdigital.id/pluginfile.php/635389/mod_page/content/16/image29.jpg)
**Gambar 16. Ilustrasi proses knowledge distillation**

Umumnya proses distilasi menggunakan bentuk loss seperti berikut untuk *output logits*:

**Formula 13**
$$ℒ_{\mathrm{distill}} = T^2 \cdot \mathrm{KL}(\mathrm{softmax}(z_T/T) \parallel \mathrm{softmax}(z_S/T))$$
**Caption: Formula 13. Loss function untuk proses distilasi**

Dimana $T$ adalah *temperature*, $z_T$ adalah vektor *logits* dari model *teacher* dan $z_S$ untuk model *student*. *Temperature* dalam softmax digunakan untuk “melebur” distribusi peluang yang diberikan softmax. Semakin tinggi temperatur, maka distribusi peluang akan semakin rata.

![Ilustrasi softmax dengan temperatur](https://lms.sdmdigital.id/pluginfile.php/635389/mod_page/content/16/img26a.png)
**Gambar 17. Ilustrasi softmax dengan temperatur 1 dan 10. Perhatikan distribusi peluang untuk T yang lebih besar semakin “mirip” satu sama lain.**

Selain *loss* di atas, saat melakukan *knowledge distillation*, loss lainnya yang menggunakan *hard label* (label ground truth) seperti cross entropy bisa digunakan. Total loss adalah *weighted sum* antara loss ini dengan loss distilasi.

---

### 4.4.2 Menerapkan knowledge distillation pada LLM

Umumnya ada dua skenario penerapan *knowledge distillation* pada LLM:
1. Kompresi base model hasil *pretraining*
2. Distilasi model yang sudah di-*tuning* (instruksi, *preference alignment*)

Pada praktiknya, proses distilasi kedua skenario ini tidak memiliki banyak perbedaan. Yang menjadi kunci di LLM adalah proses distilasi selain dilakukan pada output akhir juga dilakukan pada *intermediate outputs* seperti *hidden state*, *embeddings* dan *attention map*. Selain itu terdapat juga pendekatan distilasi yang lebih spesifik ke perilaku model seperti meniru gaya bahasa, meniru *alignment*, hingga secara langsung meniru *respon* dari sebuah model *teacher*. Salah satu model open source yang populer yaitu DeepSeek-R1 merilis informasi soal berbagai model hasil distilasinya lengkap dengan perbandingan performa.

**Tabel 8. Perbandingan performa antara model student dan teacher pada beberapa model hasil distilasi**

| Student | Teacher | Performa Teacher | Performa Student |
| :--- | :--- | :--- | :--- |
| DeepSeek-R1-Distill-Qwen-1.5B | Deepseek-R1 (685B) | AIME2024: 79.8%<br>MATH-500: 97.3% | AIME2024: 28.9%<br>MATH-500: 83.9% |
| DeepSeek-R1-Distill-Qwen-14B | Deepseek-R1 (685B) | AIME2024: 79.8%<br>MATH-500: 97.3% | AIME2024: 69.7%<br>MATH-500: 93.9% |
| DeepSeek-R1-Distill-Llama-32B | Deepseek-R1 (685B) | AIME2024: 79.8%<br>MATH-500: 97.3% | AIME2024: 72.6%<br>MATH-500: 94.3% |
| Mamba-llama3 (50% attention layers) | LLaMA 3-8B-Instruct | MT-Bench: 8.00<br>AlpacaEval: 22.90 | MT-Bench: 7.35<br>AlpacaEval: 29.61 |
| Mamba-llama3 (0% attention layers) | LLaMA 3-8B-Instruct | MT-Bench: 8.00<br>AlpacaEval: 22.90 | MT-Bench: 5.64<br>AlpacaEval: 14.49 |

Tabel di atas juga memasukkan Mamba, sebuah arsitektur LLM alternatif yang berbeda jauh dengan transformer. Hal ini menunjukkan bahwa dengan cara yang tepat, distilasi bahkan bisa dilakukan untuk model dengan arsitektur yang berbeda. Salah satu keuntungan dari melakukan distillation adalah ukuran model yang lebih kecil dan inferensi yang lebih cepat.

**Tabel 9. Faktor kompresi dan speedup dari beberapa model hasil distilasi**

| Student | Teacher | Faktor speedup (rasio throughput) | Faktor kompresi (ukuran model) |
| :--- | :--- | :--- | :--- |
| Mistral-NeMO-Minitron 8B | Mistral NeMo 12B | ~1.2x | ~63% |
| LLaMA-3.1-Minitron 4B | LLaMA 3.1 8B | ~2.7x | ~56% |

Distilasi merupakan pendekatan yang cukup lumrah di industri maupun dunia akademik. Suatu [artikel](https://techcrunch.com/2023/12/06/googles-gemini-isnt-the-generative-ai-model-we-expected/) mengatakan bahwa *Gemini nano* merupakan hasil distilasi dari varian yang lebih besar. Beberapa model dalam famili *[Deepseek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B)* juga dibuat melalui proses distilasi. Berbagai macam varian model *open source* juga dilatih dengan berbagai macam bentuk distilasi, misalnya [Vicuna](https://lmsys.org/blog/2023-03-30-vicuna/) dan [Alpaca](https://crfm.stanford.edu/2023/03/13/alpaca.html).

---

### Mini Quiz

*(Silakan akses kuis interaktif pada platform LMS untuk menguji pemahaman Anda)*

---

> **Refleksi Singkat**
>
> Distilasi bisa dilakukan untuk melakukan kompresi LLM dengan menggunakan LLM yang memiliki jumlah parameter yang cukup besar yang sudah dilatih sebagai teacher dan LLM dengan parameter yang lebih sedikit sebagai student. Sejauh apa kompresi ini bisa dilakukan sebelum ditemukan penurunan performa yang drastis? Bagaimana caranya melewati batasan tersebut? Selain distilasi, apa lagi metode yang bisa dilakukan untuk melakukan kompresi LLM?
>
> Distilasi bisa dilakukan dengan meminta student meniru distribusi output dari teacher, yang digeneralisasi dimana fitur-fitur intermediate bisa digunakan untuk melakukan distilasi. Untuk model dengan ukuran yang jauh berbeda, bagaimana caranya memasangkan antara bagian fitur intermediate tertentu dari teacher dengan bagian fitur intermediate dari student sebagai target distilasi untuk fitur intermediate tersebut?