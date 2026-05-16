---
slug: ch-11-14
title: "DeepSeek-R1"
layout: chapter
---

## DeepSeek-R1: Mendorong Kapabilitas Penalaran pada LLM melalui Pembelajaran Penguatan (Reinforcement Learning)

Penalaran umum (*General reasoning*) merupakan tantangan jangka panjang dan sangat berat dalam bidang kecerdasan buatan. Terobosan terbaru, yang dicontohkan oleh Model Bahasa Besar (*Large Language Models* / LLM) (Brown et al., 2020; OpenAI, 2023) dan pemicuan rantai pemikiran (*chain-of-thought prompting*) (Wei et al., 2022b), telah mencapai kesuksesan yang substansial pada tugas-tugas penalaran dasar. Namun, keberhasilan ini sangat bergantung pada demonstrasi ekstensif yang dianotasi oleh manusia (*human-annotated demonstrations*), dan kapabilitas model saat ini masih belum memadai untuk memecahkan masalah yang lebih kompleks. 

Dalam bab ini, kita akan mendemonstrasikan bahwa **kemampuan penalaran LLM dapat didorong murni melalui Pembelajaran Penguatan (*pure Reinforcement Learning* / RL)**, yang pada akhirnya meniadakan kebutuhan akan lintasan penalaran yang diberi label oleh manusia (*human-labeled reasoning trajectories*). 

Kerangka kerja RL yang diusulkan memfasilitasi kemunculan (*emergent development*) pola penalaran tingkat lanjut, seperti refleksi diri (*self-reflection*), verifikasi, dan adaptasi strategi dinamis. Akibatnya, model yang dilatih ini mencapai kinerja superior pada tugas-tugas yang dapat diverifikasi seperti matematika, kompetisi pemrograman, dan bidang STEM (Sains, Teknologi, Teknik, dan Matematika), melampaui rekan-rekannya yang dilatih melalui pembelajaran terselia konvensional (*conventional supervised learning*) yang menggunakan demonstrasi manusia. Lebih jauh lagi, pola penalaran *emergent* yang ditunjukkan oleh model berskala besar ini dapat dimanfaatkan secara sistematis untuk memandu dan meningkatkan kapabilitas penalaran pada model yang lebih kecil.

---

## 1. Pendahuluan

Kapabilitas penalaran, yang merupakan landasan dari kecerdasan manusia, memungkinkan tugas kognitif yang kompleks mulai dari pemecahan masalah matematika hingga deduksi logis dan pemrograman. Kemajuan terbaru dalam kecerdasan buatan telah menunjukkan bahwa LLM dapat memamerkan perilaku *emergent*, termasuk kemampuan penalaran, ketika diskalakan ke ukuran yang memadai (Kaplan et al., 2020; Wei et al., 2022a). Namun, untuk mencapai kemampuan semacam itu dalam tahap prapelatihan (*pre-training*) biasanya menuntut sumber daya komputasi yang masif. 

Secara paralel, garis penelitian pelengkap telah menunjukkan bahwa LLM dapat ditingkatkan secara efektif melalui pemicuan rantai pemikiran (*chain-of-thought / CoT prompting*). Teknik ini, yang melibatkan penyediaan contoh beberapa tembakan (*few-shot examples*) yang dirancang dengan cermat atau menggunakan pemicu minimalis seperti "Mari kita pikirkan langkah demi langkah" (*"Let’s think step by step"*) (Kojima et al., 2022; Wei et al., 2022b), memungkinkan model untuk menghasilkan langkah penalaran perantara, yang dengan demikian secara substansial meningkatkan kinerjanya pada tugas-tugas yang kompleks. Demikian pula, peningkatan kinerja lebih lanjut telah diamati ketika model mempelajari lintasan penalaran multi-langkah berkualitas tinggi selama fase pasca-pelatihan (*post-training phase*) (Chung et al., 2024; OpenAI, 2023). 

Meskipun efektif, pendekatan ini menunjukkan keterbatasan yang mencolok:
1.  Ketergantungan mereka pada jejak penalaran yang dianotasi manusia menghambat skalabilitas dan memunculkan **bias kognitif**. 
2.  Dengan membatasi model untuk mereplikasi proses pemikiran manusia, kinerjanya secara inheren **dibatasi (*capped*) oleh contoh yang diberikan oleh manusia**, yang mencegah eksplorasi jalur penalaran superior yang mungkin tidak mirip dengan cara berpikir manusia (*non-human-like reasoning pathways*).

Untuk mengatasi masalah ini, kita bertujuan untuk mengeksplorasi potensi LLM dalam mengembangkan kemampuan penalaran melalui **evolusi mandiri (*self-evolution*)** dalam kerangka kerja RL, dengan meminimalkan ketergantungan pada upaya pelabelan manusia. Secara spesifik, kita membangun di atas arsitektur `DeepSeek-V3-Base` (DeepSeek-AI, 2024b) dan menggunakan *Group Relative Policy Optimization* (GRPO) (Shao et al., 2024) sebagai kerangka kerja RL kita. Sinyal penghargaan (*reward signal*) semata-mata didasarkan pada kebenaran prediksi akhir terhadap jawaban yang menjadi kebenaran dasar (*ground-truth answers*), **tanpa memberlakukan batasan pada proses penalaran itu sendiri**. 

Yang perlu diperhatikan, **kita melewati fase Pelatihan Halus Terselia (*Supervised Fine-Tuning* / SFT) konvensional sebelum pelatihan RL**. Pilihan desain ini bermuara dari hipotesis kita bahwa pola penalaran yang didefinisikan manusia dapat membatasi eksplorasi model, sedangkan pelatihan RL tanpa batas (*unrestricted RL training*) dapat lebih baik memotivasi munculnya kemampuan penalaran baru di dalam LLM. Melalui proses ini, yang dirinci pada Bagian 2, model kita (selanjutnya disebut sebagai **DeepSeek-R1-Zero**) secara alami mengembangkan perilaku penalaran yang beragam dan canggih. Dalam memecahkan masalah penalaran, model ini menunjukkan kecenderungan untuk menghasilkan respons yang lebih panjang, menggabungkan verifikasi, refleksi, dan eksplorasi pendekatan alternatif dalam setiap respons. Meskipun kita tidak secara eksplisit mengajari model *bagaimana* cara bernalar, ia berhasil mempelajari strategi penalaran yang ditingkatkan murni melalui pembelajaran penguatan.

Meskipun DeepSeek-R1-Zero mendemonstrasikan kemampuan penalaran yang sangat baik, model ini menghadapi tantangan seperti **keterbacaan yang buruk (*poor readability*)** dan pencampuran bahasa (*language mixing*), yang kadang-kadang menggabungkan bahasa Inggris dan bahasa Mandarin dalam satu respons *chain-of-thought*. Lebih jauh lagi, tahap pelatihan RL berbasis aturan (*rule-based RL*) dari DeepSeek-R1-Zero difokuskan secara sempit pada tugas penalaran, yang menghasilkan kinerja terbatas di area yang lebih luas seperti penulisan dan penjawab pertanyaan domain terbuka (*open-domain question answering*). 

Untuk mengatasi tantangan ini, kita memperkenalkan **DeepSeek-R1**, sebuah model yang dilatih melalui kerangka kerja pembelajaran multi-tahap (*multi-stage learning framework*) yang mengintegrasikan *rejection sampling*, *reinforcement learning*, dan *supervised fine-tuning*, yang dirinci dalam Bagian 3. Alur pelatihan (*training pipeline*) ini memungkinkan DeepSeek-R1 untuk mewarisi kemampuan penalaran dari pendahulunya, DeepSeek-R1-Zero, sembari menyelaraskan (*aligning*) perilaku model dengan preferensi manusia melalui data tambahan yang bukan berupa penalaran (*non-reasoning data*).

Untuk memungkinkan akses yang lebih luas ke AI yang tangguh dengan biaya energi yang lebih rendah, kita telah mendistilasi (*distilled*) beberapa model yang lebih kecil dan menjadikannya tersedia untuk publik. Model-model hasil distilasi ini menunjukkan kemampuan penalaran yang kuat, melampaui kinerja rekan-rekan *instruction-tuned* asli mereka. Kita percaya bahwa versi *instruction-tuned* ini juga akan berkontribusi secara signifikan kepada komunitas riset dengan menyediakan sumber daya yang berharga untuk memahami mekanisme yang mendasari model penalaran CoT panjang dan untuk membina pengembangan model penalaran yang lebih kuat. Rangkaian model DeepSeek-R1 dirilis untuk publik di repositori resmi.

---

## 2. DeepSeek-R1-Zero

Kita mulai dengan menguraikan pelatihan **DeepSeek-R1-Zero**, yang mengandalkan **secara eksklusif pada pembelajaran penguatan (RL) tanpa pelatihan halus terselia (SFT)**. Untuk memfasilitasi efisiensi RL skala besar, kita mengadopsi algoritma *Group Relative Policy Optimization* (GRPO) (Shao et al., 2024).

### 2.1. Group Relative Policy Optimization (GRPO)

GRPO (Shao et al., 2024) adalah algoritma pembelajaran penguatan yang kita adopsi untuk melatih DeepSeek-R1-Zero dan DeepSeek-R1. Algoritma ini pada awalnya diusulkan untuk menyederhanakan proses pelatihan dan mengurangi konsumsi sumber daya dari *Proximal Policy Optimization* (PPO) (Schulman et al., 2017), yang saat ini digunakan secara luas dalam tahap RL pada LLM (Ouyang et al., 2022).

Untuk setiap pertanyaan $q$, GRPO mengambil sampel sekumpulan keluaran (*a group of outputs*) $\{o_1, o_2, \cdots, o_G\}$ dari model kebijakan lama (*old policy model*) $\pi_{\theta_{old}}$ dan kemudian mengoptimalkan model kebijakan (*policy model*) $\pi_\theta$ dengan memaksimalkan fungsi objektif berikut:

$$ \mathcal{J}_{GRPO}(\theta) = \mathbb{E} \left[ q \sim P(Q), \{o_i\}_{i=1}^G \sim \pi_{\theta_{old}}(O|q) \right] $$
$$ \frac{1}{G} \sum_{i=1}^G \left( \min \left( \frac{\pi_\theta(o_i|q)}{\pi_{\theta_{old}}(o_i|q)} A_i, \text{clip}\left( \frac{\pi_\theta(o_i|q)}{\pi_{\theta_{old}}(o_i|q)}, 1 - \varepsilon, 1 + \varepsilon \right) A_i \right) - \beta \mathbb{D}_{KL} (\pi_\theta || \pi_{ref}) \right), \tag{1} $$

$$ \mathbb{D}_{KL} (\pi_\theta || \pi_{ref}) = \frac{\pi_{ref}(o_i|q)}{\pi_\theta(o_i|q)} - \log \frac{\pi_{ref}(o_i|q)}{\pi_\theta(o_i|q)} - 1, \tag{2} $$

**Penjelasan Mendetail Variabel dan Rumus:**
*   $\mathcal{J}_{GRPO}(\theta)$: Fungsi objektif yang ingin **dimaksimalkan** selama pelatihan. Mengoptimalkan $\theta$ (parameter model) agar model menghasilkan jawaban yang menghasilkan *Advantage* ($A_i$) positif tinggi.
*   $\mathbb{E}$: Ekspektasi matematis atas probabilitas distribusi.
*   $q$: Kueri (pertanyaan) yang diambil dari distribusi pertanyaan $P(Q)$.
*   $G$: Ukuran grup (*group size*), yaitu jumlah keluaran/jawaban yang disampel dari model untuk satu pertanyaan $q$.
*   $\{o_i\}_{i=1}^G$: Kumpulan keluaran (jawaban) yang dihasilkan oleh model kebijakan lama $\pi_{\theta_{old}}$.
*   $\frac{\pi_\theta(o_i|q)}{\pi_{\theta_{old}}(o_i|q)}$: Rasio probabilitas model kebijakan saat ini ($\pi_\theta$) dibandingkan model kebijakan lama ($\pi_{\theta_{old}}$) dalam menghasilkan keluaran $o_i$. Jika rasio $> 1$, berarti kebijakan baru lebih mungkin menghasilkan $o_i$.
*   $A_i$: **Keuntungan (*Advantage*)**, yang menunjukkan seberapa baik suatu jawaban spesifik ($o_i$) dibandingkan dengan jawaban lain *di dalam grup yang sama*. Jika $A_i$ positif, model didorong untuk lebih sering menghasilkan jawaban tersebut.
*   $\text{clip}(\cdots, 1 - \varepsilon, 1 + \varepsilon)$: Fungsi pemotongan (*clipping function*) khas metode PPO/GRPO yang membatasi seberapa besar pembaruan kebijakan yang dapat terjadi dalam satu langkah (dikontrol oleh hiperparameter $\varepsilon$). Ini mencegah model melakukan pembaruan yang terlalu ekstrem dan tidak stabil.
*   $\beta$: Koefisien yang mengontrol seberapa ketat model dikenakan penalti.
*   $\mathbb{D}_{KL} (\pi_\theta || \pi_{ref})$: Divergensi Kullback-Leibler, yang merupakan hukuman (*penalty*) jika model kebijakan yang sedang dilatih ($\pi_\theta$) **terlalu menyimpang** dari model referensi ($\pi_{ref}$). Hal ini menjaga agar model tetap menghasilkan teks yang logis (tidak rusak tata bahasanya) selama proses pengejaran hadiah (RL).

Pada GRPO, keuntungan ($A_i$) dikomputasi menggunakan sekelompok penghargaan (*rewards*) $\{r_1, r_2, \ldots, r_G\}$ yang bersesuaian dengan keluaran-keluaran di dalam setiap grup:

$$ A_i = \frac{r_i - \text{mean}(\{r_1, r_2, \cdots, r_G\})}{\text{std}(\{r_1, r_2, \cdots, r_G\})} \tag{3} $$

**Penjelasan Variabel Rumus (3):**
*   $r_i$: Skor penghargaan (*reward*) mentah untuk keluaran $o_i$.
*   $\text{mean}(\cdot)$: Rata-rata dari skor penghargaan di dalam grup tersebut.
*   $\text{std}(\cdot)$: Simpangan baku (*standard deviation*) dari skor penghargaan di dalam grup tersebut.
*   **Signifikansi Akademis:** Rumus ini adalah inti dari "GRPO". Alih-alih melatih model nilai terpisah (*Critic Model/Value Model*) yang boros memori seperti pada PPO standar untuk memperkirakan keuntungan/ *Advantage*, GRPO menormalkan penghargaan *relatif terhadap grupnya sendiri*. Ini secara drastis menghemat memori komputasi VRAM karena kita membuang satu *neural network* raksasa.

Untuk melatih DeepSeek-R1-Zero, kita menetapkan kecepatan pembelajaran (*learning rate*) ke `3e-6`, koefisien KL ($\beta$) ke `0.001`, dan suhu pengambilan sampel (*sampling temperature*) ke `1` untuk peluncuran (*rollout*). Untuk setiap pertanyaan, kita mengambil $G = 16$ sampel keluaran dengan panjang maksimum 32.768 token (sebelum langkah ke-8.2k) dan 65.536 token setelahnya. Hal ini menghasilkan lonjakan kinerja dan panjang respons pada langkah 8.2k. Total iterasi adalah 10.400 langkah (~1,6 epoch).

**Tabel 1: Templat untuk DeepSeek-R1-Zero.**
(*Perhatikan: Model tidak diajari cara bernalar, hanya diminta meletakkan format tag dengan benar*).

```html
A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think>...</think> and <answer>...</answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>. User: prompt. Assistant:
```

### 2.2. Desain Penghargaan (*Reward Design*)

Penghargaan adalah sumber sinyal pelatihan, yang menentukan arah optimasi RL. Untuk DeepSeek-R1-Zero, kita menggunakan **penghargaan berbasis aturan (*rule-based rewards*)** untuk memberikan umpan balik yang tepat untuk data dalam domain matematika, pengkodean, dan penalaran logis.

1.  **Penghargaan Akurasi (*Accuracy rewards*):** Mengevaluasi apakah respons tersebut benar. Misalnya, untuk soal matematika dengan hasil deterministik, model diharuskan memberikan jawaban akhir dalam format tertentu (misalnya, di dalam kotak `\boxed{}`), memungkinkan verifikasi kebenaran berbasis aturan yang sangat andal. Demikian pula, untuk soal pemrograman kompetitif, kompiler (*compiler*) dapat digunakan untuk mengevaluasi kode model terhadap serangkaian kasus pengujian (*test cases*) yang telah ditentukan.
2.  **Penghargaan Format (*Format rewards*):** Melengkapi model hadiah akurasi dengan memberlakukan persyaratan pemformatan tertentu. Khususnya, model diberi insentif untuk mengemas proses penalarannya di dalam tag yang ditentukan, yaitu `<think>` dan `</think>`. 

$$ \text{Reward}_{rule} = \text{Reward}_{acc} + \text{Reward}_{format} \tag{4} $$

**Signifikansi Akademis (Absennya *Reward Model Neural*):** Kita secara sadar **menghindari** penerapan model penghargaan saraf (*neural reward models*)—baik yang berbasis hasil maupun yang berbasis proses—ke tugas-tugas penalaran. Keputusan ini didasarkan pada pengamatan kita bahwa model penghargaan saraf sangat rentan terhadap **peretasan penghargaan (*reward hacking*)** selama pembelajaran penguatan skala besar. Selain itu, melatih ulang model penilai tersebut memerlukan sumber daya komputasi yang substansial dan menyuntikkan kompleksitas yang luar biasa ke dalam alur pelatihan.

### 2.3. Memberi Insentif Kapabilitas Penalaran dalam LLM

Kita menerapkan teknik RL murni pada basis DeepSeek-V3 untuk melatih DeepSeek-R1-Zero. Secara intensional, kita membatasi batasan kita *hanya pada format struktural* (tag `think` dan `answer`), menghindari bias spesifik-konten apa pun, untuk memastikan kita dapat mengamati perkembangan alami model selama proses RL.

#### Analisis Akademik Gambar 1: Kurva Kinerja dan Evolusi Panjang Pemikiran (Thinking Length)

Sebagai pakar, mari kita bedah **Gambar 1** secara analitis yang menunjukkan fenomena *emergent* ini.

*   **Gambar 1 (a) - Akurasi AIME DeepSeek-R1-Zero selama pelatihan:**
    *   **Sumbu X:** Menggambarkan jumlah langkah pelatihan (Steps), dari 0 hingga 10.000. Setiap langkah adalah satu pembaruan kebijakan (*policy update*).
    *   **Sumbu Y:** Menggambarkan Akurasi (*Accuracy*) antara 0,2 hingga 0,9 (20% hingga 90%).
    *   **Kurva Biru (`r1-zero-pass@1`):** Menunjukkan probabilitas model menyelesaikan soal olimpiade matematika AIME dengan benar pada tebakan pertama (*Pass@1*). Terlihat kurva ini naik tajam dari 15,6% di awal dan stabil di kisaran ~77,9%.
    *   **Kurva Merah (`r1-zero-cons@16`):** Menunjukkan akurasi dengan menggunakan metode *consensus* atau pemilihan mayoritas (*majority voting*) dari 16 sampel tebakan. Kurva ini melesat jauh dan stabil di kisaran tinggi (hingga 86,7%).
    *   **Garis Putus-Putus Hijau:** Menunjukkan skor rata-rata peserta manusia dalam kompetisi AIME.
    *   **Kesimpulan:** Grafik ini mendemonstrasikan bahwa RL murni (tanpa SFT manusia) sanggup mendongkrak kejeniusan matematis model secara drastis hingga **melampaui rata-rata manusia**, murni dari mekanisme coba-salah (*trial-and-error*).
*   **Gambar 1 (b) - Rata-rata panjang respons per respons selama pelatihan:**
    *   **Sumbu X:** Langkah pelatihan (Steps).
    *   **Sumbu Y:** Rata-rata jumlah token per respons, dari 0 hingga lebih dari 15.000 token.
    *   **Kurva Biru Bergerigi:** Menunjukkan bahwa saat model dilatih melalui RL, secara instingtif dan tanpa disuruh, ia **menghasilkan token yang jauh lebih panjang (memikirkan lebih lama)**. Dari awalnya hanya ratusan token, meroket hingga lebih dari 10.000 token per jawaban!
    *   **Kesimpulan:** Evolusi mandiri DeepSeek-R1-Zero menunjukkan bahwa RL merangsang pengembangan pemikiran yang sangat panjang, memfasilitasi penalaran yang lebih matang.

Evolusi mandiri DeepSeek-R1-Zero adalah mahakarya keindahan RL: alih-alih secara eksplisit mengajarkan model *cara* memecahkan masalah, kita hanya memberikannya *insentif yang tepat* (hadiah jika jawaban akhirnya benar), dan secara otonom ia mengembangkan strategi pemecahan masalah tingkat lanjut.

**Tabel 2: Contoh momen "Aha!" (*aha moment*) dari versi menengah DeepSeek-R1-Zero**

*(Catatan Dosen: Perhatikan bagaimana model tiba-tiba melakukan refleksi internal)*
```text
Question: If a > 1, then the sum of the real solutions of \sqrt{a - \sqrt{a + x}} = x is equal to
Response: <think>
To solve the equation \sqrt{a - \sqrt{a + x}} = x, let's start by squaring both ...
(\sqrt{a - \sqrt{a + x}})^2 = x^2  => a - \sqrt{a + x} = x^2.
Rearrange to isolate the inner square root term:
(a - x^2)^2 = a + x => a^2 - 2ax^2 + (x^2)^2 = a + x => x^4 - 2ax^2 - x + (a^2 - a) = 0
...
Wait, wait. Wait. That's an aha moment I can flag here.
Let's reevaluate this step-by-step to identify if the correct sum can be ...
...
```
Fenomena "Tunggu, tunggu. Tunggu" (*Wait, wait. Wait*) adalah bukti otentik dari **refleksi-diri (*self-reflection*)**. Model secara mandiri menyadari bahwa metode substitusi polinomial derajat 4 terlalu rumit, membatalkan arah pikirannya, lalu melakukan pemikiran ulang (*rethink*). Ini semua muncul organik dari RL, bukan meniru manusia.

---

## 3. DeepSeek-R1

Meskipun DeepSeek-R1-Zero sangat cerdas, ia memiliki masalah fatal: **Keterbacaan yang buruk (*poor readability*)** dan bahasa yang bercampur (beralih-alih antara Inggris dan Mandarin). Ia juga sangat payah dalam tugas di luar matematika (seperti menjawab pertanyaan umum/sastra) karena ia hanya dilatih mengejar angka "Benar/Salah". 

Untuk mengatasi hal ini, kita memperkenalkan model **DeepSeek-R1** standar, yang alur pelatihannya dijelaskan dengan sangat elegan pada **Gambar 2**.

### Analisis Akademik Gambar 2: Alur Pipa Pembelajaran Multi-Tahap (Multi-stage Pipeline) DeepSeek-R1

Mari kita urai kompleksitas blok-blok pelatihan pada diagram ini.

*   **Deskripsi Visual:** Diagram alir terstruktur secara berurutan, dimulai dari titik kiri atas hingga ke titik kanan bawah, merepresentasikan siklus kehidupan pembentukan DeepSeek-R1.
*   **Warna Legenda:** 
    *   Ungu: Model (*Models*).
    *   Abu-abu: Prompts+Responses.
    *   Biru Tua: Algoritma Pelatihan (*Training Algorithms* - misal SFT, RL).
    *   Biru Muda: Kumpulan pertanyaan (Prompts).
    *   Abu-abu Biru: Sinyal Penghargaan (*Rewards*).
    *   Abu-abu Gelap: Pasca-pemrosesan (*Post-Processing*).
*   **Tahap 1: RL Generasi Pertama (Kiri Atas ke Bawah):** 
    *   Model `DeepSeek V3 Base` dikenakan RL menggunakan *Reasoning Prompts* (Pertanyaan Matematika/Logika) dan diukur dengan skor `Accuracy & Format`. Inilah yang melahirkan **`DeepSeek R1 Zero`** (Model yang sangat pintar tapi tidak bisa diajak komunikasi interaktif).
*   **Tahap 2: SFT Awal (Cold Start SFT) - Titik Tengah Kiri:** 
    *   Kita tidak memakai R1-Zero secara langsung. Sebaliknya, kita melakukan `Sampling` (Pengambilan sampel jawaban bagus) dari R1-Zero, lalu dipasangkan dengan anotasi manusia (`V3+Human`) setelah melewati filter kualitas (`Filter & Format Refine`). 
    *   Data super rapi berjumlah kecil ini (`Cold Start Long CoT`) digunakan untuk melakukan **SFT** (Pelatihan Halus) ke model `DeepSeek V3 Base` yang masih "bersih". Hasilnya adalah model **`DeepSeek R1 Dev-1`**. Model ini cerdas namun kini bicaranya sudah rapi dan mudah dibaca manusia.
*   **Tahap 3: RL Berorientasi Penalaran (Reasoning RL) - Bawah Tengah:**
    *   Model `R1 Dev-1` ini dikenakan RL lagi, tapi kali ini dengan `Rule-based Reward` (Benar/Salah) DAN `Language Consistency Reward` (Hukuman jika ia mencampur bahasa Inggris-Mandarin). Hasilnya adalah **`DeepSeek R1 Dev-2`**.
*   **Tahap 4: SFT Massal Multi-Domain (Kanan Atas):**
    *   Kita melakukan `Sampling` lagi dari `R1 Dev-2` untuk soal-soal penalaran. DI TAMBAH dengan data non-penalaran yang sangat masif (`Non-Reasoning SFT data`) untuk soal-soal sastra, tanya jawab umum, dll.
    *   Gabungan data hibrida raksasa ini diumpankan kembali ke `DeepSeek V3 Base` untuk SFT tahap kedua. Menghasilkan **`DeepSeek R1 Dev-3`**. Model ini kini jago matematika DAN jago menulis esai sastra.
*   **Tahap 5: RL Alignment Manusia (RLHF) - Kanan Bawah:**
    *   `R1 Dev-3` dikenakan RL final menggunakan himpunan *prompt* yang sangat luas (`Diverse Prompts`). 
    *   Sinyal Penghargaannya sekarang hibrida: `Rule-based Reward` (untuk soal eksak) + `Preference Reward Model` (untuk soal sastra, berguna untuk memastikan model sopan, tidak toksik, dan bermanfaat).
    *   Hasil akhirnya adalah mahakarya: **`DeepSeek R1`**.

Secara ringkas, pipa pelatihan ini membuktikan bahwa kita harus membangun kepintaran eksak terlebih dahulu, lalu membungkusnya dengan tata bahasa, memperluasnya ke domain umum, dan menyelaraskannya dengan moral manusia.

### 3.1. Penghargaan Berbasis Model (*Model-based Rewards*)

Untuk data pertanyaan umum (*general data*), aturan "Benar/Salah" (seperti di matematika) tidak dapat digunakan. Apakah puisi itu bagus? Apakah ringkasan itu tepat? Untuk ini, kita terpaksa melatih model terpisah yang disebut **Reward Model**.

1.  **Helpful Reward Model:** Dilatih menggunakan data di mana AI V3 memberi 2 jawaban berbeda, lalu evaluator membedakan jawaban mana yang lebih baik. Arsitektur model penilai ini identik dengan DeepSeek-R1 dengan tambahan lapisan ujung (head) prediksi skalar tunggal.
    $$ \text{Reward}_{helpful} = RM_{helpful} (\text{Response}_A, \text{Response}_B) \tag{5} $$
2.  **Safety Reward Model:** Dilatih secara *point-wise* (menilai jawaban tunggal, bukan pasang-pasangan) untuk mendeteksi unsur bahaya, ujaran kebencian, atau instruksi ilegal.
    $$ \text{Reward}_{safety} = RM_{safety} (\text{Response}) \tag{6} $$

### 3.2. Detail Pelatihan (*Training Details*)

#### 3.2.1. Detail Tahap RL Pertama (Reasoning RL)
Di tahap ini, agar model tidak lagi mencampuradukkan bahasa (Mandarin-Inggris secara acak saat berpikir), kita memperkenalkan **Penghargaan Konsistensi Bahasa (*Language Consistency Reward*)** selama RL. Nilainya dikalkulasi secara sederhana sebagai proporsi kata-kata dari bahasa target (*target language*) di dalam kotak `<think>`.

$$ \text{Reward}_{language} = \frac{\text{Num}(\text{Words}_{target})}{\text{Num}(\text{Words})} \tag{7} $$

Meskipun ini sedikit menurunkan "Akurasi" karena model "dipaksa" merapikan tata bahasa, hal ini sangat vital agar hasil model bisa dibaca oleh manusia.

#### 3.2.2. Detail Tahap RL Kedua (Alignment RL Final)
Tahap akhir RL menggabungkan *semua* sinyal menjadi satu. Rumus fusi penghargaannya adalah:

$$ \text{Reward} = \text{Reward}_{reasoning} + \text{Reward}_{general} + \text{Reward}_{language} \tag{8} $$
$$ \text{where, } \text{Reward}_{reasoning} = \text{Reward}_{rule} \tag{9} $$
$$ \text{Reward}_{general} = \text{Reward}_{reward\_model} + \text{Reward}_{format} \tag{10} $$

**Penjelasan Akademis Rumus 8, 9, 10:**
Pada tahap akhir, jika *prompt*-nya adalah soal matematika (kategori *reasoning*), maka reward yang dipakai murni berasal dari verifikasi *compiler* atau kunci jawaban eksak ($Reward_{rule}$). Sebaliknya, jika *prompt*-nya adalah menyusun email pimpinan (kategori *general*), maka reward diatur oleh $Reward_{reward\_model}$ (Model Preferensi Manusia). Kedua jalur kueri ini secara simultan dikenakan denda jika format balasan atau bahasanya kacau ($Reward_{language}$ dan $Reward_{format}$).

Suhu (*temperature*) pada RL akhir ini diturunkan menjadi `0.7` untuk mencegah model berhalusinasi terlalu liar. RL dilakukan selama 1.700 langkah pembaruan kebijakan.

---

## 4. Eksperimen

Kita mengevaluasi DeepSeek-R1 di berbagai medan pertempuran (tolok ukur akademik tingkat dewa), mulai dari ujian pengacara, ujian logika, hingga kompetisi koding dunia.

*(Sebagai Dosen, saya tidak akan melampirkan seluruh Tabel besar, namun saya akan merangkum dan mengekstrak signifikansi akademis paling brutal dari Tabel 3 makalah).*

**Analisis Hasil Eksperimen Utama (Merujuk Tabel 3 pada Makalah Asli):**
*   Tabel membandingkan kinerja dari kelima generasi model: `R1-Zero`, `R1-Dev1`, `R1-Dev2`, `R1-Dev3`, dan Produk Final `R1`.
*   **Pembuktian Hipotesis Pipa Pelatihan:** 
    *   Pada metrik ujian sastra/format penulisan ketat (contoh: **IFEval**), model asli `R1-Zero` hancur lebur di angka **46.6**. Mengapa? Karena ia hanya peduli matematika. Namun pada produk final `R1`, nilainya melonjak nyaris dua kali lipat menjadi **83.3**. Ini membuktikan kesuksesan injeksi SFT Data Umum pada Tahap 4.
    *   Pada ujian Pemrograman Kompetitif **CodeForces**, produk final `R1` berhasil menembus Rating Elo **2029**, di mana skor ini melampaui **96.3% kompetitor manusia**.
    *   Pada tolok ukur spesifik **AIME 2024** (Kompetisi matematika elit di AS), `R1-Zero` mendapat skor 77.9. Saat masuk tahap `R1-Dev1` (setelah dilatih dengan SFT manusia buatan yang sedikit itu), *skornya anjlok parah ke 59.0*. Ini fenomena yang sangat menarik secara ilmiah! Mengapa? Karena "Gaya Pikir Manusia" yang disuapkan saat SFT ternyata **membonsai** logika berpikir liar AI tersebut. Namun untungnya, setelah dikenakan mesin RL lanjutan (Dev 2, Dev 3), insting kecerdasan liarnya kembali dan produk akhir `R1` menembus angka **79.8**. 

---

## 5. Etika dan Pernyataan Keselamatan (*Ethics and Safety Statement*)

Dengan tingkat kognitif setinggi ini, risiko bahaya (seperti panduan membuat senjata kimia) menjadi nyata. Oleh karena itu, RL juga difokuskan untuk keselamatan model (*Model Safety*).
Peneliti menyuapkan kueri-kueri "menjebak" (*jailbreak attacks*) dan melatih *Safety Reward Model*. Jika model mengeluarkan kata yang menyetujui kueri bahaya tersebut, model dihukum berat secara gradien. 
Hasil evaluasi komparatif menyatakan bahwa DeepSeek-R1, meskipun *open-source*, secara statistik mampu mengimbangi ketangguhan sensor keamanan level atas dari model seperti GPT-4o, dengan tingkat penolakan kueri tak pantas yang memuaskan namun tidak kelewat rewel (*false refusals*).

---

## 6. Kesimpulan, Batasan, dan Penelitian Mendatang (*Conclusion, Limitation, and Future Work*)

Makalah ini mempresentasikan mahakarya eksperimental: **DeepSeek-R1-Zero** dan **DeepSeek-R1**. Eksperimen ini mematahkan keyakinan industri bahwa AI harus disuapi jutaan baris pemikiran manusia. Cukup dengan landasan prapelatihan yang kuat dan algoritma Pembelajaran Penguatan berbasis skala besar (RL), model secara organik dan otonom menumbuhkan perilaku kejeniusan (seperti mengoreksi kesalahan sendiri, melakukan pelacakan balik, dan berpikir ribuan kata).

**Keterbatasan (Limitasi):**
Sebagai saintis, kita harus mengakui batasan:
1.  **Output Terstruktur dan Penggunaan Alat (*Structure Output and Tool Use*):** R1 saat ini masih payah jika diminta mengeluarkan file JSON ketat murni tanpa komentar, dan ia tidak memiliki insting otomatis untuk memanggil mesin pencari (browser) jika ia kekurangan data, berbeda dengan sistem keagenan di GPT-4.
2.  **Efisiensi Token (*Token Efficiency*):** Model R1 seringkali menderita sindrom *overthinking*. Terkadang, untuk pertanyaan sepele seperti "Berapa 1+1?", ia memuntahkan ratusan kata pemikiran reflektif. Belum ada efisiensi kognitif untuk membedakan bobot masalah.
3.  **Pencampuran Bahasa (*Language Mixing*):** Karena R1 secara krusial dioptimasi untuk Mandarin dan Inggris, pertanyaan dalam bahasa Prancis, Indonesia, atau Arab berpotensi memicu ia berpikir (*thinking*) dalam bahasa Inggris dan baru menjawab dalam bahasa asal kueri, menyebabkan komputasi lingual yang gado-gado.
4.  **Sensitivitas Rekayasa Prompt (*Prompting Engineering*):** DeepSeek-R1 benci diberikan "Contoh Jawaban" (*few-shot prompting*). Memberikan AI ini contoh jawaban manusia justru mengkerdilkan akurasinya karena ia terpaksa meniru logika pendek manusia alih-alih melepaskan naluri RL panjangnya. Pendekatan **Zero-shot** (berikan pertanyaan langsung secara lugas tanpa basa-basi format *few-shot*) sangat direkomendasikan.

**Risiko Jangka Panjang (Peretasan Penghargaan / *Reward Hacking*):**
Di masa depan, jika kita melatih AI untuk menulis kode perusahaan yang subjektif, ketiadaan kompilator biner "Benar/Salah" memaksa kita bergantung pada *Reward Model neural network*. AI yang jenius pasti akan menemukan *celah statistik* (*shortcuts*) di *Reward Model* tersebut, menipu penilai untuk mendapatkan skor tinggi tanpa benar-benar menulis kode berkualitas. Skalabilitas metode RL murni pada tugas-tugas non-deterministik (tugas kualitatif) ini merupakan tantangan riset akademis paling berdarah di dekade berikutnya.

*(Penutup Modul Diktat Kuliah DeepSeek-R1)*.
