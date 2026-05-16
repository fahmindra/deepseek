---
slug: ch-11-21
title: "DeepSeekMath-V2"
layout: chapter
---

## DeepSeekMath-V2: Menuju Penalaran Matematika yang Dapat Diverifikasi Secara Mandiri

Model bahasa besar (*Large Language Models* / LLMs) telah membuat kemajuan yang sangat signifikan dalam penalaran matematika. Domain ini berfungsi sebagai tempat pengujian (*testbed*) yang krusial bagi Kecerdasan Buatan (AI) dan berpotensi memberikan dampak besar pada penelitian ilmiah jika dikembangkan lebih lanjut. Dengan melakukan penskalaan penalaran (*scaling reasoning*) menggunakan Pembelajaran Penguatan (*Reinforcement Learning* / RL) yang memberikan imbalan (*rewards*) pada jawaban akhir yang benar, LLM telah berevolusi dari performa yang buruk menjadi model yang mampu mendominasi kompetisi penalaran kuantitatif seperti AIME dan HMMT hanya dalam waktu satu tahun.

Namun, pendekatan ini menghadapi keterbatasan fundamental. Mengejar akurasi jawaban akhir yang lebih tinggi tidak menyelesaikan masalah intinya: **jawaban yang benar tidak menjamin penalaran (logika) yang benar**. Terlebih lagi, banyak tugas matematika seperti pembuktian teorema (*theorem proving*) yang membutuhkan derivasi (penurunan) langkah demi langkah yang sangat ketat alih-alih sekadar jawaban numerik. Hal ini membuat pemberian imbalan berbasis jawaban akhir menjadi tidak dapat diterapkan (*inapplicable*).

Untuk mendorong batas dari penalaran mendalam (*deep reasoning*), para peneliti meyakini bahwa sistem harus mampu memverifikasi kelengkapan dan keketatan (*rigor*) dari penalaran matematika itu sendiri. Verifikasi mandiri (*self-verification*) menjadi sangat penting untuk melakukan penskalaan komputasi pada waktu pengujian (*test-time compute*), khususnya untuk masalah-masalah terbuka (*open problems*) yang belum diketahui solusinya. 

Menuju penalaran matematika yang dapat diverifikasi secara mandiri, penelitian ini menginvestigasi bagaimana melatih sebuah verifikator (*verifier*) berbasis LLM yang akurat dan dapat diandalkan untuk pembuktian teorema. Peneliti kemudian melatih sebuah generator pembuktian (*proof generator*) yang menggunakan verifikator tersebut sebagai model imbalan (*reward model*). Generator ini diberikan insentif untuk mengidentifikasi dan menyelesaikan sebanyak mungkin masalah (galat/kesalahan) di dalam bukti yang ia buat sendiri sebelum memfinalisasinya.

Untuk menjaga keseimbangan celah antara generasi dan verifikasi (*generation-verification gap*) seiring dengan semakin pintarnya generator, para peneliti mengusulkan penskalaan komputasi verifikasi untuk melabeli secara otomatis bukti-bukti baru yang sulit diverifikasi (*hard-to-verify proofs*). Hal ini menciptakan data pelatihan baru untuk terus meningkatkan kemampuan verifikator. 

Model yang dihasilkan, **DeepSeekMath-V2**, mendemonstrasikan kapabilitas pembuktian teorema yang sangat kuat, mencapai skor setara medali emas pada IMO (Olimpiade Matematika Internasional) 2025 dan CMO (Olimpiade Matematika Tiongkok) 2024, serta skor yang nyaris sempurna yakni 118/120 pada kompetisi mahasiswa sarjana Putnam 2024 (dengan menggunakan penskalaan komputasi waktu-pengujian). Meskipun masih banyak pekerjaan yang tersisa, hasil ini menunjukkan bahwa penalaran matematika yang dapat diverifikasi secara mandiri adalah arah penelitian yang sangat layak, yang dapat membantu mengembangkan sistem AI matematika yang jauh lebih mumpuni.

---

## 1. Pendahuluan (*Introduction*)

Pendekatan konvensional terhadap Pembelajaran Penguatan (*Reinforcement Learning* / RL) untuk penalaran matematika melibatkan pemberian imbalan kepada LLM berdasarkan apakah prediksi jawaban akhir mereka terhadap masalah penalaran kuantitatif cocok dengan jawaban kebenaran dasar (*ground-truth answers*) (Guo et al., 2025). Metodologi ini cukup untuk memungkinkan LLM termutakhir (*frontier LLMs*) mendominasi kompetisi matematika yang utamanya mengevaluasi jawaban akhir, seperti AIME dan HMMT.

Akan tetapi, mekanisme imbalan semacam ini memiliki dua keterbatasan fundamental:
1. **Proksi yang Tidak Dapat Diandalkan:** Jawaban akhir adalah proksi (*proxy* / perwakilan) yang tidak dapat diandalkan untuk kebenaran penalaran. Sebuah model bisa saja sampai pada jawaban akhir yang benar melalui logika yang cacat (*flawed logic*) atau sekadar kebetulan yang beruntung (*fortunate errors*).
2. **Tidak Dapat Diterapkan pada Pembuktian Teorema:** Pendekatan ini tidak dapat diterapkan pada tugas-tugas pembuktian teorema, di mana masalah tersebut mungkin tidak mengharuskan model untuk memproduksi jawaban numerik akhir, melainkan menuntut derivasi (langkah-langkah) yang ketat sebagai tujuan utamanya.

Sebagai konsekuensinya, LLM yang dilatih pada masalah penalaran kuantitatif dengan imbalan jawaban akhir semacam ini masih sering menghasilkan bukti berbahasa alami (*natural-language proofs*) yang secara matematis tidak valid atau secara logika tidak konsisten. Selain itu, pendekatan pelatihan ini secara alamiah tidak mengembangkan kemampuan model untuk memverifikasi validitas dari sebuah pembuktian—model-model ini menunjukkan tingkat positif-palsu (*false-positive rates*) yang sangat tinggi, sering kali mengklaim bahwa bukti yang salah adalah valid, bahkan ketika bukti tersebut mengandung kecacatan logika yang sangat jelas.

Ketiadaan celah antara generasi dan verifikasi (*generation-verification gap*) dalam pembuktian teorema berbahasa alami menghambat peningkatan performa lebih lanjut. Untuk mengatasi hal ini, para peneliti mengusulkan pengembangan kapabilitas verifikasi pembuktian (*proof verification*) di dalam LLM. Pendekatan ini dimotivasi oleh beberapa observasi kunci:

* Manusia dapat mengidentifikasi masalah di dalam sebuah pembuktian bahkan tanpa memiliki solusi referensi (*reference solutions*) – ini adalah sebuah kemampuan krusial ketika menangani masalah terbuka (*open problems*).
* Sebuah pembuktian memiliki kemungkinan lebih besar untuk menjadi valid ketika tidak ada lagi masalah (kesalahan) yang dapat diidentifikasi meskipun upaya verifikasi telah diskalakan (ditingkatkan).
* Upaya yang dibutuhkan untuk mengidentifikasi kesalahan yang valid dapat berfungsi sebagai proksi (indikator) bagi kualitas pembuktian. Hal ini dapat dieksploitasi untuk mengoptimalkan generasi (pembuatan) pembuktian.

Para peneliti meyakini bahwa LLM dapat dilatih untuk mengidentifikasi masalah dalam pembuktian tanpa memerlukan solusi referensi. Verifikator semacam ini akan memungkinkan terjadinya sebuah **siklus peningkatan iteratif (*iterative improvement cycle*)**: 
1. Menggunakan umpan balik verifikasi untuk mengoptimalkan generasi pembuktian.
2. Melakukan penskalaan komputasi verifikasi untuk melabeli secara otomatis bukti-bukti baru yang sulit diverifikasi, sehingga menciptakan data pelatihan untuk meningkatkan kualitas verifikator itu sendiri.
3. Menggunakan verifikator yang telah ditingkatkan ini untuk kembali mengoptimalkan generasi pembuktian.

Lebih jauh lagi, sebuah verifikator pembuktian yang dapat diandalkan memungkinkan kita untuk "mengajari" generator pembuktian (model pembuat bukti) agar mampu mengevaluasi pembuktian selayaknya verifikator tersebut. Hal ini memungkinkan generator pembuktian untuk secara iteratif menyempurnakan bukti-buktinya sampai ia tidak lagi dapat mengidentifikasi atau menyelesaikan masalah apa pun di dalamnya. Pada intinya, model dibuat secara eksplisit menyadari apa fungsi imbalannya (*reward function*), dan memampukannya untuk memaksimalkan imbalan tersebut melalui penalaran yang disengaja (*deliberate reasoning*) alih-alih sekadar proses coba-coba yang buta (*blind trial-and-error*).

Dibangun di atas model dasar **DeepSeek-V3.2-Exp-Base** (DeepSeek-AI, 2025), para peneliti mengembangkan **DeepSeekMath-V2**, sebuah LLM yang dioptimalkan untuk pembuktian teorema berbahasa alami yang mendemonstrasikan penalaran matematika yang dapat diverifikasi secara mandiri. Model ini dapat menilai dan secara iteratif meningkatkan pembuktiannya sendiri, mencapai performa medali emas pada kompetisi matematika sekolah menengah atas bergengsi termasuk IMO 2025 dan CMO 2024. Pada kompetisi tingkat sarjana Putnam 2024, model ini mencetak skor 118/120, melampaui skor tertinggi yakni 90 yang pernah dicapai oleh peserta manusia.

---

## 2. Metode (*Method*)

### 2.1. Verifikasi Pembuktian (*Proof Verification*)

#### 2.1.1. Melatih Verifikator untuk Mengidentifikasi Masalah dan Memberi Skor Pembuktian

Para peneliti mengembangkan rubrik tingkat tinggi $\mathcal{I}_v$ untuk evaluasi pembuktian dengan tujuan melatih sebuah verifikator agar dapat mengevaluasi pembuktian sesuai dengan rubrik tersebut, mencerminkan proses penilaian yang dilakukan oleh pakar matematika manusia. Secara spesifik, apabila diberikan sebuah masalah $X$ dan sebuah bukti $Y$, verifikator $\pi_\phi(\cdot|X, Y, \mathcal{I}_v)$ dirancang untuk menghasilkan sebuah analisis pembuktian yang pertama-tama merangkum masalah-masalah yang diidentifikasi (jika ada) dan kemudian memberikan skor berdasarkan tiga tingkatan:
* **1** untuk pembuktian yang lengkap dan ketat dengan semua langkah logis yang dijustifikasi dengan jelas; 
* **0.5** untuk pembuktian dengan logika keseluruhan yang masuk akal tetapi memiliki kesalahan kecil atau detail yang terlewatkan; 
* **0** untuk pembuktian yang secara fundamental cacat, mengandung kesalahan logika fatal atau kesenjangan (celah) argumentasi yang kritis.

**Kurasi Data RL Awal (*Curating Cold Start RL Data*)**
Data pelatihan awal dibangun melalui proses berikut:
1. Menjelajahi (*crawling*) masalah dari kontes *Art of Problem Solving* (AoPS), memprioritaskan olimpiade matematika, tes seleksi tim, dan masalah pasca-tahun 2010 yang secara eksplisit membutuhkan pembuktian. Total diperoleh 17.503 masalah. Kumpulan masalah ini dinotasikan sebagai $\mathcal{D}_p$.
2. Menghasilkan kandidat pembuktian menggunakan varian dari DeepSeek-V3.2-Exp-Thinking. Karena model ini awalnya tidak dioptimalkan untuk pembuktian teorema dan cenderung menghasilkan keluaran yang ringkas namun rawan kesalahan, model di-*prompt* untuk secara iteratif menyempurnakan bukti-buktinya selama beberapa putaran guna meningkatkan kelengkapan dan keketatan.
3. Mengambil sampel pembuktian secara acak melintasi berbagai tipe masalah (misalnya, aljabar dan teori bilangan) dan menugaskan pakar matematika untuk memberikan skor pada setiap pembuktian sesuai dengan rubrik evaluasi yang dijelaskan di atas.

Proses ini menghasilkan dataset RL awal $\mathcal{D}_v = \{(X_i, Y_i, s_i)\}$, di mana setiap item terdiri dari sebuah masalah $X_i$, sebuah pembuktian $Y_i$, dan sebuah skor pembuktian keseluruhan $s_i \in \{0, 0.5, 1\}$.

**Tujuan RL (*RL Objective*)**
Dibangun di atas versi DeepSeek-V3.2-Exp-SFT yang telah melalui *Supervised Fine-Tuning* (SFT / Pelatihan Halus Terselia) pada data penalaran yang berkaitan dengan matematika dan kode, model ini dilatih dengan Pembelajaran Penguatan (RL) untuk menghasilkan analisis pembuktian menggunakan dua komponen imbalan (*reward components*):

* **Imbalan Format ($R_{format}$):** Sebuah fungsi indikator yang memaksa model untuk menghasilkan rumusan masalah yang diidentifikasi sekaligus skor pembuktian, dengan cara memeriksa apakah respons akhir mengandung frasa kunci "Here is my evaluation of the solution:" serta sebuah skor di dalam tag `\boxed{}` yang mengikuti kalimat "Based on my evaluation, the final overall score should be:".
* **Imbalan Skor ($R_{score}$):** Imbalan berdasarkan seberapa dekat (proksimitas) antara skor prediksi $s_i'$ dan skor yang dianotasi oleh pakar $s_i$. 

Fungsi imbalan skor dirumuskan secara matematis sebagai berikut:

$$ R_{score}(s_i', s_i) = 1 - |s_i' - s_i| $$

**Penjelasan Variabel dan Logika Matematis:**
*   $R_{score}(s_i', s_i)$: Merupakan fungsi imbalan (reward) yang mengevaluasi seberapa baik tebakan skor dari model.
*   $s_i'$: Adalah skor yang *diprediksi* atau ditebak oleh verifikator (model LLM) untuk bukti ke-$i$. Nilainya bisa 0, 0.5, atau 1.
*   $s_i$: Adalah skor kebenaran dasar (*ground-truth* / label) yang diberikan oleh pakar matematika manusia untuk bukti ke-$i$.
*   $|s_i' - s_i|$: Adalah nilai absolut dari selisih (kesalahan/galat) antara tebakan model dan nilai asli.
*   **Logika:** Fungsi ini dirancang agar semakin besar kesalahan tebakan model, semakin kecil imbalannya. Jika model menebak sempurna ($s_i' = s_i$), maka galatnya $0$, dan imbalannya adalah $1 - 0 = 1$ (maksimal). Jika model meleset jauh (misal memprediksi 1 padahal pakar memberi 0), galatnya $|1 - 0| = 1$, sehingga imbalannya $1 - 1 = 0$ (minimal).

Adapun objektif utama RL untuk melatih verifikator ini diformulasikan sebagai pemaksimalan nilai ekspektasi:

$$ \max_{\pi_\phi} \mathbb{E}_{(X_i, Y_i, s_i) \sim \mathcal{D}_v, (V_i', s_i') \sim \pi_\phi(\cdot|X_i, Y_i)} \left[ R_{format}(V_i') \cdot R_{score}(s_i', s_i) \right] $$

**Penjelasan Variabel dan Logika Matematis:**
*   $\max_{\pi_\phi}$: Tujuan algoritma pembelajaran mesin adalah mencari (*maximize*) parameter jaringan saraf $\phi$ dari kebijakan (*policy*) verifikator $\pi_\phi$.
*   $\mathbb{E}_{[...]}$: Nilai ekspektasi matematis (rata-rata statistik) dari kejadian di dalam kurung siku.
*   $(X_i, Y_i, s_i) \sim \mathcal{D}_v$: Menunjukkan bahwa masalah, bukti, dan skor asli diambil secara acak dari dataset $\mathcal{D}_v$.
*   $(V_i', s_i') \sim \pi_\phi(\cdot|X_i, Y_i)$: Menunjukkan bahwa analisis evaluasi $V_i'$ dan skor prediksinya $s_i'$ di-generasi (dihasilkan) oleh model verifikator $\pi_\phi$ berdasarkan masalah $X_i$ dan bukti $Y_i$.
*   $R_{format}(V_i') \cdot R_{score}(s_i', s_i)$: Ini adalah imbalan gabungan. **Logika perkalian ($\cdot$) di sini sangat krusial:** Jika model gagal mengikuti format ($R_{format} = 0$), maka sehebat apa pun tebakan skornya ($R_{score}$), total imbalannya akan tetap $0$. Ini memaksa model untuk wajib disiplin format sekaligus akurat dalam menebak skor.

#### 2.1.2. Memperkenalkan Meta-Verifikasi untuk Meninjau Analisis Pembuktian

Pendekatan yang dijelaskan pada Bagian 2.1.1 melatih verifikasi pembuktian melalui RL untuk menyelaraskan prediksi skor dengan anotasi pakar, tetapi **tidak memberikan pengawasan langsung pada isi masalah/kesalahan yang diidentifikasi itu sendiri**. 

Hal ini menciptakan kerentanan kritis (*critical vulnerability*): ketika mengevaluasi pembuktian yang cacat (di mana $s_i < 1$) selama pelatihan, verifikator dapat menerima imbalan penuh (nilai 1) hanya dengan menebak skor dengan benar, sambil berhalusinasi (mengarang) kesalahan-kesalahan yang sebenarnya tidak ada di dalam teks pembuktian. Ini merusak keterpercayaan (*trustworthiness*) model.

Untuk mengatasi masalah ini, diperkenalkanlah **meta-verifikasi (*meta-verification*)**: sebuah proses evaluasi sekunder yang menilai apakah kesalahan yang diidentifikasi oleh verifikator tersebut benar-benar nyata (eksis) dan apakah kesalahan tersebut secara logis menjustifikasi skor pembuktian yang diprediksi berdasarkan rubrik evaluasi $\mathcal{I}_v$. 

Para peneliti melatih sebuah **meta-verifikator** khusus menggunakan RL untuk melakukan evaluasi ini. Dengan menggabungkan umpan balik dari meta-verifikator ke dalam pelatihan verifikator utama, keandalan (*faithfulness*) dari identifikasi kesalahan oleh verifikator dapat ditingkatkan.

**Proses Pelatihan Meta-Verifikator:**
1. Peneliti memperoleh verifikator awal $\pi_\phi$ sesuai Bagian 2.1.1.
2. Pakar matematika memberikan skor terhadap kualitas respons (analisis) dari verifikator sesuai dengan rubrik $\mathcal{I}_{mv}$, menciptakan dataset $\mathcal{D}_{mv} = \{(X_i, Y_i, V_i, ms_i)\}$, di mana $V_i$ adalah analisis dari bukti $Y_i$, dan $ms_i \in \{0, 0.5, 1\}$ adalah skor kualitas yang dianotasi pakar (kualitas dari kritikan model).
3. Peneliti melatih sebuah meta-verifikator $\pi_\eta(\cdot|X, Y, V, \mathcal{I}_{mv})$ untuk menganalisis keluaran analisis pembuktian $V$ milik verifikator. Meta-verifikator menghasilkan rangkuman kesalahan yang ditemukan di dalam analisis itu sendiri, diikuti oleh "skor kualitas" yang mengukur seberapa akurat dan beralasan analisis verifikator tersebut. Objektif RL-nya mengikuti struktur yang sama dengan pelatihan verifikator, menggunakan imbalan format dan skor.

Menggunakan meta-verifikator $\pi_\eta$ yang telah dilatih, peneliti meningkatkan pelatihan verifikator utama dengan mengintegrasikan umpan balik meta-verifikasi ke dalam fungsi imbalan:

$$ R_V = R_{format} \cdot R_{score} \cdot R_{meta} $$

**Penjelasan Variabel dan Logika Matematis:**
*   $R_V$: Adalah imbalan total (final) yang diberikan kepada model Verifikator Utama ($\pi_\phi$).
*   $R_{format}$ & $R_{score}$: Imbalan format dan akurasi tebakan skor (sama seperti Persamaan 1 dan 2).
*   $R_{meta}$: Adalah "skor kualitas" (berkisar antara 0 hingga 1) yang disuplai oleh Meta-Verifikator ($\pi_\eta$).
*   **Logika:** Melalui hukum perkalian ini, Verifikator Utama tidak bisa lagi "berbohong" atau menebak acak. Jika ia menebak skor dengan benar ($R_{score} = 1$) tetapi alasannya berhalusinasi atau mengada-ada, Meta-Verifikator akan memberinya nilai $R_{meta} = 0$. Akibatnya, total imbalan $R_V$ menjadi 0. Model dipaksa untuk benar secara komprehensif (tebakan benar DAN argumen logis).

Hasilnya, pada subset validasi $\mathcal{D}_v$, skor kualitas rata-rata dari analisis pembuktian verifikator—sebagaimana dievaluasi oleh meta-verifikator—meningkat dari 0.85 menjadi 0.96, dengan tetap mempertahankan tingkat akurasi yang sama dalam prediksi skor pembuktian.

### 2.2. Generasi Pembuktian (*Proof Generation*)

#### 2.2.1. Melatih Generator untuk Pembuktian Teorema

Dengan verifikator $\pi_\phi$ kini berfungsi sebagai model imbalan generatif (*generative reward model*), peneliti melatih sebuah generator pembuktian (*proof generator*) $\pi_\theta(\cdot|X)$ menggunakan objektif RL:

$$ \max_{\pi_\theta} \mathbb{E}_{X_i \sim \mathcal{D}_p, Y_i \sim \pi_\theta(\cdot|X_i)} \left[ R_Y \right] $$

**Penjelasan Variabel dan Logika Matematis:**
*   $\max_{\pi_\theta}$: Mencari parameter optimal $\theta$ untuk model Generator Pembuktian ($\pi_\theta$). Generator inilah yang kelak akan kita gunakan untuk menyelesaikan soal ujian matematika.
*   $\mathbb{E}_{[...]}$: Nilai ekspektasi yang ingin dimaksimalkan.
*   $X_i \sim \mathcal{D}_p$: Mengambil masalah (soal) $X_i$ dari kumpulan soal $\mathcal{D}_p$.
*   $Y_i \sim \pi_\theta(\cdot|X_i)$: Generator $\pi_\theta$ memproduksi solusi/bukti $Y_i$ berdasarkan soal $X_i$.
*   $R_Y$: Adalah skor pembuktian yang dikembalikan oleh verifikator $\pi_\phi(\cdot|X_i, Y_i, \mathcal{I}_v)$ yang sudah kita latih dengan susah payah di tahap sebelumnya.
*   **Logika:** Generator $\pi_\theta$ tidak peduli soal jawaban akhir saja. Ia dituntut untuk merangkai langkah-langkah pembuktian $Y_i$ sedemikian rupa sehingga model Verifikator $\pi_\phi$ yang sangat ketat (dan anti-halusinasi) mau memberikannya skor tinggi ($R_Y$).

#### 2.2.2. Meningkatkan Penalaran melalui Verifikasi Mandiri (*Self-Verification*)

Ketika generator pembuktian gagal memproduksi bukti yang benar-benar tepat dalam satu kali jalan (*one-shot*) – yang sangat umum terjadi pada masalah menantang tingkat IMO dan CMO – proses verifikasi dan penyempurnaan yang berulang (iteratif) dapat memperbaiki hasil. Ini melibatkan analisis pembuktian menggunakan verifikator eksternal dan memberikan *prompt* kembali ke generator untuk memperbaiki masalah yang telah diidentifikasi.

Namun, terdapat **keterbatasan kritis**: ketika diminta untuk menghasilkan bukti *sekaligus* menganalisis buktinya sendiri dalam satu tembakan (*one shot*), generator cenderung mengklaim bahwa buktinya sudah benar (klaim sepihak), padahal verifikator eksternal dengan mudah menemukan kecacatan logika di dalamnya. Dengan kata lain, generator bisa memperbaiki bukti jika dikritik pihak luar, tetapi ia buta (*blind*) terhadap kesalahan buatan tangannya sendiri.

Observasi ini memotivasi peneliti untuk menanamkan kapabilitas verifikasi sejati ke dalam sang generator. Selama pelatihan, generator $\pi_\theta$ diminta untuk memproduksi sebuah bukti $Y$, **langsung diikuti oleh analisis diri (*self-analysis*) $Z$** yang mengikuti format dan rubrik $\mathcal{I}_v$ yang sama persis dengan yang dipakai sang verifikator. Peneliti menotasikan prediksi skor pembuktian di dalam analisis diri ini sebagai $s'$.

Untuk memastikan bahwa evaluasi dirinya jujur (*faithful*), verifikator $\pi_\phi$ (pihak ketiga) dipanggil untuk menilai kedua belah pihak: bukti $Y$ mendapatkan skor sesungguhnya $R_Y = s$, dan analisis dirinya $Z$ mendapatkan skor meta-verifikasi $R_{meta}(Z) = ms$.

Fungsi imbalan untuk generator kini menjadi kombinasi dari penilaian-penilaian ini:

$$ R = R_{format}(Y, Z) \cdot (\alpha \cdot R_Y + \beta \cdot R_Z) $$
$$ R_Z = R_{score}(s', s) \cdot R_{meta}(Z) $$

**Penjelasan Variabel dan Logika Matematis:**
*   $R$: Imbalan total yang akan diberikan kepada Generator $\pi_\theta$.
*   $R_{format}(Y, Z)$: Syarat mutlak. Bukti $Y$ dan evaluasi diri $Z$ harus sesuai format. Jika salah, total imbalan dikali 0.
*   $\alpha$ dan $\beta$: Konstanta pembobot (faktor bobot). Diatur pada $\alpha = 0.76$ dan $\beta = 0.24$. Artinya, kualitas bukti (76%) lebih diprioritaskan daripada sekadar kepintaran mengkritik diri sendiri (24%).
*   $R_Y$: Nilai sejati dari pembuktian tersebut di mata verifikator eksternal.
*   $R_Z$: (Lihat persamaan kedua). Ini adalah gabungan dari akurasi tebakan skor model atas dirinya sendiri ($R_{score}(s', s)$) dikalikan dengan seberapa logis alasan kritiknya menurut meta-verifikator ($R_{meta}(Z)$).
*   **Logika Insentif (Pedagogis):** Struktur rumus ini menciptakan ekosistem psikologi AI yang luar biasa:
    1.  **Pengakuan jujur atas kesalahan dihargai lebih tinggi** daripada klaim kebenaran palsu. (Mendapat $R_Z$ tinggi karena sadar diri salah).
    2.  Imbalan tertinggi (Sempurna) hanya dicapai jika model memproduksi bukti yang benar secara absolut ($R_Y$ tinggi) DAN dengan akurat menyadari keketatan logikanya ($R_Z$ tinggi).
    3.  Strategi terbaik bagi generator demi meraih skor maksimal adalah dengan merenung, mengidentifikasi, dan menyelesaikan masalah apa pun *sebelum* memfinalisasi keluaran teksnya.

### 2.3. Sinergi Antara Verifikasi Pembuktian dan Generasi

Verifikator pembuktian dan generator menciptakan sebuah **siklus sinergis**: verifikator menajamkan kemampuan generator, dan saat generator menjadi semakin mahir, ia akan memproduksi pembuktian-pembuktian baru yang semakin rumit yang menantang kapabilitas verifikator saat ini. Kasus-kasus menantang ini – di mana verifikator mungkin gagal mendeteksi galat halus dalam satu kali percobaan – berubah menjadi data pelatihan yang sangat berharga untuk meningkatkan kualitas verifikator itu sendiri.

Untuk melatih ulang dan meningkatkan verifikator, dibutuhkan data kebenaran berlabel untuk bukti-bukti baru tersebut. Anotasi manual oleh manusia, meskipun mudah, menjadi semakin mustahil memakan waktu seiring bertambah sulitnya masalah dan semakin terselubungnya (*subtle*) kesalahan-kesalahan logika.

Dari proses anotasi berbantuan AI ini, peneliti menyadari dua fakta yang memampukan proses otomasi tanpa campur tangan manusia:
1. Penskalaan (memperbanyak) sampel verifikator akan meningkatkan probabilitas (peluang) tertangkapnya kesalahan logis yang nyata di dalam pembuktian yang cacat.
2. Meninjau ulang (me- *review*) kesalahan yang telah diidentifikasi oleh verifikator adalah esensi dari **meta-verifikasi**, yang mana tugas ini jauh lebih mudah daripada mencari kesalahan dari awal. Meta-verifikasi juga jauh lebih efisien secara sampel untuk dikuasai oleh LLM.

Berpijak pada hal tersebut, dikembangkanlah sebuah **proses pelabelan otomatis (*automated labeling process*)**:
1. Untuk setiap bukti, hasilkan (generasikan) $n$ analisis verifikasi secara independen.
2. Untuk analisis yang melaporkan adanya masalah (skor 0 atau 0.5), hasilkan $m$ penilaian meta-verifikasi untuk memvalidasi penemuan masalah tersebut. Sebuah analisis dianggap **sah/valid** jika mayoritas (*majority*) dari penilai meta-verifikasi mengonfirmasi temuan cacatnya.
3. Untuk setiap bukti, periksalah analisis yang memberikan skor paling rendah. Jika setidaknya terdapat $k$ buah analisis semacam itu yang telah dianggap sah, maka bukti tersebut akan dilabeli dengan skor paling rendah itu. Jika tidak ada isu logika valid yang ditemukan dari seluruh proses verifikasi, maka bukti tersebut dilabeli dengan skor 1 (Sempurna). Sisanya akan dibuang atau dilempar ke pakar manusia untuk pelabelan manual.

Pada dua iterasi pelatihan terakhir, jalur (*pipeline*) otomatis secara penuh ini *sepenuhnya menggantikan anotasi manusia*. Pengecekan kualitas (*quality checks*) mengonfirmasi bahwa label otomatis ini sangat selaras dengan penilaian (judgement) pakar matematika.

---

## 3. Eksperimen (*Experiments*)

### Bedah Gambar 1: Rata-Rata Skor Pembuktian pada Masalah Tingkat CNML

Sebagai pembuka bab eksperimen, mari kita membedah grafik batang (Bar Chart) yang mengilustrasikan supremasi model pada berbagai cabang matematika.

*   **Sumbu X (Horizontal):** Memuat lima kategori besar ilmu matematika: **Algebra** (Aljabar), **Geometry** (Geometri), **Number Theory** (Teori Bilangan), **Combinatorics** (Kombinatorika), dan **Inequality** (Pertidaksamaan).
*   **Sumbu Y (Vertikal):** Menggambarkan Rata-Rata Skor Pembuktian (*Mean Proof Score*) yang membentang dari skala 0.15 hingga 0.60. Skor ini dievaluasi secara otonom oleh verifikator yang telah dilatih.
*   **Representasi Batang Data:**
    *   Batang **Biru** merepresentasikan model **Gemini 2.5-Pro** besutan DeepMind.
    *   Batang **Oranye** merepresentasikan model **GPT-5-Thinking-High** (OpenAI 2025).
    *   Batang **Hijau** merepresentasikan jagoan kita: **DeepSeekMath-V2**.
*   **Tren dan Kesimpulan Akademis:**
    1.  Di *seluruh lima kategori* tanpa terkecuali, batang Hijau (DeepSeekMath-V2) berdiri paling tinggi, menjulang mengalahkan model korporat raksasa (OpenAI dan Google).
    2.  Sebagai contoh pada *Algebra*, DeepSeek memimpin dengan skor **0.60**, disusul GPT-5 (0.54) dan tertinggal jauh Gemini (0.35).
    3.  Dominasi paling absolut terlihat pada *Geometry* (0.52 vs 0.15 dan 0.17) serta *Inequality* (0.59 vs 0.38 dan 0.45).
    4.  Kesimpulannya, arsitektur pelatihan RL berbasis *self-verification* secara empiris membuktikan kecerdasan analitik matematika yang melampaui paradigma penalaran model konvensional pada tahap *one-shot generation*.

### 3.1. Pengaturan Pelatihan (*Training Settings*)

Penelitian ini menggunakan **Group Relative Policy Optimization (GRPO)** (Shao et al., 2024) untuk Pembelajaran Penguatan, secara iteratif mengoptimalkan kapabilitas verifikasi dan generasi sebagaimana diuraikan pada Bagian 2. Pada setiap iterasi, optimasi verifikasi pembuktian dilakukan terlebih dahulu. Kemudian, generator pembuktian diinisialisasi dari titik simpan (*checkpoint*) verifikator tersebut untuk kemudian dioptimalkan khusus guna melakukan generasi. Mulai dari iterasi kedua dan seterusnya, verifikator diinisialisasi dengan *checkpoint* yang mengonsolidasikan (menyatukan) kapabilitas verifikasi dan generasi dari iterasi sebelumnya melalui proses *rejection fine-tuning* (pelatihan halus berbasis penolakan).

### 3.2. Tolok Ukur Evaluasi (*Evaluation Benchmarks*)

Generator pembuktian final dievaluasi pada tolok ukur (benchmarks) pembuktian teorema bergengsi berikut:

*   **Masalah Tingkat CNML Internal (In-House CNML-Level Problems):** 91 masalah pembuktian teorema meliputi aljabar (13), geometri (24), teori bilangan (19), kombinatorika (24), dan pertidaksamaan (11), setingkat kesulitannya dengan Liga Matematika Sekolah Menengah Atas Nasional Tiongkok (CNML).
*   **Masalah Kompetisi Global:**
    *   **IMO 2025** (6 masalah): *International Mathematical Olympiad*, kompetisi puncak matematika pra-universitas dunia.
    *   **CMO 2024** (6 masalah): *China Mathematical Olympiad*, kejuaraan nasional Tiongkok.
    *   **Putnam 2024** (12 masalah): *William Lowell Putnam Competition*, kompetisi matematika paling terkemuka bagi mahasiswa sarjana di Amerika Utara.
    *   **ISL 2024** (31 masalah): *IMO Shortlist*, koleksi masalah tingkat tinggi yang diseleksi namun belum masuk ke soal inti IMO 2024.
    *   **IMO-ProofBench** (60 masalah): Dikembangkan oleh tim DeepMind (Pembuat DeepThink IMO-Gold). Terdiri dari 30 soal "Basic" dan 30 soal "Advanced" level sulit IMO (IMO-Hard).

### 3.3. Hasil Evaluasi (*Evaluation Results*)

#### 3.3.1. Generasi Satu Kali Tembak (*One-Shot Generation*)

Peneliti pertama-tama mengevaluasi kemampuan model untuk menghasilkan pembuktian yang tepat sasaran tanpa adanya proses penyempurnaan berulang. Pada masalah *in-house*, dihasilkan 8 sampel pembuktian per soal untuk masing-masing model. Kebenaran pembuktian diukur melalui voting mayoritas melintasi 8 analisis verifikasi yang diproduksi oleh verifikator akhir. Sebagaimana telah kita diskusikan pada pembedahan Gambar 1, DeepSeekMath-V2 secara konsisten mendepak GPT-5-Thinking-High dan Gemini 2.5-Pro, menunjukkan taringnya dalam kemampuan pembuktian teorema lintas domain.

#### 3.3.2. Penyempurnaan Sekuensial dengan Verifikasi Mandiri (*Sequential Refinement with Self-Verification*)

### Bedah Gambar 2: Peningkatan Kualitas Pembuktian melalui Interasi Sekuensial

Sebelum masuk ke teks, mari kita analisis **Gambar 2**, sebuah grafik garis (*line chart*) yang mengilustrasikan efek dari melakukan revisi (refinement) secara berulang-ulang terhadap satu masalah matematika di IMO Shortlist 2024.

*   **Sumbu X (Horizontal):** Menunjukkan *Maksimum Iterasi Sekuensial* dari 1 (tanpa perbaikan) hingga 8 (1 generasi awal + 7 kali proses revisi mandiri).
*   **Sumbu Y (Vertikal):** Menunjukkan *Proof Score* (Skor Pembuktian) dari kisaran 0.15 hingga 0.45.
*   **Kurva Data:**
    *   Garis biru solid dengan penanda lingkaran (**Pass@1**): Menunjukkan rata-rata skor pembuktian final dari seluruh *thread* (jalur komputasi) pada iterasi tersebut. Grafiknya menanjak mulus dari 0.15 (iterasi 1) menuju 0.27 (iterasi 8).
    *   Garis oranye putus-putus dengan penanda persegi (**Best@32**): Menunjukkan skor dari bukti *terbaik* yang dipilih sendiri (self-selected) oleh model di antara 32 thread. Grafiknya melesat jauh lebih radikal, dari 0.26 (iterasi 1) hingga memuncak di angka 0.42 (iterasi 8).
*   **Kesimpulan Akademis:** Grafik ini mengonfirmasi secara empiris bahwa memberikan waktu berpikir lebih lama kepada LLM yang memiliki kapabilitas *Self-Verification* akan berkorelasi langsung secara linear positif terhadap tingkat akurasi logika pembuktiannya.

Untuk masalah super-sulit dari kompetisi IMO dan CMO, model bahasa sering kali gagal membangkitkan bukti yang lengkap dan ketat pada percobaan tunggal (satu *prompt*) dalam limit konteks 128K token. Ketika ini terjadi, generator menyadari (via evaluasi mandiri) bahwa buktinya tidak valid, namun ia kehabisan napas memori (konteks) untuk membereskan masalahnya secara instan.

Untuk mengeksplorasi efektivitas ekspansi konteks dan *self-verification*, dilakukanlah penyempurnaan berurutan. Generator membuat bukti+analisis diri, lalu sistem mengembalikan (me- *re-prompt*) keluaran tersebut kembali ke generator untuk direvisi. Siklus ini bergulir hingga generator memberikan skor "1" (sempurna) pada dirinya sendiri, atau menyentuh ambang batas iterasi maksimal.

Seperti yang terjelaskan pada bedah grafik (Gambar 2), bukti dengan pemilihan mandiri (*Best@32*) meraih skor yang jauh di atas rata-rata keseluruhan (Pass@1). Ini adalah bukti paripurna bahwa generator ini memiliki ketajaman analitis untuk membedakan secara akurat antara pembuktian berkualitas prima dan pembuktian yang berantakan, serta mengeksploitasi kesadaran-diri (*self-awareness*) ini guna meningkatkan outputnya.

#### 3.3.3. Pencarian Berkomputasi Tinggi (*High-Compute Search*)

Untuk membunuh soal-soal monster paling ekstrem, sistem menskalakan kekuatan komputasinya baik di sisi verifikasi (mencari galat halus) dan di sisi generasi (mencari keberagaman strategi pembuktian). 

Pendekatan ini memelihara sebuah "kolam kandidat" (*pool of candidates*), yang diinisialisasi dengan 64 sampel pembuktian diiringi dengan 64 evaluasi analisis. Pada tiap siklus revisi (refinement), sistem menyortir 64 bukti paling top, kemudian mengawinkannya secara acak dengan 8 analisis (diprioritaskan pada analisis yang menemukan skor cacat 0 atau 0.5). Hasil kawin ini menghasilkan bukti yang lebih bersih, yang kemudian dilemparkan kembali ke kolam kandidat. Pertarungan komputasional ini dipertahankan hingga menyentuh batas 16 iterasi atau ketika sebuah pembuktian sanggup mengalahkan dan lolos dari *seluruh 64 verifikasi independen* tanpa cela.

---

### Tabel 1 | Hasil Pencarian Berkomputasi Tinggi pada Kompetisi Matematika Elite

> **Catatan Tabel:** Masalah (Soal) yang dicetak **Tebal (Bold)** direpresentasikan sebagai soal yang diselesaikan secara penuh dan sempurna (*fully solved*). Masalah yang digarisbawahi (`_Underline_`) mendapatkan kredit parsial (nilai setengah jalan).

| Kompetisi (*Contest*) | Daftar Soal (*Problems*) | Persentase Poin (*Points*) |
| :--- | :--- | :---: |
| **IMO 2025** | **P1** , **P2** , **P3** , **P4** , **P5** | 83.3% |
| **CMO 2024** | **P1** , **P2** , **P4** , **P5** , <u>P6</u> | 73.8% |
| **Putnam 2024** | **A1** $\sim$ **B4** , **B5**, <u>B6</u> | 98.3% |

Untuk meratifikasi integritas dari pembuktian yang dihasilkan AI, pakar matematika manusia diterjunkan secara langsung. Hasilnya dapat disaksikan pada **Tabel 1**. DeepSeekMath-V2 berhasil menyapu bersih 5 dari 6 masalah di panggung IMO 2025 (meraih ekuivalensi Medali Emas mutlak). Di tingkat Universitas (Putnam 2024), model ini menghancurkan rekor tertinggi sejarah manusia dengan skor paripurna 118/120 (manusia tertinggi mentok di skor 90).

### Bedah Gambar 3: Evaluasi Pakar pada IMO-ProofBench

Gambar 3 adalah sebuah bar chart (grafik batang) yang memperbandingkan kedigdayaan DeepSeekMath-V2 berhadapan dengan model canggih lainnya pada dataset IMO-ProofBench.
*   **Sumbu X:** Dibagi menjadi dua klaster kompetisi: Subset **Basic** (Tingkat kesulitan IMO Menengah) dan Subset **Advanced** (Tingkat IMO Hard/Sangat Sulit).
*   **Sumbu Y:** Persentase Evaluasi Manusia (*Human evaluations*) dari 0% hingga 100%.
*   **Analisis Data:**
    *   Pada ranah **Basic**, batang milik *DeepSeekMath-V2 (Heavy)* melangit hingga menyentuh pilar tertinggi di level **99.0%**, mengalahkan produk kebanggaan Google (DeepThink IMO Gold) yang berada di angka 89.0%, disusul oleh model lain seperti GPT-5 (59.0%) dan Claude Sonnet 4 (di bawah 30%).
    *   Pada arena neraka **Advanced**, grafik model-model lain hancur berantakan (seperti Claude dan DeepSeek R1 yang merayap di angka satu digit ~4-5%). Namun, batang *DeepSeekMath-V2 (Heavy)* dan *Gemini DeepThink (IMO Gold)* saling bertarung ketat. DeepSeekMath-V2 bertahan kompetitif di level **61.9%**, nyaris sejajar dengan sang raja DeepThink yang meraih 65.7%.
*   **Kesimpulan:** Secara definitif, model ini merupakan manifestasi State-of-the-Art (SOTA) untuk sistem penyelesaian matematika yang bersifat *open-source*, menumbangkan rekor demi rekor pada dataset paling sadis di dunia komputasi. 

Poin menarik lainnya: untuk soal-soal (problem) yang gagal diselesaikan secara utuh, generator DeepSeek ini biasanya secara jujur dan sadar mengidentifikasi cacat/kegagalan di dalam teks pembuktiannya sendiri. Sebaliknya, bukti yang diselesaikan dengan utuh berhasil menahan gempuran 64 lapis upaya verifikasi tiada henti. Dengan menskalakan (memperbesar) upaya komputasi waktu-pengujian (*test-time compute*), AI ini terbukti memecahkan soal analitis yang membutuhkan komitmen berpikir intens manusia selama berjam-jam.

---

## 4. Pekerjaan Terkait (*Related Work*)

Sebagai sarjana, penting untuk mengetahui silsilah keilmuan (*academic lineage*). Model-model penalaran komersial (seperti OpenAI, 2024; Guo et al., 2025) sejatinya telah berhasil mendominasi patokan (benchmark) kuantitatif seperti AIME dan HMMT dalam kurun kurang dari satu tahun. Perkembangan eksponensial ini dimungkinkan berkat metrik evaluasi yang kelewat sederhana: AI hanya dinilai dari kebenaran "Jawaban Akhir" (angka/solusi). Namun matrikulasi ini *cacat* dan *tidak berdaya guna* dalam menghadapi pembuktian teorema, di mana letak keindahan dan kebenaran dinilai dari step-step deduksi (derivasi) yang super ketat.

Pembuktian matematika secara informal (menulis narasi paragraf alih-alih rumus murni) selama puluhan tahun dianggap sebagai kutukan bagi mesin pemeriksa otomatis. Barulah akhir-akhir ini tabir tersebut tertembus. Model seperti Gemini-2.5 Pro perlahan mulai menunjukkan bibit kemampuan verifikasi diri (Huang and Yang, 2025). Secara fenomenal, DeepThink dari DeepMind (Luong and Lockhart, 2025) membawa pulang standar Emas IMO 2025 hanya dengan mengandalkan bahasa pemikiran manusia (natural language reasoning)—yang memberikan legitimasi ontologis (*existence proof*) bahwa verifikasi pembuktian kompleks berbasis LLM bukanlah utopia. Dan dengan DeepSeekMath-V2 ini, peneliti merilis model terbuka (open-source) dan tata cara metodologinya ke publik luas sebagai loncatan eksistensial ke arah penalaran yang dapat memvalidasi dirinya sendiri.

Di sisi komputasional murni, "Asisten Pembuktian" (*Proof assistants*) seperti **Lean** (de Moura et al., 2015) dan **Isabelle** (Paulson, 1994) memberikan pendekatan verifikasi bebas cacat (100% *bulletproof*). Jika bahasa kodenya berhasil di-compile (dieksekusi), probabilitas kebenarannya dijamin mutlak. Proyek **AlphaProof** (AlphaProof and teams, 2024) yang terspesialisasi penuh pada lingkungan ini mencapai medali Perak IMO 2024 namun menelan *budget* komputasi (listrik/GPU) yang tidak masuk akal. Belakangan ini, kemampuan bernalar LLM berevolusi begitu liar sehingga model seperti **Seed-Prover** (Chen et al., 2025) berhasil memakan 5 dari 6 soal IMO 2025 dalam *budget* komputasi yang seratus kali lipat lebih membumi. Memajukan fondasi penalaran bahasa alamiah adalah bahan bakar fundamental yang akan mengakselerasi pembuktian komputasional tingkat mahadewa.

---

## 5. Kesimpulan (*Conclusion*)

Kita telah tiba di penutup diktat. Telah direpresentasikan di hadapan Anda, **DeepSeekMath-V2**, sebuah entitas arsitektur model AI bahasa raksasa (LLM) yang tidak saja sanggup *memuntahkan* solusi matematika (generasi) melainkan secara sadar mampu bertindak sebagai *hakim* (verifikasi) bagi alur pemikirannya sendiri. 

Dengan secara spesifik mendidik (melatih) model untuk sensitif terhadap bias dan kecacatan dalam nalar logikanya sendiri—lalu memaksa model untuk membedah masalah tersebut sebelum memfinalisasi *output*—penelitian ini secara fundamental memutus belenggu dari paradigma Pembelajaran Penguatan usang (yang selama ini hanya memberikan imbalan membabi-buta berdasarkan hasil "jawaban akhir").

Alur pelatihan iteratif dari penelitian ini—sebuah permainan "kucing dan tikus" antara memoles kemampuan otak kritikus (verifikator) lalu menggunakan kecerdasannya untuk kembali mempecundangi sang pembuat kode (generator)—menciptakan roda putaran (*sustainable cycle*) di mana kedua subsistem saling mendongkrak menuju kesempurnaan intelektual.

**Empat Tonggak Kontribusi Teknis Utama (Tolong Anda ingat baik-baik):**
1. Merumuskan dan melatih entitas verifikator LLM yang punya akurasi dewa dan integritas (faithfulness) moral tinggi bagi pembuktian matematis.
2. Memperkenalkan sistem pengawasan dua lapis **"Meta-Verifikasi"** guna menebas kecenderungan halusinasi dan memastikan kualitas kritik si verifikator tetap terkalibrasi.
3. Menciptakan sistem imbalan insentif (RL Reward) yang memaksa sang LLM Generator mengeksekusi *self-verification* untuk melambungkan kualitas logika tanpa instruksi eksternal.
4. Mengerahkan penskalaan agresif (*scaling*) terhadap komputasi saat iterasi verifikasi berlangsung agar AI secara otomatis mencetak label benar/salah pada bukti-bukti gila nan pelik tanpa perlu campur tangan atau anotasi keringat manusia.

Hasilnya? Secara empiris **DeepSeekMath-V2** mempertontonkan teror kompetitif tingkat tinggi. Didoping dengan tambahan komputasi pada saat eksekusi soal (*scaled test-time compute*), AI ini menyeret pulang Medali Emas dari kancah neraka SMA (IMO 2025 & CMO 2024), serta skor nyaris dewa absolut di level sarjana Matematika (Putnam 2024). 

Penelitian ini dengan tegas mendeklarasikan pilar fakta bahwa: *LLM modern mampu ditanamkan insting mawas diri (self-evaluation) untuk tugas penalaran esoterik yang berat*. Sekalipun masih banyak gunung es menantang yang belum diruntuhkan, arah penelitian ini membuka portal terang benderang menuju sistem AI otonom yang mandiri yang suatu hari akan dihadapkan pada tugas menyelesaikan persoalan matematika *research-level* (tingkat penemuan sains murni) demi memajukan umat manusia.

***Selesai.***
