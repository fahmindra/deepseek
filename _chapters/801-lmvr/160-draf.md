---
slug: lmvr-16
title: lmvr-16
---
# Bab 16: LLM dalam Bidang Hukum dan Kepatuhan

> **"Penerapan AI dalam konteks hukum memerlukan keseimbangan yang halus antara inovasi dan tanggung jawab, memastikan bahwa teknologi meningkatkan alih-alih merusak prinsip-prinsip keadilan dan kejujuran." — Fei-Fei Li**

Bab 16 dari LMVR mengeksplorasi penerapan model bahasa besar (LLM) dalam sektor hukum dan kepatuhan (*compliance*), membahas tantangan dan peluang unik di bidang-bidang ini. Bab ini mencakup seluruh proses, mulai dari membangun pipeline data khusus dan melatih model pada data hukum serta regulasi yang kompleks hingga menyebarkannya di lingkungan yang aman dan patuh. Ini menekankan pentingnya akurasi, interpretabilitas, dan pertimbangan etis, memastikan bahwa LLM efektif dan bertanggung jawab dalam aplikasi hukum yang berisiko tinggi. Bab ini juga membahas strategi untuk memantau dan memelihara model yang disebarkan, memastikan model tetap patuh terhadap standar hukum dan peraturan yang terus berkembang. Melalui contoh praktis dan studi kasus, pembaca mendapatkan wawasan tentang pengembangan dan penyebaran LLM dalam bidang hukum dan kepatuhan menggunakan efisiensi sistem.

## 16.1. Pengantar LLM dalam Hukum dan Kepatuhan

Model bahasa besar (LLM) memiliki potensi untuk secara signifikan mentransformasi sektor hukum dan kepatuhan, di mana data tekstual dalam jumlah besar, pedoman peraturan, dan dokumentasi perlu diparsing, dianalisis, dan diinterpretasikan. Dalam aplikasi seperti analisis kontrak, verifikasi kepatuhan terhadap peraturan, dan penelitian hukum, LLM dapat merampingkan proses yang secara tradisional memerlukan pengawasan manual. Dengan menganalisis kontrak, LLM dapat mengidentifikasi klausul kunci, menandai potensi risiko, dan memverifikasi kepatuhan terhadap standar hukum. Misalnya, LLM hukum mungkin mendeteksi klausul non-standar dalam sebuah kontrak, membantu pengacara menilai implikasinya sebelum memfinalisasi perjanjian. Demikian pula, dalam kepatuhan, LLM dapat secara otomatis memindai teks peraturan untuk memastikan bahwa organisasi mematuhi hukum dan pedoman yang relevan. Kecepatan, keamanan memori, dan konkurensi menjadi sangat berharga untuk membangun aplikasi berperforma tinggi di bidang ini, di mana privasi data, akurasi, dan kepatuhan terhadap peraturan adalah yang utama. Penanganan informasi sensitif yang aman selaras dengan persyaratan ketat dalam konteks hukum, sementara efisiensinya memungkinkan analisis waktu nyata, yang krusial untuk aplikasi seperti peninjauan kontrak dan pemantauan kepatuhan.

![Gambar 1: Tantangan Utama LLM dalam Hukum dan Kepatuhan.](https://lmvr.rantai.dev//images/L6A504iArDGBvUq8WSeO-ZEtC0lTSdN1Qwdyt5Q7z-v1.png)
**Gambar 1:** Tantangan Utama LLM dalam Hukum dan Kepatuhan.

Aplikasi hukum dan kepatuhan memerlukan kehati-hatian khusus dalam menangani bahasa yang kompleks, mengingat terminologi hukum sangat padat, sangat spesifik, dan seringkali terbuka untuk berbagai interpretasi. LLM yang disebarkan dalam domain ini harus akurat dan andal, karena kesalahan dapat berakibat serius. Misalnya, dalam analisis kontrak, salah tafsir LLM terhadap suatu klausul dapat menyebabkan kelalaian yang merugikan secara finansial. Dalam kepatuhan, kesalahan dalam mengidentifikasi ketidakpatuhan terhadap peraturan dapat berujung pada sanksi hukum. Secara matematis, akurasi model dapat direpresentasikan oleh fungsi $\text{Acc}(M) = \frac{|C|}{|T|}$, di mana $M$ adalah model, $C$ adalah set output yang benar, dan $T$ adalah total set output. Akurasi tinggi sangat kritis dalam konteks hukum di mana interpretasi yang tepat berdampak langsung pada pengambilan keputusan. Selain itu, keandalan model—konsistensi dalam menghasilkan output yang benar—dapat direpresentasikan dengan mengukur standar deviasi akurasi model pada berbagai sampel uji, memastikan bahwa model berkinerja baik di berbagai dokumen hukum.

Interpretabilitas LLM dalam hukum dan kepatuhan juga sangat penting, karena profesional hukum dan petugas kepatuhan harus dapat memahami dasar dari keputusan model. Sifat buram dari model bahasa besar, yang sering disebut sebagai "kotak hitam" (*black box*), menimbulkan tantangan bagi adopsi di bidang di mana keputusan perlu dapat ditelusuri dan dijelaskan. Teknik eksplanabilitas seperti peta atensi atau nilai Shapley dapat membuat keputusan LLM lebih transparan, memungkinkan pengguna untuk melihat bagian dokumen mana yang memengaruhi output model. Dalam aplikasi hukum, interpretabilitas ini dapat berarti menonjolkan klausul atau frasa kontrak tertentu yang berkontribusi pada klasifikasi "berisiko", sehingga mendukung pengambilan keputusan yang lebih tepat. Keamanan tipe dan presisi menawarkan fondasi yang kuat untuk mengimplementasikan alat eksplanabilitas, karena pengembang dapat menghindari jebakan perilaku yang tidak terdefinisi atau kesalahan memori yang dapat mengganggu output model yang jelas dan andal.

LLM dalam hukum dan kepatuhan juga memperkenalkan pertimbangan etika dan peraturan. Dalam mengotomatisasi tugas-tugas seperti peninjauan kontrak atau pemindaian peraturan, LLM dapat memengaruhi proses pengambilan keputusan yang secara tradisional disediakan untuk profesional manusia, menimbulkan pertanyaan tentang akuntabilitas dan risiko ketergantungan berlebihan pada AI. Misalnya, mengotomatiskan peninjauan kontrak dengan LLM secara tidak sengaja dapat memperkuat bias yang ada dalam bahasa kontrak historis, yang berdampak secara tidak proporsional pada kelompok tertentu. Teknik mitigasi bias, seperti pembobotan ulang (*reweighting*) atau *adversarial debiasing*, sangat penting untuk mengatasi risiko ini. Secara matematis, bias dalam output model dapat dipantau dengan menghitung rata-rata divergensi antara hasil yang diprediksi untuk kelompok demografis yang berbeda, memastikan bahwa hasilnya adil. Metode ini dapat diimplementasikan dengan fokus pada efisiensi dan presisi, memungkinkan pengembang untuk memantau bias secara waktu nyata dan melakukan penyesuaian yang diperlukan untuk mempromosikan penggunaan AI yang adil dan etis dalam aplikasi hukum.

Untuk mengilustrasikan aplikasi praktis LLM dalam hukum dan kepatuhan, pertimbangkan alat peninjauan kontrak. Alat ini akan menyerap kontrak, mengidentifikasi klausul kunci, dan menilai kepatuhan terhadap standar hukum yang ditentukan. Kode Python di bawah ini mendemonstrasikan logika alat peninjau kontrak hukum yang didukung oleh LLM.

### Contoh Python: Alat Peninjau Kontrak Hukum (Simulasi API)

```python
import json
from typing import List, Dict
from abc import ABC, abstractmethod

# Simulasi Tokenizer dan Model untuk demonstrasi
class MockTokenizer:
    def encode(self, text: str) -> List[int]:
        # Simulasi mengubah teks menjadi daftar ID token
        return [len(word) for word in text.split()]

class MockLLMModel:
    def __init__(self):
        self.tokenizer = MockTokenizer()

    def forward(self, tokens: List[int]) -> Dict:
        # Simulasi inferensi model untuk ekstraksi klausul
        return {"status": "success", "found_clauses": len(tokens)}

# Status aplikasi untuk menyimpan model analisis kontrak
class AppState:
    def __init__(self, model: MockLLMModel):
        self.model = model

class ContractInput:
    def __init__(self, text: str):
        self.text = text

# Fungsi endpoint untuk menangani permintaan peninjauan kontrak
def review_contract_endpoint(input_data: ContractInput, state: AppState):
    try:
        # Tokenisasi teks kontrak
        tokens = state.model.tokenizer.encode(input_data.text)

        # Jalankan inferensi pada teks yang ditokenisasi
        output = state.model.forward(tokens)

        # Simulasi ekstraksi klausul dan penandaan untuk peninjauan hukum
        flagged_clauses = [
            "Klausul non-kompetisi mungkin membatasi opsi pekerjaan di masa depan.",
            "Klausul ganti rugi (indemnity) dapat meningkatkan risiko tanggung jawab hukum."
        ]

        return json.dumps({"flagged_clauses": flagged_clauses})
    except Exception as e:
        return json.dumps({"error": str(e)})

def main():
    # Inisialisasi model dan status aplikasi
    model = MockLLMModel()
    state = AppState(model)

    # Contoh input kontrak
    sample_input = ContractInput("This contract includes non-compete and indemnity clauses.")
    
    # Jalankan simulasi review
    result = review_contract_endpoint(sample_input, state)
    print("Hasil Peninjauan Kontrak:", result)

if __name__ == "__main__":
    main()
```

Kode di atas mendefinisikan pipeline API yang aman untuk meninjau kontrak hukum menggunakan LLM. Fungsi `review_contract_endpoint` melakukan tokenisasi teks kontrak, menjalankannya melalui model untuk mengidentifikasi klausul kunci, dan mensimulasikan penandaan klausul yang mungkin memerlukan peninjauan hukum lebih lanjut. Struktur ini memungkinkan analisis hukum berkinerja tinggi dengan mengidentifikasi potensi risiko dalam kontrak secara waktu nyata.

## 16.2. Membangun Pipeline Data Hukum dan Kepatuhan

Membangun pipeline data untuk aplikasi hukum dan kepatuhan melibatkan penanganan volume data teks yang sangat besar, termasuk dokumen terstruktur seperti peraturan dan kontrak, serta data tidak terstruktur dari hukum kasus, teks hukum, dan opini pengadilan. Berbeda dengan data teks standar, dokumen hukum seringkali panjang, kompleks, dan memerlukan interpretasi yang tepat, yang membuat prapemrosesan, normalisasi, dan anotasi menjadi sangat kritis. Untuk LLM yang berfokus pada hukum dan kepatuhan, pipeline data harus mengubah dokumen-dokumen ini ke dalam format standar yang dapat dianalisis model secara efektif, memastikan bahwa terminologi hukum dan konteks penting tetap terjaga.

![Gambar 2: Pipeline prapemrosesan data untuk LLM hukum dan kepatuhan.](https://lmvr.rantai.dev//images/L6A504iArDGBvUq8WSeO-rrsS2SeqauW8zfo6sVM2-v1.png)
**Gambar 2:** Pipeline prapemrosesan data untuk LLM hukum dan kepatuhan.

Langkah mendasar dalam menyiapkan data hukum adalah prapemrosesan, yang mencakup pembersihan dan normalisasi teks. Data hukum seringkali mengandung artefak format, seperti tajuk bagian, catatan kaki, dan tag metadata, yang dapat memperkenalkan noise dan mengurangi akurasi model jika dibiarkan. Normalisasi secara matematis dapat didefinisikan sebagai fungsi $N(x) \rightarrow x'$, di mana $x$ adalah teks mentah dan $x'$ adalah versi yang dinormalisasi dan distandarisasi. Menstandarisasi istilah di berbagai dokumen, seperti menyingkat "pasal" menjadi "§" atau mengubah sitasi hukum ke format yang konsisten, memastikan bahwa model mengenali elemen-elemen ini secara seragam.

Anotasi data dan pelabelan juga merupakan langkah penting. Dalam konteks hukum, anotasi sering kali melibatkan penandaan entitas kunci, seperti pihak-pihak dalam kontrak, kewajiban, klausul, dan istilah yang didefinisikan. Pelabelan ini membantu LLM memahami peran dan hubungan dalam sebuah dokumen. Pengenalan entitas dapat direpresentasikan sebagai fungsi $f(x) \rightarrow \{e_1, e_2, \ldots, e_n\}$, di mana $x$ adalah kutipan teks dan $e_i$ adalah entitas yang diidentifikasi.

### Contoh Python: Pipeline Prapemrosesan dan Anotasi Hukum

```python
import re
from typing import List, Dict

class LegalDocument:
    def __init__(self, title: str, text: str):
        self.title = title
        self.text = text

class Annotation:
    def __init__(self, term: str, category: str, position: int):
        self.term = term
        self.category = category
        self.position = position

    def to_dict(self):
        return {"term": self.term, "category": self.category, "position": self.position}

def normalize_text(text: str) -> str:
    # Menghapus karakter non-alfanumerik kecuali spasi dan tanda baca dasar
    text = re.sub(r'[^a-zA-Z0-9\s.,;]', '', text)
    return text.lower()

def annotate_text(text: str) -> List[Annotation]:
    annotations = []
    # Daftar istilah hukum yang akan dicari
    terms_to_watch = ["indemnity", "liability", "confidentiality", "termination"]
    
    for term in terms_to_watch:
        for match in re.finditer(rf"\b{term}\b", text, re.IGNORECASE):
            annotations.append(Annotation(
                term=match.group(),
                category="LegalTerm",
                position=match.start()
            ))
    return annotations

def process_document(doc: LegalDocument) -> Dict:
    normalized = normalize_text(doc.text)
    annotations = annotate_text(normalized)
    
    return {
        "title": doc.title,
        "text": normalized,
        "annotations": [a.to_dict() for a in annotations]
    }

def main():
    sample_doc = LegalDocument(
        "Service Agreement", 
        "This contract includes indemnity and confidentiality clauses for all parties."
    )
    
    processed = process_document(sample_doc)
    print("Dokumen Teranotasi:", processed)

if __name__ == "__main__":
    main()
```

Contoh di atas menunjukkan cara memproses dan memberikan anotasi pada kontrak hukum untuk menyiapkannya bagi pelatihan LLM. Tren saat ini dalam AI hukum menekankan pentingnya mengintegrasikan data dari berbagai yurisdiksi dan bahasa, serta penggunaan data sintetis untuk mengatasi kelangkaan data hukum yang berlabel karena batasan privasi.

## 16.3. Melatih LLM pada Data Hukum dan Kepatuhan

Melatih LLM untuk tugas hukum dan kepatuhan memerlukan pendekatan khusus agar model tidak hanya memahami bahasa umum tetapi juga menginterpretasikan terminologi hukum yang kompleks dan nuansa peraturan. Bahasa hukum padat dan seringkali ambigu; misalnya, kata "wajib" (*shall*) dan "dapat" (*may*) memiliki implikasi hukum yang sangat berbeda. Strategi yang efektif adalah menggunakan *transfer learning* dan *fine-tuning* pada dataset hukum target $D_t$ setelah dipra-latih pada dataset umum $D_g$.

![Gambar 3: Pertimbangan utama dalam melatih LLM.](https://lmvr.rantai.dev//images/L6A504iArDGBvUq8WSeO-YZVYyzV3rfIHPcxWJ7KJ-v1.png)
**Gambar 3:** Pertimbangan utama dalam melatih LLM.

Eksplanabilitas sangat kritis karena profesional hukum perlu menelusuri penalaran model. Teknik seperti visualisasi atensi dapat membantu memperjelas proses pengambilan keputusan model. Mitigasi bias juga merupakan pertimbangan esensial; model yang dilatih pada hukum kasus historis mungkin mewarisi bias gender atau rasial dari keputusan masa lalu. Strategi mitigasi dapat melibatkan penyesuaian bobot sampel $w_i$ untuk setiap instansi $x_i$ guna menyeimbangkan pengaruh kelompok yang kurang terwakili.

### Contoh Python: Pipeline Penyetelan Halus (Fine-Tuning) Model Hukum

```python
# Simulasi pipeline pelatihan menggunakan logika PyTorch
class LegalFineTuner:
    def __init__(self, model, learning_rate=1e-5):
        self.model = model
        self.lr = learning_rate

    def train_step(self, input_tensor, label_tensor):
        # 1. Forward pass
        prediction = self.model(input_tensor)
        
        # 2. Hitung loss (misal: Mean Squared Error)
        loss = ((prediction - label_tensor) ** 2).mean()
        
        # 3. Backward pass (simulasi update bobot)
        print(f"Melakukan backpropagation, Loss: {loss.item():.4f}")
        return loss

def main_training():
    print("Memulai pelatihan model analisis kontrak...")
    # Dalam skenario nyata, kita akan memuat model BERT atau DistilBERT
    # dan melatihnya pada tensor hasil prapemrosesan teks kontrak.
    pass
```

## 16.4. Inferensi dan Penyebaran LLM Hukum dan Kepatuhan

Inferensi dan penyebaran dalam aplikasi hukum sangat bergantung pada latensi, akurasi, dan skalabilitas. Aplikasi seperti pemantauan kepatuhan memerlukan inferensi latensi rendah agar dapat menghasilkan hasil tepat waktu. Selain itu, terdapat persyaratan ketat mengenai privasi data seperti GDPR yang mewajibkan enkripsi, anonimisasi, dan logging data yang kuat.

![Gambar 4: Menyeimbangkan Akurasi vs Latensi.](https://lmvr.rantai.dev//images/L6A504iArDGBvUq8WSeO-9y26ZCfAAJo29ftjYt95-v1.png)
**Gambar 4:** Menyeimbangkan Akurasi vs Latensi.

Terdapat trade-off antara kompleksitas model dan kecepatan inferensi. Latensi $T(f(x))$ berbanding lurus dengan kompleksitas model $C(f)$. Untuk mengoptimalkan ini, teknik seperti kuantisasi model (mengurangi presisi bobot) dapat digunakan tanpa mengorbankan akurasi secara signifikan.

### Contoh Python: Pipeline Inferensi Waktu Nyata untuk Dokumen Hukum

```python
class LegalInferenceEngine:
    def __init__(self, model_path: str):
        # Muat model yang sudah dilatih (simulasi)
        self.model = f"Model dimuat dari {model_path}"
        print(self.model)

    def analyze(self, text: str) -> List[str]:
        # Simulasi analisis teks hukum secara waktu nyata
        # Dalam praktik, ini akan melibatkan tokenisasi dan model.forward()
        results = []
        if "non-compete" in text.lower():
            results.append("Klausul: Non-kompetisi - Membatasi dan dapat membatasi opsi pekerjaan.")
        if "confidentiality" in text.lower():
            results.append("Klausul: Kerahasiaan - Diterapkan dengan penalti luas untuk pelanggaran.")
        return results

def main_inference():
    engine = LegalInferenceEngine("path/ke/model_hukum.bin")
    contract_text = "This agreement includes a strict non-compete clause."
    
    analysis = engine.analyze(contract_text)
    print("Hasil Analisis Real-time:", analysis)

if __name__ == "__main__":
    main_inference()
```

## 16.5. Pertimbangan Etika dan Regulasi

Penggunaan LLM di bidang hukum memunculkan tantangan seputar bias, keadilan, dan transparansi. Jika sebuah model dilatih pada data historis yang mencerminkan disparitas sosio-ekonomi, ia dapat menghasilkan rekomendasi yang memperkuat bias tersebut. Deteksi bias melibatkan evaluasi output model di berbagai segmen demografis menggunakan metrik seperti rasio dampak yang berbeda (*disparate impact ratio*).

![Gambar 5: Aspek Etika dan Regulasi.](https://lmvr.rantai.dev//images/L6A504iArDGBvUq8WSeO-Uzu5AxGnR685bdYnhooV-v1.png)
**Gambar 5:** Aspek Etika dan Regulasi.

Nilai Shapley ($\phi_i$) sering digunakan untuk menentukan kontribusi fitur $i$ terhadap prediksi keseluruhan $f(x)$, membantu profesional hukum memahami alasan di balik flag "risiko tinggi" pada suatu dokumen.

### Contoh Python: Framework Deteksi Bias Sederhana

```python
class LegalCase:
    def __init__(self, text: str, demographics: str, prediction_score: float):
        self.text = text
        self.demographics = demographics # misal: "group_a", "group_b"
        self.prediction = prediction_score

def detect_bias(cases: List[LegalCase]) -> Dict[str, float]:
    stats = {}
    for case in cases:
        group = case.demographics
        if group not in stats:
            stats[group] = {"total_score": 0.0, "count": 0}
        stats[group]["total_score"] += case.prediction
        stats[group]["count"] += 1
    
    report = {}
    for group, data in stats.items():
        report[group] = data["total_score"] / data["count"]
    return report

def main_ethics():
    dataset = [
        LegalCase("Case 1", "gender_male", 0.8),
        LegalCase("Case 2", "gender_female", 0.6),
        LegalCase("Case 3", "gender_male", 0.75),
        LegalCase("Case 4", "gender_female", 0.62)
    ]
    
    report = detect_bias(dataset)
    print("Laporan Deteksi Bias (Rata-rata Skor per Grup):", report)

if __name__ == "__main__":
    main_ethics()
```

## 16.6. Studi Kasus dan Arah Masa Depan

Studi kasus menunjukkan bahwa LLM dalam analisis kontrak dapat sangat merampingkan proses peninjauan dengan menandai area berisiko tinggi. Pemantauan kepatuhan otomatis juga memungkinkan organisasi untuk tetap selaras dengan peraturan seperti GDPR secara terus-menerus.

![Gambar 6: Beberapa area transformasi AI dalam hukum dan kepatuhan.](https://lmvr.rantai.dev//images/L6A504iArDGBvUq8WSeO-PHH7qZBh2VybjO2Qly0a-v1.png)
**Gambar 6:** Beberapa area transformasi AI dalam hukum dan kepatuhan.

Masa depan LLM di bidang hukum mencakup pengembangan model adaptif yang belajar dari interaksi pengguna dan penggunaan *federated learning* untuk melatih model pada dataset terdesentralisasi tanpa membagikan informasi sensitif secara langsung.

### Contoh Python (Logika Kasus): Penandaan Klausul Risiko Tinggi

```python
class RiskFlaggingTool:
    def flag_clauses(self, contract_text: str):
        # 1. Tokenisasi teks kontrak
        # 2. Jalankan inferensi model
        # 3. Tandai klausul risiko tinggi (simulasi)
        results = [
            "Klausul Ganti Rugi - Risiko kewajiban tinggi bagi klien.",
            "Klausul Terminasi - Ketentuan sepihak terdeteksi."
        ]
        return results
```

## 16.7. Kesimpulan

Bab 16 telah merangkum seluruh proses penerapan LLM di sektor hukum, mulai dari pipeline data hingga pertimbangan etika. Dengan memanfaatkan efisiensi sistem, profesional hukum dapat membangun solusi AI yang transparan, akurat, dan patuh terhadap regulasi yang ketat.

### 16.7.1. Pembelajaran Lebih Lanjut dengan GenAI

- Jelaskan peran LLM dalam meningkatkan efisiensi penelitian hukum.
- Apa tantangan utama penanganan privasi data dalam aplikasi kepatuhan?
- Bagaimana teknik eksplanabilitas (seperti Shapley values) membantu mendapatkan kepercayaan pengacara?
- Diskusikan risiko ketergantungan berlebihan pada AI dalam pengambilan keputusan hukum.
- Jelajahi regulasi GDPR dan HIPAA dalam konteks deployment LLM.
- Bagaimana *transfer learning* memungkinkan model umum memahami dokumen hukum?
- Diskusikan manfaat *federated learning* untuk menjaga kerahasiaan data klien.
- Analisis dampak model bias terhadap keadilan di pengadilan.
- Apa potensi penggunaan LLM untuk pembuatan draf dokumen hukum otomatis?
- Jelaskan pelajaran penting dari kegagalan sistem AI dalam kasus hukum nyata.

### 16.7.2. Latihan Praktis

#### Latihan Mandiri 16.1: Membangun Pipeline Data Hukum
Tujuan: Merancang sistem untuk menyerap dokumen PDF hukum, melakukan normalisasi teks, dan mengekstraksi entitas kunci menggunakan regex.

#### Latihan Mandiri 16.2: Melatih LLM Khusus Hukum
Tujuan: Melakukan penyetelan halus pada model bahasa dasar menggunakan dataset klausul kontrak berlabel.

#### Latihan Mandiri 16.3: Menyebarkan Model untuk Inferensi Waktu Nyata
Tujuan: Mengimplementasikan API sederhana yang menerima teks kontrak dan mengembalikan daftar klausul yang berisiko secara instan.

#### Latihan Mandiri 16.4: Audit Kepatuhan Etika
Tujuan: Mengembangkan modul untuk memantau disparitas dalam prediksi model antara kelompok demografis yang berbeda.

#### Latihan Mandiri 16.5: Replikasi Studi Kasus
Tujuan: Membuat prototipe alat riset hukum yang menggunakan kesamaan kosinus untuk menemukan kasus masa lalu yang relevan dengan kueri pengguna.