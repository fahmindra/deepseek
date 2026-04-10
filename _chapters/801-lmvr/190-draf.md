---
slug: lmvr-19
title: lmvr-19
---
# Bab 19: Jaringan Saraf Graf (GNN) dan LLM

> **"Kekuatan sejati dari AI terletak pada kemampuannya untuk menggabungkan berbagai jenis model dan data, seperti jaringan saraf graf dan model bahasa, untuk mencapai pemahaman yang lebih mendalam dan pengambilan keputusan yang lebih baik di berbagai domain yang kompleks." — Yann LeCun**

> *Bab 19 dari LMVR menggali integrasi Jaringan Saraf Graf (GNN) dengan Model Bahasa Besar (LLM) menggunakan Rust, menawarkan kerangka kerja yang kuat untuk memproses dan menalar data berstruktur graf. Bab ini mencakup seluruh siklus hidup pengembangan model GNN-LLM, mulai dari membangun pipeline data untuk data graf dan melatih GNN pada dataset skala besar hingga menyebarkan model-model ini di lingkungan yang skalabel dan waktu nyata. Bab ini menekankan pentingnya mengatasi tantangan seperti over-smoothing, skalabilitas, dan pertimbangan etis, memastikan bahwa model GNN-LLM tidak hanya kuat dan efisien tetapi juga adil dan transparan. Dengan memanfaatkan fitur kinerja dan keamanan Rust, bab ini membekali pembaca dengan pengetahuan untuk menciptakan model AI terintegrasi tingkat lanjut yang unggul dalam tugas-tugas yang memerlukan penalaran data terstruktur.*

## 19.1. Pengantar Jaringan Saraf Graf (GNN)

Jaringan Saraf Graf (GNN) mewakili kelas jaringan saraf transformatif yang dirancang khusus untuk memproses dan menganalisis data berstruktur graf. Graf lazim di berbagai domain, termasuk jejaring sosial, biologi molekuler, sistem rekomendasi, dan banyak lagi, di mana titik data (atau node) saling terhubung oleh hubungan (atau tepi/edge). Berbeda dengan jaringan saraf tradisional yang beroperasi pada data terstruktur seperti gambar atau teks, GNN secara inheren fleksibel, dirancang untuk menangani data yang tidak teratur dan non-Euclidean. GNN memanfaatkan struktur graf untuk mempelajari representasi node, tepi, atau seluruh graf, membuatnya sangat cocok untuk tugas-tugas seperti klasifikasi node, prediksi tautan, dan klasifikasi graf. Setiap node dalam graf menyimpan informasi unik, dan tepi mendefinisikan hubungan yang dapat meningkatkan akurasi prediksi dengan menangkap ketergantungan antar titik data. Struktur ini memungkinkan GNN untuk menangkap pola lokal dan global dalam graf, sehingga memberikan wawasan yang tidak dapat ditawarkan oleh jaringan saraf tradisional.

GNN unggul dalam aplikasi di berbagai bidang di mana hubungan dan struktur adalah kuncinya. Dalam pemodelan molekuler, GNN memprediksi sifat molekul dengan memperlakukan atom sebagai node dan ikatan sebagai tepi, membantu penemuan obat dan ilmu material. Dalam graf pengetahuan, GNN meningkatkan sistem rekomendasi dan mesin pencari dengan menganalisis entitas dan hubungan untuk menyimpulkan koneksi baru. Untuk jaringan informasi, seperti jaringan sitasi, GNN dapat mengklasifikasikan dokumen atau memprediksi tautan di antara mereka. Dalam ilmu saraf, GNN memodelkan interaksi saraf otak yang kompleks, membantu dalam memahami fungsi otak dan penyakit. Untuk data genomik, GNN menganalisis interaksi gen, mengungkapkan wawasan tentang regulasi gen dan mekanisme penyakit. Dalam jaringan komunikasi, GNN mengoptimalkan perutean data dengan memahami topologi jaringan. Dalam analisis perangkat lunak, GNN membantu dalam analisis kode, deteksi bug, dan identifikasi kerentanan dengan merepresentasikan struktur kode sebagai graf. Terakhir, di media sosial, GNN menganalisis koneksi dan interaksi pengguna untuk mendeteksi komunitas, tren, dan anomali, mendorong wawasan dalam dinamika sosial dan rekomendasi yang ditargetkan. Aplikasi-aplikasi ini menunjukkan keserbagunaan GNN dalam menganalisis sistem data yang saling terhubung secara kompleks.

![Gambar 1: Bidang aplikasi Jaringan Saraf Graf (GNN).](https://lmvr.rantai.dev//images/L6A504iArDGBvUq8WSeO-UUqKcglEwSzOqJUKW5rS-v1.jpeg)
**Gambar 1:** Bidang aplikasi Jaringan Saraf Graf (GNN).

Blok bangunan fundamental GNN adalah node, tepi, dan operasi konvolusi graf. Dalam graf $G = (V, E)$, di mana $V$ menunjukkan set node dan $E$ menunjukkan set tepi, setiap node $v \in V$ memiliki fitur terkait $x_v$, dan tepi $e \in E$ mewakili hubungan atau koneksi antar node. Inti dari GNN terletak pada kemampuannya untuk merambatkan informasi di seluruh node dan tepi ini. Operasi konvolusi graf mengagregasi dan mentransformasi fitur node tetangga secara iteratif, memungkinkan setiap node untuk belajar dari struktur dan atribut lingkungannya. Secara matematis, ini dapat direpresentasikan oleh aturan pembaruan dalam bentuk $h_v^{(k+1)} = f(h_v^{(k)}, \text{AGGREGATE}(\{h_u^{(k)} : u \in \mathcal{N}(v)\}))$, di mana $h_v^{(k)}$ menunjukkan representasi node $v$ pada lapisan ke-$k$, $\mathcal{N}(v)$ menunjukkan tetangga dari $v$, serta $f$ dan AGGREGATE adalah fungsi yang menggabungkan informasi dari node tetangga. Proses iteratif ini memungkinkan GNN untuk menangkap pola graf spesifik node dan global, membuatnya tangguh untuk berbagai tugas.

Pengembangan GNN muncul dari keterbatasan jaringan saraf tradisional ketika diterapkan pada data graf. Jaringan saraf konvolusional (CNN), misalnya, unggul dalam struktur seperti kisi seperti gambar tetapi kurang fleksibel untuk memproses data dengan koneksi arbitrer. GNN dirancang untuk menjembatani kesenjangan ini, menawarkan metode bagi jaringan saraf untuk beroperasi pada data berstruktur graf secara langsung. Seiring waktu, berbagai arsitektur telah muncul untuk menangani jenis tugas berbasis graf tertentu. Graph Convolutional Networks (GCN) menggunakan konvolusi berbasis spektral untuk melakukan pengiriman pesan antar node, mengagregasi informasi melalui lapisan konvolusi graf. Graph Attention Networks (GAT) memperkenalkan mekanisme atensi untuk menimbang pentingnya node tetangga, memungkinkan model untuk fokus pada tetangga yang paling relevan secara dinamis. GraphSAGE, arsitektur populer lainnya, menerapkan teknik pengambilan sampel untuk menangani graf besar secara efisien dengan hanya mengagregasi informasi dari subset tetangga. Arsitektur ini telah berkontribusi pada keserbagunaan GNN, memungkinkan mereka menangani struktur graf yang beragam dan menangkap berbagai tingkat ketergantungan node dan relevansi kontekstual.

Tren yang muncul dalam deep learning adalah integrasi GNN dengan model bahasa besar (LLM), yang meningkatkan kemampuan LLM untuk memproses dan menalar data relasional. Integrasi ini dapat sangat bermanfaat dalam aplikasi di mana struktur relasional itu penting, seperti graf pengetahuan, pengelompokan dokumen, dan analisis media sosial. Misalnya, ketika diterapkan pada graf pengetahuan, GNN dapat meningkatkan pemahaman LLM tentang hubungan antar entitas dengan mengodekan koneksi ini sebagai embedding graf, yang kemudian dapat diproses oleh LLM sebagai informasi kontekstual. Dengan memperluas kemampuan LLM untuk menangani data relasional terstruktur, GNN membuka jalan bagi aplikasi yang memerlukan pemahaman mendalam tentang entitas yang saling terkait. Kemampuan untuk memproses data terstruktur dan tidak terstruktur membuat GNN menjadi tambahan yang berharga untuk kotak peralatan dalam mengembangkan aplikasi LLM yang lebih bernuansa dan sadar konteks.

Berikut adalah implementasi dasar GNN untuk klasifikasi node menggunakan Python.

### Contoh Python: Implementasi GNN Sederhana (Menggunakan NumPy)

```python
import numpy as np

# Mendefinisikan struktur graf
class SimpleGraph:
    def __init__(self, nodes, edges):
        self.nodes = nodes  # Fitur node: dict {node_id: array_fitur}
        self.edges = edges  # List adjasensi: dict {node_id: [tetangga_id]}

# Inisialisasi lapisan GNN sederhana untuk pengiriman pesan (message passing)
def gnn_layer(graph, node_embeddings):
    new_embeddings = {}
    
    for node, neighbors in graph.edges.items():
        # Agregasi embedding tetangga
        if not neighbors:
            aggregated = np.zeros_like(node_embeddings[node])
        else:
            # Mengambil embedding tetangga dan menghitung rata-ratanya
            neighbor_feats = [node_embeddings[neighbor] for neighbor in neighbors]
            aggregated = np.mean(neighbor_feats, axis=0)
        
        # Perbarui embedding node dengan transformasi sederhana
        # Terapkan aktivasi non-linear tanh
        new_embeddings[node] = np.tanh(aggregated)
        
    return new_embeddings

def main():
    # Mendefinisikan graf sederhana
    # Setiap node memiliki 3 fitur
    nodes = {
        1: np.array([0.1, 0.2, 0.3], dtype=np.float32),
        2: np.array([0.4, 0.1, 0.2], dtype=np.float32),
        3: np.array([0.5, 0.7, 0.2], dtype=np.float32)
    }
    
    edges = {
        1: [2, 3],
        2: [1, 3],
        3: [1, 2]
    }

    graph = SimpleGraph(nodes, edges)

    # Inisialisasi embedding node dari fitur asli
    embeddings = graph.nodes.copy()

    # Terapkan lapisan GNN untuk pembaruan embedding node
    updated_embeddings = gnn_layer(graph, embeddings)
    
    print("Embedding node setelah diperbarui:")
    for node_id, feat in updated_embeddings.items():
        print(f"Node {node_id}: {feat}")

if __name__ == "__main__":
    main()
```

Studi kasus menunjukkan potensi transformatif GNN dalam meningkatkan kemampuan LLM dan model AI lainnya di berbagai domain. Misalnya, dalam penemuan obat, GNN telah berhasil diterapkan untuk memprediksi sifat molekul, memungkinkan peneliti menyaring senyawa potensial secara efisien. Dengan merepresentasikan molekul sebagai graf, di mana node sesuai dengan atom dan tepi mewakili ikatan, GNN dapat menangkap sifat relasional dalam molekul, menghasilkan prediksi yang lebih akurat daripada model tradisional. Demikian pula, dalam sistem rekomendasi, GNN membantu memodelkan hubungan pengguna-item dengan menganalisis perilaku dan koneksi pengguna dalam graf sosial, menghasilkan rekomendasi personal yang lebih baik.

Masa depan GNN siap untuk mendorong batas dalam representasi dan analisis data, dengan penelitian berkelanjutan yang berfokus pada penskalaan GNN untuk graf masif, meningkatkan interpretabilitas model, dan mengintegrasikan GNN dengan arsitektur saraf lainnya, termasuk LLM. Penskalaan GNN untuk menangani graf skala besar, seperti jejaring sosial dengan miliaran node, menghadirkan tantangan signifikan dalam manajemen memori dan efisiensi komputasi. Teknik seperti pengambilan sampel (sampling), pengelompokan (clustering), dan komputasi terdistribusi sedang dieksplorasi untuk mengatasi tantangan ini. Interpretabilitas adalah area penelitian aktif lainnya, di mana upaya dilakukan untuk menjelaskan bagaimana GNN menurunkan embedding dan membuat prediksi. Transparansi ini sangat penting untuk aplikasi di mana GNN digunakan dalam domain kritis, seperti layanan kesehatan dan keuangan. Terakhir, integrasi GNN dengan LLM dan arsitektur saraf lainnya membuka batasan baru dalam AI, memungkinkan model untuk memproses data relasional terstruktur bersama dengan teks tidak terstruktur, sehingga memberikan konteks yang lebih kaya dan wawasan yang lebih dalam.

Sebagai kesimpulan, Jaringan Saraf Graf (GNN) mewakili alat yang ampuh untuk menganalisis data berstruktur graf, dan aplikasinya berkembang pesat di berbagai domain. Dengan memanfaatkan efisiensi sistem dan keamanan memori, pengembang dapat mengimplementasikan GNN yang efisien dan skalabel, memungkinkan pemrosesan data graf yang kompleks dengan mudah. Bagian ini menyoroti peran teknologi dalam memfasilitasi pertumbuhan GNN, memberikan fondasi untuk kemajuan masa depan dalam penalaran relasional berbasis AI dan memperluas cakupan aplikasi LLM melalui wawasan berbasis graf. Seiring GNN terus berkembang, pengembangan sistem akan memainkan peran penting dalam mendukung kemajuan ini, mendorong inovasi di bidang padat data di mana konteks relasional dan struktur sangat penting.

---

## 19.2. Membangun Pipeline Data untuk GNN

Membuat pipeline data untuk Jaringan Saraf Graf (GNN) memerlukan penanganan struktur data khusus, termasuk node, tepi, dan fitur terkaitnya, yang semuanya menentukan kompleksitas data graf. Data yang digunakan dalam GNN berbeda secara signifikan dari data jaringan saraf lainnya, di mana setiap node (mewakili entitas seperti orang, molekul, atau lokasi) dapat membawa fitur unik, dan tepi mewakili hubungan yang menangkap ketergantungan atau interaksi antar node. Data berstruktur graf ini, direpresentasikan secara matematis sebagai $G = (V, E)$, di mana $V$ adalah set node dan $E$ adalah set tepi, membentuk tulang punggung GNN dan memerlukan pipeline data yang mampu mengelola data skala besar, yang seringkali jarang (sparse), secara efisien.

Pipeline data untuk membangun Jaringan Saraf Graf (GNN) dimulai dengan akuisisi data, di mana data dikumpulkan dari sumber yang relevan dengan node dan hubungan (tepi) dalam sistem target. Selanjutnya, prapemrosesan menstandarisasi dan membersihkan data, menangani nilai yang hilang dan menormalisasi fitur. Dalam pengelompokan (clustering), entitas yang serupa dikelompokkan untuk mengurangi noise dan mengidentifikasi pola, membantu definisi struktur graf. Konstruksi graf menyusul, di mana node mewakili entitas, dan tepi mewakili hubungan berdasarkan aturan domain spesifik (misalnya, metrik kesamaan). Dalam pembelajaran representasi graf, GNN mempelajari embedding untuk node dengan mengagregasi dan mentransformasi fitur dari tetangga, mengodekan struktur graf dan interaksi fitur ke dalam representasi setiap node. Akhirnya, dalam tugas prediksi hilir seperti regresi, GNN yang terlatih menggunakan embedding node untuk memprediksi nilai kontinu (misalnya, memprediksi sifat molekuler atau pengaruh sosial), memberikan wawasan dan prediksi berdasarkan struktur graf yang dipelajari. Pipeline ini memungkinkan penggunaan data relasional yang efisien dan bermakna dalam tugas pemodelan prediktif.

![Gambar 2: Ilustrasi prapemrosesan data untuk GNN.](https://lmvr.rantai.dev//images/L6A504iArDGBvUq8WSeO-xoIPWSLmRHAkitOz8ZCQ-v1.png)
**Gambar 2:** Ilustrasi prapemrosesan data untuk GNN.

Prapemrosesan data dan rekayasa fitur adalah langkah dasar dalam mengembangkan dataset siap GNN. Node dan tepi biasanya memiliki atribut terkait, seperti atribut pengguna di jejaring sosial (usia, minat) atau jenis ikatan dalam graf molekuler. Tahap prapemrosesan menstandarisasi atribut-atribut ini, menangani masalah seperti data yang hilang dan mengubah input mentah menjadi representasi node atau tepi yang bermakna. Secara matematis, misalkan $x_v$ menunjukkan vektor fitur dari node $v$ dan $e_{uv}$ vektor fitur dari tepi yang menghubungkan node $u$ dan $v$. Vektor-vektor ini perlu diubah menjadi format ternormalisasi yang meningkatkan kemampuan model untuk mempelajari pola yang bermakna.

Aspek kritis lainnya dari penyiapan data adalah augmentasi graf, yang dapat meningkatkan kinerja GNN dengan memperkenalkan variabilitas dan meningkatkan representasi fitur. Teknik augmentasi graf, seperti augmentasi fitur dan pengambilan sampel graf, sangat penting untuk meningkatkan generalisasi GNN, terutama ketika data pelatihan terbatas atau sangat spesifik. Augmentasi fitur dapat melibatkan pembuatan fitur node tambahan, seperti mengagregasi atribut node tetangga untuk membuat atribut baru bagi setiap node. Secara matematis, ini dapat direpresentasikan sebagai $x_v' = \text{AGGREGATE}(\{x_u : u \in \mathcal{N}(v)\})$, di mana $x_v'$ adalah fitur yang diaugmentasi untuk node $v$ dan $\mathcal{N}(v)$ menunjukkan set tetangga. Teknik pengambilan sampel seperti pengambilan sampel node atau pengambilan sampel subgraf membantu menangani graf besar dengan memilih subset representatif yang lebih kecil, mengurangi tuntutan komputasi dan meningkatkan skalabilitas.

Menangani data berstruktur graf menghadirkan beberapa tantangan, terutama dalam hal skalabilitas, kelangkaan, dan integritas data. Graf besar, seperti yang ada di jaringan sosial atau sitasi, dapat berisi jutaan node dan tepi, membuat penyimpanan dan pemrosesan data menjadi kompleks. Representasi jarang (sparse) sangat umum, karena tidak semua node saling terhubung, dan kepadatan koneksi sering bervariasi di seluruh graf. Teknik penyimpanan data jarang, seperti list adjasensi atau format *compressed sparse row* (CSR), sangat penting untuk mengelola penggunaan memori secara efisien. Dengan merepresentasikan graf dalam format jarang, pipeline dapat mencapai penghematan memori yang signifikan.

Berikut adalah kode Python yang mendemonstrasikan pipeline data dasar untuk menyerap, memproses, dan menambah fitur data graf menggunakan library `networkx`.

### Contoh Python: Pipeline Data Graf Sederhana (Menggunakan NetworkX)

```python
import networkx as nx
import numpy as np

class GraphDataPipeline:
    def __init__(self):
        # Membuat graf tak berarah menggunakan networkx
        self.graph = nx.Graph()
        self.node_features = {}

    def initialize_graph(self):
        # Menambahkan node dengan fitur awal (vektor numpy)
        # Node 0, 1, 2
        self.graph.add_node(0, features=np.array([0.2, 0.5], dtype=np.float32))
        self.graph.add_node(1, features=np.array([0.4, 0.1], dtype=np.float32))
        self.graph.add_node(2, features=np.array([0.3, 0.6], dtype=np.float32))

        # Menambahkan tepi dengan bobot
        self.graph.add_edge(0, 1, weight=1.0)
        self.graph.add_edge(1, 2, weight=0.8)
        
        # Menyimpan fitur awal ke dalam dict lokal untuk manipulasi mudah
        for node, data in self.graph.nodes(data=True):
            self.node_features[node] = data['features']

    def augment_features(self):
        """Augmentasi fitur dengan mengagregasi fitur tetangga (Mean Aggregation)"""
        new_features = {}
        for node in self.graph.nodes():
            neighbors = list(self.graph.neighbors(node))
            if not neighbors:
                new_features[node] = self.node_features[node]
                continue
            
            # Ambil fitur dari semua tetangga
            neighbor_feats = [self.node_features[n] for n in neighbors]
            # Hitung rata-rata fitur tetangga
            mean_neighbor_feat = np.mean(neighbor_feats, axis=0)
            
            # Di sini kita bisa menggabungkan fitur asli dengan fitur tetangga
            # Untuk demo, kita ganti saja dengan rata-rata tetangga
            new_features[node] = mean_neighbor_feat
            
        # Update state fitur
        self.node_features.update(new_features)

def main():
    pipeline = GraphDataPipeline()
    pipeline.initialize_graph()
    
    print("Fitur node awal:")
    for node, feat in pipeline.node_features.items():
        print(f"Node {node}: {feat}")
        
    pipeline.augment_features()
    
    print("\nFitur node setelah augmentasi (rata-rata tetangga):")
    for node, feat in pipeline.node_features.items():
        print(f"Node {node}: {feat}")

if __name__ == "__main__":
    main()
```

Aplikasi industri dari pipeline data GNN menonjolkan kebutuhan akan solusi manajemen data yang efisien dan skalabel. Misalnya, dalam sistem rekomendasi, perusahaan menggunakan GNN untuk memodelkan interaksi pengguna-item dengan menganalisis perilaku dan koneksi dalam graf sosial. Di sini, pipeline data harus menangani data berdimensi tinggi dan jarang, seperti atribut pengguna dan riwayat pembelian, yang memerlukan teknik pemrosesan dan penyimpanan data yang efisien untuk memenuhi tuntutan rekomendasi waktu nyata. Aplikasi lain dalam penemuan obat melibatkan prediksi interaksi molekuler berdasarkan koneksi atom. GNN memproses graf molekuler ini untuk menyimpulkan sifat obat potensial, mempercepat proses penyaringan senyawa baru.

Masa depan pipeline data GNN melibatkan integrasi yang ditingkatkan dengan sistem pemrosesan terdistribusi dan waktu nyata. Seiring dataset tumbuh dalam ukuran dan kompleksitas, pipeline data harus beradaptasi untuk menangani volume tersebut, seringkali memerlukan teknik pemrosesan graf terdistribusi. Selain itu, pemrosesan waktu nyata semakin relevan, terutama dalam aplikasi seperti analisis transaksi keuangan atau pemantauan media sosial, di mana graf berkembang secara terus-menerus. Pemrosesan graf waktu nyata memerlukan pipeline data yang sangat responsif, dan kemampuan latensi rendah serta multithreading yang efisien sangat menguntungkan di lingkungan ini.

Sebagai kesimpulan, membangun pipeline data untuk GNN memerlukan pemahaman mendalam tentang struktur data graf dan tantangan mengelola dataset yang jarang dan berdimensi tinggi. Efisiensi sistem dan dukungan konkurensi memberikan fondasi yang kuat untuk membangun pipeline yang mampu menskalakan untuk memenuhi tuntutan aplikasi berbasis graf modern. Dengan memfasilitasi prapemrosesan data, rekayasa fitur, dan augmentasi yang efisien, pipeline ini memberdayakan GNN untuk beroperasi secara optimal, membuka jalan baru untuk menerapkan GNN di berbagai bidang, dari sistem rekomendasi hingga analisis molekuler.

---

## 19.3. Melatih GNN pada Data Graf Menggunakan Rust

Melatih Jaringan Saraf Graf (GNN) pada data graf memerlukan teknik khusus dan infrastruktur yang disesuaikan untuk menangani sifat unik struktur graf. Berbeda dengan jaringan saraf konvensional, di mana data direpresentasikan dalam ruang Euclidean seperti kisi atau urutan, GNN memproses data graf yang non-Euclidean dan jarang, mencakup node (entitas), tepi (hubungan), dan seringkali vektor fitur berdimensi tinggi. Struktur ini memungkinkan GNN untuk melakukan tugas-tugas seperti klasifikasi node, prediksi tautan, dan klasifikasi graf. Performa tinggi dan kontrol tingkat sistem sangat menguntungkan untuk melatih GNN, terutama saat bekerja dengan dataset graf skala besar, karena memungkinkan traversal graf, agregasi lingkungan, dan manipulasi data yang efisien.

Salah satu operasi inti dalam pelatihan GNN adalah konvolusi graf, di mana setiap node mengagregasi informasi dari tetangganya untuk mempelajari representasi yang menangkap struktur graf lokal dan global. Secara matematis, misalkan $h_v^{(k+1)}$ mewakili embedding fitur dari node $v$ pada lapisan $k+1$, yang dihitung dengan mengagregasi embedding dari tetangganya $\mathcal{N}(v)$. Aturan pembaruan dapat direpresentasikan sebagai:

$$ h_v^{(k+1)} = \sigma\left( W^{(k)} \cdot \text{AGGREGATE}(\{h_u^{(k)} : u \in \mathcal{N}(v)\}) \right) $$

di mana $W^{(k)}$ adalah matriks bobot yang dapat dipelajari untuk lapisan $k$, AGGREGATE adalah fungsi yang menggabungkan embedding node tetangga (misalnya, rata-rata, jumlah, atau maksimum), dan $\sigma$ adalah fungsi aktivasi non-linier. Melalui beberapa lapisan konvolusi, GNN mempelajari embedding yang mewakili tidak hanya node individu tetapi juga hubungan mereka dan struktur graf.

Melatih GNN pada graf besar menghadirkan beberapa tantangan unik. *Over-smoothing*, misalnya, adalah masalah umum di mana, seiring bertambahnya lapisan, embedding node menjadi tidak dapat dibedakan, yang menyebabkan hilangnya spesifisitas. Teknik seperti koneksi residual, yang memperkenalkan kembali fitur node asli setelah setiap agregasi, dapat membantu memitigasi *over-smoothing*. Tantangan lainnya adalah skalabilitas, karena pelatihan batch penuh (*full-batch training*) pada graf besar sangat mahal secara komputasi dan intensif memori. Pelatihan mini-batch, di mana hanya subset node dan lingkungannya yang diproses di setiap batch, memberikan solusi yang skalabel. Pendekatan ini sering melibatkan teknik pengambilan sampel, seperti pengambilan sampel *random walk* atau pengambilan sampel subgraf, yang mengurangi beban komputasi.

Berikut adalah contoh Python menggunakan PyTorch untuk mendemonstrasikan pipeline pelatihan GNN sederhana, berfokus pada agregasi lingkungan dan operasi konvolusi graf.

### Contoh Python: Pelatihan GNN (Menggunakan PyTorch)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# Mendefinisikan model GNN sederhana untuk operasi konvolusi graf dasar
class SimpleGNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleGNN, self).__init__()
        # Matriks bobot yang dapat dipelajari
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, adj_matrix, features):
        """
        adj_matrix: Matriks adjasensi normalisasi (N, N)
        features: Matriks fitur node (N, input_dim)
        """
        # Langkah 1: Agregasi Lingkungan melalui perkalian matriks adjasensi
        # H_agg = A * X
        support = torch.mm(features, self.weight)
        
        # Langkah 2: Transformasi dan Aktivasi
        # H_next = sigma(A * X * W)
        output = torch.mm(adj_matrix, support)
        
        return torch.tanh(output)

def main():
    # Inisialisasi graf kecil (3 node, 2 fitur per node)
    # Node 0 terhubung ke 1 dan 2
    adj = torch.FloatTensor([
        [1, 1, 1],
        [1, 1, 0],
        [1, 0, 1]
    ])
    # Normalisasi sederhana: membagi baris dengan jumlah elemen di baris tersebut
    row_sum = adj.sum(1, keepdim=True)
    adj_norm = adj / row_sum

    features = torch.FloatTensor([
        [0.2, 0.5],
        [0.4, 0.1],
        [0.3, 0.6]
    ])

    # Membuat model dan menjalankan forward pass
    model = SimpleGNN(input_dim=2, output_dim=2)
    updated_features = model(adj_norm, features)
    
    print("Fitur node setelah diperbarui oleh GNN:")
    print(updated_features)

if __name__ == "__main__":
    main()
```

Pelatihan GNN juga menuntut interpretabilitas dan eksplanabilitas model, terutama di domain seperti layanan kesehatan dan keuangan. Memahami pentingnya node atau tepi tertentu dalam prediksi model sangat penting untuk membuat keputusan yang terinformasi dan akuntabel. Teknik seperti mekanisme atensi, di mana bobot diberikan kepada tetangga berdasarkan kepentingan relatifnya, meningkatkan interpretabilitas. Dalam GNN berbasis atensi, fungsi AGGREGATE dapat melibatkan jumlah tertimbang, di mana bobot dipelajari dan disesuaikan berdasarkan kepentingan fitur setiap tetangga.

Studi kasus mendemonstrasikan tantangan praktis dan keuntungan melatih GNN dalam skenario dunia nyata. Misalnya, dalam deteksi penipuan, jaringan keuangan dimodelkan sebagai graf, di mana node mewakili pengguna dan transaksi, dan tepi menunjukkan hubungan seperti transfer akun. Dengan melatih GNN pada struktur ini, perusahaan dapat mendeteksi pola yang tidak biasa dan menandai akun yang berpotensi curang. Untuk menangani volume data transaksional yang tinggi, pengembang menggunakan pelatihan mini-batch dikombinasikan dengan pengambilan sampel, memungkinkan pelatihan GNN skalabel yang mengidentifikasi pola kunci tanpa mengorbankan efisiensi komputasi.

Contoh lainnya adalah dalam penemuan obat, di mana molekul direpresentasikan sebagai graf dengan atom sebagai node dan ikatan kimia sebagai tepi. Melatih GNN untuk memprediksi sifat molekuler melibatkan graf besar yang jarang. Kontrol sistem atas manajemen memori sangat kritis di sini, karena memungkinkan pengembang untuk meminimalkan penggunaan sumber daya dengan mengimplementasikan teknik pengambilan sampel yang efisien dan perkalian matriks jarang.

Sebagai kesimpulan, melatih GNN pada data graf menggunakan infrastruktur sistem yang efisien memberikan pendekatan yang kuat untuk menangani tantangan unik yang ditimbulkan oleh struktur graf. Dari agregasi data dan operasi konvolusi graf hingga pengambilan sampel dan interpretabilitas, teknologi sistem memungkinkan pengembang untuk menciptakan pipeline GNN skalabel yang sangat cocok untuk dataset besar, jarang, dan kompleks. Bagian ini menggarisbawahi peran efisiensi dalam memajukan metodologi pelatihan GNN, memungkinkan pembangunan model GNN yang berkinerja tinggi dan dapat diinterpretasikan untuk memenuhi tuntutan aplikasi modern di berbagai bidang seperti jejaring sosial, layanan kesehatan, dan keuangan.

---

## 19.4. Mengintegrasikan GNN dengan LLM Menggunakan Rust

Integrasi Jaringan Saraf Graf (GNN) dengan Model Bahasa Besar (LLM) menawarkan pendekatan yang kuat untuk meningkatkan kemampuan sistem pembelajaran mesin dalam menangani teks tidak terstruktur dan data relasional terstruktur. Sementara LLM unggul dalam tugas pemrosesan bahasa—seperti menghasilkan teks yang koheren dan memahami sintaksis yang kompleks—GNN secara unik cocok untuk menangkap hubungan dan ketergantungan dalam data berstruktur graf, seperti jejaring sosial, graf pengetahuan, dan struktur molekuler. Dengan menggabungkan kekuatan kedua model tersebut, pengembang dapat menciptakan arsitektur hibrida yang menangani tugas-tugas kompleks yang memerlukan pemahaman teks mendalam dan penalaran relasional. Misalnya, tugas-tugas seperti penyelesaian graf pengetahuan, penautan entitas, dan sistem rekomendasi semuanya mendapat manfaat dari kekuatan hibrida pemahaman bahasa alami LLM dan pemrosesan data terstruktur GNN.

Mengintegrasikan Jaringan Saraf Graf (GNN) dan Model Bahasa Besar (LLM) membuka jalan baru untuk meningkatkan tugas berbasis graf dengan penalaran linguistik dan sebaliknya. Dalam skenario di mana LLM memperkuat algoritma graf, model bahasa dapat memberikan pengetahuan kontekstual atau menginterpretasikan hubungan kompleks yang membantu algoritma GNN dalam memahami semantik, seperti menyimpulkan label node atau tepi. Dalam prediksi tugas graf, LLM dapat membantu dalam mengantisipasi node atau tepi yang memenuhi peran tertentu, seperti mengidentifikasi node berpengaruh dalam jejaring sosial atau situs molekuler utama dalam graf biomedis. Untuk membangun graf, LLM dapat menghasilkan struktur graf dengan memparsing data teks, seperti membuat graf pengetahuan dari dokumen. Dalam pipeline yang lebih luas, LLM menambahkan lapisan penalaran ke struktur graf, mendukung rantai inferensi kompleks di atas graf, seperti "Graphs of Thoughts" atau "Chain of Thoughts" yang memungkinkan penalaran iteratif atas node graf selaras dengan jalur naratif. Prompting "Tree of Thoughts" memperluas konsep ini, di mana LLM menyarankan beberapa jalur atau hipotesis, menciptakan struktur percabangan yang mengeksplorasi berbagai kemungkinan penalaran. Terakhir, dalam sistem multi-agen, LLM dapat mensimulasikan interaksi kolaboratif atau kompetitif pada struktur graf, di mana setiap agen mewakili node dengan tujuan yang berbeda, menghasilkan analisis multi-perspektif tingkat lanjut untuk penyelesaian masalah atau tugas simulasi. Skenario-skenario ini mengilustrasikan potensi menggabungkan kekuatan struktur dan hubungan GNN dengan kemampuan penalaran dan bahasa LLM yang luas.

![Gambar 3: Kasus penggunaan umum integrasi LLM dan GNN.](https://lmvr.rantai.dev//images/L6A504iArDGBvUq8WSeO-T7fPKFPP7uSvrlEqCN7x-v1.webp)
**Gambar 3:** Kasus penggunaan umum integrasi LLM dan GNN.

Satu pendekatan untuk mengintegrasikan GNN dengan LLM melibatkan arsitektur seperti Graph-BERT dan Graph2Seq, yang mengadaptasi model bahasa untuk menyertakan struktur graf. Dalam arsitektur Graph-BERT, lapisan GNN memproses data graf menjadi embedding bermakna yang menangkap informasi struktural dan relasional dari node dan tepi. Embedding ini kemudian diumpankan sebagai token input ke model BERT, yang memprosesnya seolah-olah itu adalah token bahasa. Secara formal, untuk node $v$ tertentu dengan tetangga $\mathcal{N}(v)$, kita menghitung embedding berbasis GNN $h_v = f(\{x_u : u \in \mathcal{N}(v)\})$, di mana $f$ mewakili fungsi agregasi GNN. Embedding node $h_v$ ini kemudian digunakan sebagai token input dalam model bahasa, memungkinkan LLM untuk memproses tidak hanya teks tetapi juga informasi struktural dari graf.

Graph2Seq adalah arsitektur lain yang memanfaatkan GNN untuk tugas encoder-decoder, membuatnya cocok untuk aplikasi seperti penalaran graf pengetahuan dan tanya jawab multi-hop. Dalam model ini, GNN pertama-tama mengodekan node graf ke dalam representasi laten. Representasi ini kemudian diumpankan secara sekuensial ke decoder LLM, yang menghasilkan deskripsi bahasa alami atau jawaban berdasarkan graf yang dikodekan. Pipeline ini dapat direpresentasikan secara matematis oleh fungsi dua langkah $y = g(f(G))$, di mana $f(G)$ mewakili fungsi pengodean GNN pada graf $G$, dan $g$ menunjukkan fungsi decoding LLM. Pipeline ini, meskipun kuat, memerlukan manajemen memori yang cermat untuk menangani graf besar dan token bahasa berdimensi tinggi.

Mengintegrasikan GNN dengan LLM menghadirkan beberapa tantangan, terutama dalam menjaga koherensi antara struktur graf dan pemrosesan teks sekuensial. Struktur data graf tidak secara alami selaras dengan sifat sekuensial bahasa, mengharuskan model hibrida untuk menjaga konektivitas graf dan informasi relasional saat embedding diteruskan ke LLM. Skalabilitas adalah masalah kritis lainnya, terutama untuk graf besar dengan jutaan node dan tepi. Teknik pengambilan sampel graf, seperti pengambilan sampel lingkungan atau pengelompokan hierarkis, sangat penting untuk mengurangi beban komputasi. Teknik-teknik ini memungkinkan model untuk fokus pada subgraf atau wilayah graf yang paling relevan dengan tugas tertentu, sehingga mengurangi kompleksitas sambil tetap mempertahankan fidelitas struktural.

Berikut adalah contoh Python yang mendemonstrasikan pipeline untuk menghasilkan representasi node (GNN) dan mengintegrasikannya dengan fitur teks (simulasi LLM).

### Contoh Python: Integrasi GNN dan LLM Sederhana (PyTorch)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. Komponen GNN (Graph Encoder)
class GNNEncoder(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, adj, x):
        # Agregasi sederhana: A * X
        h = torch.mm(adj, x)
        return torch.relu(self.linear(h))

# 2. Komponen LLM (Simulasi dengan Linear Layer untuk pemrosesan teks)
class TextProcessor(nn.Module):
    def __init__(self, text_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(text_dim, out_dim)

    def forward(self, text_embeddings):
        # Simulasi pemrosesan output LLM
        return torch.relu(self.linear(text_embeddings))

# 3. Model Hibrida GNN-LLM
class HybridGNNLLM(nn.Module):
    def __init__(self, gnn_dim, text_dim, hidden_dim):
        super().__init__()
        self.gnn_encoder = GNNEncoder(gnn_dim, hidden_dim)
        self.llm_sim = TextProcessor(text_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim * 2, 1)

    def forward(self, adj, node_feats, text_feats):
        # Ambil fitur dari struktur graf
        graph_context = self.gnn_encoder(adj, node_feats)
        # Ambil fitur dari pemrosesan teks
        text_context = self.llm_sim(text_feats)
        
        # Gabungkan konteks struktur dan bahasa
        combined = torch.cat([graph_context, text_context], dim=1)
        return torch.sigmoid(self.classifier(combined))

def main():
    # Inisialisasi data simulasi
    num_nodes = 4
    gnn_feat_dim = 16
    text_feat_dim = 128
    hidden_dim = 32

    adj = torch.eye(num_nodes) # Matriks identitas sebagai adjasensi dasar
    node_features = torch.randn(num_nodes, gnn_feat_dim)
    text_features = torch.randn(num_nodes, text_feat_dim)

    model = HybridGNNLLM(gnn_feat_dim, text_feat_dim, hidden_dim)
    output = model(adj, node_features, text_features)
    
    print("Output Prediksi Hibrida GNN-LLM:")
    print(output)

if __name__ == "__main__":
    main()
```

Mengintegrasikan GNN dengan LLM memiliki implikasi signifikan di berbagai domain. Dalam sistem rekomendasi, GNN dapat merepresentasikan hubungan pengguna-item, yang kemudian diumpankan ke LLM yang menghasilkan rekomendasi yang dipersonalisasi. Dalam penyelesaian graf pengetahuan, integrasi ini memungkinkan penalaran atas data berstruktur graf, di mana entitas dan hubungan dapat dikaitkan, disimpulkan, atau dilengkapi menggunakan kemampuan pemahaman bahasa yang kaya dari LLM. Dalam pemahaman bahasa alami, model hibrida meningkatkan pemahaman kontekstual dengan menyertakan data relasional, meningkatkan akurasi dalam tugas-tugas seperti tanya jawab multi-hop dan agen percakapan.

Masa depan integrasi GNN dengan LLM siap mendorong kemajuan signifikan dalam kecerdasan buatan, terutama dalam aplikasi yang memerlukan penalaran relasional yang mendalam. Arsitektur baru mulai muncul yang memadukan mekanisme atensi berbasis graf secara langsung di dalam LLM, memungkinkan pelatihan ujung-ke-ujung tanpa perlu penyerahan embedding secara eksplisit. Kebangkitan transformer dan GNN berbasis atensi juga menyarankan konvergensi di mana keuntungan dari kedua model dapat disatukan, memberikan generalisasi yang lebih kuat di berbagai jenis data terstruktur dan tidak terstruktur. Model hibrida ini diharapkan memainkan peran sentral dalam bidang yang berkembang seperti Semantic Web, di mana data terstruktur dan tidak terstruktur hidup berdampingan.

Sebagai kesimpulan, mengintegrasikan GNN dengan LLM mewakili perbatasan yang menarik dalam pembelajaran mesin, memungkinkan penalaran kompleks dan pemahaman relasional di berbagai aplikasi. Kemampuan sistem dalam penanganan memori yang efisien, konkurensi, dan manajemen data memberikan lingkungan yang ideal untuk membangun arsitektur hibrida ini, menawarkan kinerja dan skalabilitas untuk pemrosesan data graf dan teks skala besar. Dengan menggabungkan GNN dan LLM, pengembang dapat memanfaatkan kekuatan kedua model untuk membangun sistem yang tidak hanya memahami bahasa tetapi juga hubungan dan struktur kompleks yang tertanam dalam data dunia nyata.

---

## 19.5. Inferensi dan Penyebaran Model GNN-LLM Menggunakan Rust

Menyebarkan model GNN-LLM di lingkungan produksi memerlukan keseimbangan yang cermat antara kompleksitas model, kecepatan inferensi, dan skalabilitas, terutama untuk aplikasi di mana pemrosesan waktu nyata sangat penting. Hibrida GNN dan LLM semakin banyak diterapkan di domain seperti sistem rekomendasi, analisis jejaring sosial, dan deteksi penipuan. Proses inferensi untuk model GNN-LLM melibatkan pembuatan prediksi berdasarkan data berstruktur graf dan teks, menjadikannya tugas intensif komputasi yang harus dioptimalkan untuk throughput tinggi dan latensi rendah. Kemampuan berorientasi performa dalam manajemen memori dan pemrosesan konkuren menjadikannya pilihan yang sangat baik untuk mengimplementasikan dan menyebarkan model-model ini dalam skala besar.

![Gambar 4: Kompleksitas penyebaran model GNN-LLM.](https://lmvr.rantai.dev//images/L6A504iArDGBvUq8WSeO-33mPvOUsNhGVLlj8CHj6-v1.png)
**Gambar 4:** Kompleksitas penyebaran model GNN-LLM.

Dalam model GNN-LLM, inferensi biasanya melibatkan dua tahap utama: komponen GNN memproses informasi berbasis graf untuk menghasilkan embedding yang menangkap hubungan dalam dataset, sementara LLM menggunakan embedding ini bersama data tekstual untuk melakukan tugas-tugas seperti penautan entitas, klasifikasi dokumen, atau pembuatan rekomendasi. Pendekatan inferensi ganda ini dapat direpresentasikan secara formal sebagai $y = f(g(X_G), X_T)$, di mana $X_G$ adalah input data graf, $X_T$ adalah input data teks, $g$ adalah fungsi embedding GNN, dan $f$ mewakili lapisan tugas LLM. Pipeline ini tidak hanya memerlukan penanganan data yang efisien tetapi juga menuntut integrasi yang ketat antara komponen GNN dan LLM untuk meminimalkan latensi.

Salah satu tantangan utama dalam menyebarkan model GNN-LLM adalah menangani dataset graf besar dalam aplikasi waktu nyata, karena data graf bisa padat dan berkembang secara dinamis. Misalnya, dalam jejaring sosial, node (pengguna) dan tepi (hubungan) baru terus ditambahkan, yang dapat secara signifikan memengaruhi akurasi rekomendasi. Strategi seperti pemrosesan batch, yang memproses kelompok node secara bersamaan, dan pengambilan sampel graf waktu nyata membantu mengelola kompleksitas ini. Kemampuan multithreading dan asinkron memungkinkan pemrosesan batch yang sangat efisien, memungkinkan model untuk menghasilkan rekomendasi dengan segera, bahkan saat data graf berkembang.

Strategi penyebaran untuk model GNN-LLM juga harus memperhitungkan beban komputasi dan infrastruktur yang diperlukan untuk mendukung inferensi yang skalabel. Tergantung pada kebutuhan aplikasi, strategi seperti inferensi terdistribusi, penyebaran edge, dan kuantisasi model dapat digunakan. Dalam inferensi terdistribusi, beban kerja dibagi di beberapa node. Penyebaran edge membawa model lebih dekat ke sumber data, mengurangi latensi jaringan. Teknik kuantisasi model, yang mengurangi presisi bobot dan aktivasi, dapat membantu mengurangi penggunaan memori tanpa kehilangan akurasi yang signifikan.

Berikut adalah implementasi Python untuk pipeline inferensi hibrida GNN-LLM sederhana menggunakan PyTorch.

### Contoh Python: Pipeline Inferensi Hibrida (GNN + LLM)

```python
import torch
import torch.nn as nn

class InferenceGNNLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(dim, dim))

    def forward(self, adj, x):
        # x shape: (num_nodes, dim)
        # adj shape: (num_nodes, num_nodes)
        h = torch.mm(adj, x)
        return torch.tanh(torch.mm(h, self.weight))

class LLMInferenceHead(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.output_layer = nn.Linear(dim, 1)

    def forward(self, gnn_embeddings, text_embedding):
        # Menggabungkan semua embedding node dengan embedding teks global
        # Sederhananya, kita rata-ratakan embedding node GNN
        fused_gnn = torch.mean(gnn_embeddings, dim=0, keepdim=True)
        # Gabungkan (Concatenate)
        combined = fused_gnn + text_embedding
        return self.output_layer(combined)

def main_inference():
    dim = 64
    num_nodes = 5
    
    # Data graf simulasi
    adj = torch.eye(num_nodes)
    node_features = torch.randn(num_nodes, dim)
    # Fitur teks simulasi (hasil dari LLM encoder)
    text_embedding = torch.randn(1, dim)

    # Inisialisasi komponen model
    gnn_layer = InferenceGNNLayer(dim)
    llm_head = LLMInferenceHead(dim)

    # Langkah 1: Inferensi GNN
    with torch.no_grad():
        gnn_out = gnn_layer(adj, node_features)
        
        # Langkah 2: Inferensi LLM Head (Penggabungan)
        result = llm_head(gnn_out, text_embedding)

    print(f"Hasil Inferensi Hibrida: {result.item():.4f}")

if __name__ == "__main__":
    main_inference()
```

Saat menyebarkan model GNN-LLM, sangat penting untuk membangun sistem untuk memantau dan mempertahankan performa dari waktu ke waktu. *Model drift*, di mana performa model menurun karena perubahan distribusi data, adalah masalah umum. Sistem pemantauan dapat diatur untuk melacak metrik kinerja utama, seperti latensi, akurasi, dan penggunaan sumber daya. Ketika penyimpangan signifikan terdeteksi, model mungkin perlu dilatih ulang atau disetel halus. Penanganan data konkuren yang presisi sangat cocok untuk mengimplementasikan sistem pemantauan ini, memastikan bahwa kinerja model tetap stabil dan efisien bahkan dalam aplikasi lalu lintas tinggi.

Studi kasus menggarisbawahi nilai inferensi waktu nyata dalam penyebaran GNN-LLM. Misalnya, dalam analisis jejaring sosial, model GNN-LLM dapat melacak interaksi pengguna secara waktu nyata untuk merekomendasikan konten yang relevan atau mengidentifikasi pengguna berpengaruh. Sistem rekomendasi dalam jejaring sosial skala besar mungkin menangani ribuan permintaan per detik, memerlukan model yang dapat memproses data relasional secara efisien. Contoh lain adalah dalam deteksi penipuan, di mana inferensi waktu nyata sangat penting untuk mengidentifikasi transaksi mencurigakan saat terjadi.

Sebagai kesimpulan, menyebarkan model GNN-LLM di lingkungan sistem yang efisien menawarkan solusi komprehensif untuk menangani kompleksitas tugas berbasis graf dan bahasa dalam aplikasi dunia nyata. Keamanan memori, konkurensi, dan kontrol data yang tepat memungkinkan pengembang untuk mengimplementasikan pipeline inferensi efisien yang memberikan kecepatan dan akurasi pada skala besar. Dengan menggabungkan kekuatan GNN untuk data terstruktur dan LLM untuk teks tidak terstruktur, model ini membuka kemungkinan baru dalam berbagai sistem cerdas.

---

## 19.6. Pertimbangan Etis dan Praktis dalam Penyebaran GNN-LLM

Penyebaran model GNN-LLM membawa peluang sekaligus tantangan, terutama bila diterapkan pada domain sensitif seperti analisis jejaring sosial, sistem rekomendasi, deteksi penipuan, dan layanan kesehatan. Model hibrida ini unggul dalam menangani hubungan data yang kompleks dan bahasa alami, menjadikannya sangat efektif untuk tugas-tugas yang melibatkan data terstruktur dan tidak terstruktur. Namun, mereka juga memperkenalkan pertimbangan etis dan praktis, terutama dalam hal keadilan, transparansi, dan akuntabilitas. Karena ketergantungannya pada struktur graf dan data tekstual, model GNN-LLM rentan terhadap berbagai bias yang, jika tidak diatasi, dapat menyebabkan hasil yang tidak adil atau bahkan berbahaya. Bagian ini menggali potensi bias yang inheren pada model GNN-LLM, memeriksa strategi untuk memitigasi bias ini, dan mengeksplorasi praktik terbaik untuk memastikan bahwa model-model ini mematuhi pedoman etika dan standar peraturan.

![Gambar 5: Cakupan Penyebaran Model GNN-LLM.](https://lmvr.rantai.dev//images/L6A504iArDGBvUq8WSeO-wrmeF4lkRHdVzS6W25cg-v1.png)
**Gambar 5:** Cakupan Penyebaran Model GNN-LLM.

Bias dalam model GNN-LLM dapat muncul dari berbagai sumber. Misalnya, struktur graf dapat mencerminkan bias yang ada dalam jejaring sosial atau sistem ekonomi, di mana kelompok atau node tertentu diwakili secara berlebihan atau kurang, yang menyebabkan rekomendasi atau wawasan yang miring. Atribut node, yang berisi fitur individu seperti demografi pengguna atau riwayat transaksi, juga dapat memperkenalkan bias jika mereka mencerminkan atribut sensitif seperti ras, jenis kelamin, atau status sosial ekonomi. Selain itu, data pelatihan untuk GNN dan LLM sering kali berasal dari dataset historis atau tidak seimbang, menanamkan bias masyarakat ke dalam representasi yang dipelajari model. Secara matematis, ini dapat dinyatakan sebagai kemiringan dalam distribusi probabilitas bersyarat $P(y | X)$, di mana $y$ adalah output model dan $X$ mewakili data graf dan tekstual.

Menyebarkan model GNN-LLM di area dengan pengawasan peraturan, seperti keuangan atau layanan kesehatan, memerlukan komitmen terhadap transparansi dan akuntabilitas. Banyak organisasi sekarang diwajibkan memenuhi standar yang menuntut model dapat diinterpretasikan dan keputusan dapat dijelaskan. Eksplanabilitas dalam model GNN-LLM sering kali memerlukan pengembangan mekanisme untuk melacak bagaimana node tertentu dan hubungan mereka memengaruhi output model. Misalnya, mekanisme atensi dalam lapisan GNN dapat membantu menunjukkan node atau tepi mana yang memiliki dampak paling signifikan pada prediksi akhir, menawarkan bentuk interpretabilitas. Mematuhi prinsip-prinsip ini bukan hanya masalah tanggung jawab etis tetapi juga penting untuk memenuhi kepatuhan peraturan dan membina kepercayaan pengguna.

Mengimplementasikan strategi deteksi dan mitigasi bias sangat penting untuk penyebaran GNN-LLM yang etis. Pendekatan praktis adalah mengadopsi metode pelatihan yang sadar keadilan, seperti pembobotan ulang (*reweighting*), yang memberikan bobot pada node atau tepi berdasarkan representasi mereka dalam dataset. Misalnya, jika node minoritas kurang terwakili, mereka dapat diberi bobot lebih tinggi selama pelatihan untuk memastikan model mempelajari representasi yang seimbang. Pendekatan lain melibatkan *adversarial debiasing*, di mana jaringan adversarial dilatih bersama model GNN-LLM untuk mengidentifikasi dan menghukum prediksi yang bias.

Berikut adalah kode Python untuk mendemonstrasikan pelacakan bobot atensi dalam lapisan GNN untuk meningkatkan interpretabilitas model.

### Contoh Python: GNN dengan Atensi (Interpretabilitas)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GATLayer(nn.Module):
    """Graph Attention Layer sederhana untuk interpretasi"""
    def __init__(self, in_dim, out_dim):
        super(GATLayer, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        # Vektor atensi
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)

    def forward(self, adj, x):
        # adj: matriks adjasensi (N, N)
        # x: fitur node (N, in_dim)
        N = x.size(0)
        h = self.fc(x) # (N, out_dim)

        # Siapkan pasangan untuk atensi
        # (N, N, 2*out_dim)
        h_repeated_in_chunks = h.repeat_interleave(N, dim=0)
        h_repeated_alternating = h.repeat(N, 1)
        all_combinations_matrix = torch.cat([h_repeated_in_chunks, h_repeated_alternating], dim=1)
        all_combinations_matrix = all_combinations_matrix.view(N, N, 2 * h.size(1))

        # Hitung skor atensi mentah
        e = F.leaky_relu(self.attn_fc(all_combinations_matrix).squeeze(2)) # (N, N)

        # Masking: Hanya perhatikan tetangga yang ada (adj == 1)
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        
        # Softmax normalisasi bobot atensi
        attention = F.softmax(attention, dim=1)
        # attention_weights ini bisa disimpan/visualisasi untuk interpretasi
        
        h_prime = torch.mm(attention, h)
        return F.elu(h_prime), attention

def main_attention():
    # Graf dengan 3 node
    adj = torch.tensor([[1, 1, 0], [1, 1, 1], [0, 1, 1]], dtype=torch.float32)
    features = torch.randn(3, 16)
    
    gat = GATLayer(16, 8)
    output, attn_weights = gat(adj, features)
    
    print("Bobot Atensi antar Node (Matriks):")
    print(attn_weights)
    print("\nNode 1 paling memperhatikan Node:", torch.argmax(attn_weights[0]).item())

if __name__ == "__main__":
    main_attention()
```

Selain interpretabilitas, penyebaran model GNN-LLM harus mencakup mekanisme pemantauan untuk mendeteksi dan menangani bias atau *drift* secara waktu nyata. *Model drift* terjadi ketika distribusi data input berubah seiring waktu, menyebabkan akurasi model menurun. Misalnya, dalam analisis jejaring sosial, node dan tepi baru dapat mengubah struktur graf secara signifikan. Sistem pemantauan yang melacak metrik utama, seperti akurasi prediksi dan ukuran keadilan, dapat memperingatkan operator ketika kinerja turun, mendorong pelatihan ulang atau penyesuaian model.

Pentingnya penyebaran GNN-LLM yang etis ditekankan dalam industri seperti keuangan, di mana model yang bias dapat mengakibatkan praktik peminjaman atau investasi yang diskriminatif. Di domain layanan kesehatan, model GNN-LLM yang diterapkan pada jaringan pasien dapat meningkatkan diagnostik, tetapi harus dikalibrasi dengan cermat untuk menghindari pengenalan bias berdasarkan ras, usia, atau status sosial ekonomi. Pengawasan etis dan mitigasi bias sangat penting untuk menyebarkan model yang efektif sekaligus adil.

Sebagai kesimpulan, pertimbangan etis dan praktis adalah pusat dari penyebaran model GNN-LLM, terutama dalam aplikasi yang berdampak pada individu dan komunitas. Deteksi bias, interpretabilitas model, dan kepatuhan terhadap pedoman etika sangat penting untuk memastikan model-model ini beroperasi secara adil dan transparan. Bagian ini menggarisbawahi pentingnya menyebarkan model GNN-LLM secara bertanggung jawab, memberikan peta jalan untuk membangun sistem yang menghormati hak pengguna, mematuhi standar peraturan, dan memupuk kepercayaan pada aplikasi AI.

---

## 19.7. Studi Kasus dan Arah Masa Depan

Kombinasi Jaringan Saraf Graf (GNN) dan Model Bahasa Besar (LLM) telah membuka kemungkinan baru bagi aplikasi AI, terutama di domain yang memerlukan pemahaman tentang data terstruktur dan tidak terstruktur. Bagian ini mengeksplorasi beberapa studi kasus dunia nyata, mengilustrasikan bagaimana model GNN-LLM menangani tantangan kompleks dan mengungkap wawasan yang sebelumnya sulit diperoleh. Kasus-kasus ini memberikan pelajaran berharga tentang skalabilitas, interpretabilitas, dan penyebaran etis, sambil juga menyoroti tren yang muncul.

![Gambar 6: Aplikasi GNN-LLM dan Kompleksitasnya.](https://lmvr.rantai.dev//images/L6A504iArDGBvUq8WSeO-9RqlMpgzEshNJ3Zl9mQV-v1.png)
**Gambar 6:** Aplikasi GNN-LLM dan Kompleksitasnya.

Salah satu implementasi sukses dari model GNN-LLM adalah di bidang sistem rekomendasi. Dalam sebuah kasus baru-baru ini, sebuah platform streaming memanfaatkan model GNN-LLM untuk menggabungkan data graf jejaring sosial dengan preferensi pengguna yang diekstraksi dari ulasan teks. Dengan merepresentasikan pengguna sebagai node dan hubungan (seperti minat bersama atau riwayat tontonan) sebagai tepi, komponen GNN mempelajari pola perilaku pengguna yang rumit, sementara LLM memproses data tidak terstruktur dari ulasan teks untuk menangkap preferensi pengguna yang bernuansa. Pendekatan modalitas ganda ini meningkatkan akurasi rekomendasi platform sebesar lebih dari 30% dibandingkan dengan metode tradisional.

Dalam industri layanan kesehatan, model GNN-LLM telah menunjukkan potensi untuk meningkatkan akurasi diagnostik melalui penalaran graf pengetahuan. Sebuah studi baru-baru ini menyebarkan model GNN-LLM untuk memprediksi hasil pasien dengan mengintegrasikan data pasien terstruktur (seperti hasil lab) dengan catatan dokter yang tidak terstruktur. GNN bertugas memodelkan data relasional, seperti ko-morbiditas, sementara LLM menginterpretasikan data tekstual untuk menangkap konteks tambahan tentang gejala pasien. Tantangan utamanya adalah memastikan interpretabilitas model agar dokter dapat memverifikasi alasan di balik prediksi.

Seiring model GNN-LLM terus berkembang, tren yang muncul seperti AI yang dapat dijelaskan (*explainable AI*), rekomendasi berbasis graf, dan penalaran graf pengetahuan tingkat lanjut menjadi yang terdepan. Dalam keuangan, misalnya, model GNN-LLM digunakan untuk mendeteksi penipuan dengan menganalisis graf transaksi dan hubungan klien. Dengan eksplanabilitas, lembaga keuangan dapat memperoleh wawasan tentang bagaimana model-model ini sampai pada prediksi penipuan tertentu, sehingga memupuk kepercayaan dan transparansi.

Pelajaran dari penyebaran yang ada menunjukkan perlunya menyeimbangkan kompleksitas model dengan skalabilitas. Untuk mencapai skalabilitas tanpa mengorbankan akurasi, model hibrida sering mengimplementasikan teknik pengambilan sampel graf dan menggunakan kerangka kerja eksekusi multi-threaded. Dukungan untuk paralelisme dan kontrol atas alokasi memori sangat cocok untuk persyaratan ini, memungkinkan pengembang untuk membagi graf besar secara efektif dan meminimalkan latensi dalam aplikasi yang sensitif terhadap waktu.

Berikut adalah contoh Python terakhir yang menunjukkan integrasi antara penalaran graf pengetahuan (GNN) dan output bahasa (simulasi LLM).

### Contoh Python: Penalaran Graf Pengetahuan (GNN-LLM Pipeline)

```python
import torch
import torch.nn as nn

class KnowledgeGraphReasoning:
    def __init__(self, node_dim, relation_dim):
        # Parameter sederhana untuk merepresentasikan entitas
        self.entity_embeddings = nn.Embedding(100, node_dim)
        self.gnn = nn.Linear(node_dim, node_dim)
        self.llm_decoder = nn.Linear(node_dim, 50) # Simulasi output kosakata

    def reason(self, entity_ids, adj):
        # 1. Ambil embedding entitas
        e = self.entity_embeddings(entity_ids)
        
        # 2. Proses relasi melalui GNN
        # adj shape (num_entities, num_entities)
        # e shape (num_entities, node_dim)
        graph_rep = torch.mm(adj, e)
        h = torch.relu(self.gnn(graph_rep))
        
        # 3. Hasil agregat digunakan sebagai konteks untuk LLM
        context = torch.mean(h, dim=0, keepdim=True)
        
        # 4. LLM menghasilkan prediksi/teks (simulasi logit)
        logits = self.llm_decoder(context)
        return logits

def main_case_study():
    # Simulasi 3 entitas dalam subgraf pengetahuan
    entity_ids = torch.tensor([1, 5, 10])
    adj = torch.tensor([
        [1, 1, 0],
        [1, 1, 1],
        [0, 1, 1]
    ], dtype=torch.float32)
    
    reasoner = KnowledgeGraphReasoning(node_dim=32, relation_dim=16)
    output_logits = reasoner.reason(entity_ids, adj)
    
    print("Logit Prediksi dari Penalaran Graf:")
    print(output_logits.shape)

if __name__ == "__main__":
    main_case_study()
```

Salah satu tantangan utama dalam merealisasikan potensi penuh model GNN-LLM terletak pada ketersediaan data dan interpretabilitas model. Penyebaran yang efektif memerlukan akses ke data graf berkualitas tinggi dan dataset tekstual yang komprehensif, yang keduanya mungkin terbatas di industri seperti layanan kesehatan karena masalah privasi. Interpretabilitas juga tetap menjadi tantangan berkelanjutan, karena kompleksitas model GNN-LLM dapat menyulitkan pengguna untuk memahami bagaimana keputusan tertentu dicapai.

Sebagai kesimpulan, masa depan model GNN-LLM sangat menjanjikan, dengan aplikasi di berbagai bidang yang mengandalkan data terstruktur dan tidak terstruktur. Kemampuan sistem dalam hal efisiensi dan keamanan memberikan alat yang ampuh untuk memajukan kapabilitas GNN-LLM. Tantangan berkelanjutan dari ketersediaan data, interpretabilitas, dan penyebaran etis menggarisbawahi perlunya inovasi berkelanjutan, saat pengembang berupaya menciptakan model GNN-LLM yang transparan, adil, dan efektif yang dapat mentransformasi industri dan meningkatkan pengambilan keputusan.

---

## 19.8. Kesimpulan

Bab 19 memberikan pemahaman mendalam kepada pembaca tentang cara mengintegrasikan Jaringan Saraf Graf dan Model Bahasa Besar secara efektif. Dengan menguasai teknik-teknik ini, pembaca dapat mengembangkan model AI mutakhir yang mampu menangani data berstruktur graf yang kompleks sambil memastikan keadilan dan skalabilitas, memposisikan diri mereka di garis depan inovasi AI di berbagai industri.

### 19.8.1. Pembelajaran Lebih Lanjut dengan GenAI

- Jelaskan perbedaan mendasar antara jaringan saraf tradisional dan Jaringan Saraf Graf (GNN).
- Diskusikan tantangan utama penerapan GNN pada data graf, khususnya dalam hal skalabilitas dan *over-smoothing*.
- Deskripsikan proses pembangunan pipeline data graf yang tangguh.
- Analisis dampak teknik augmentasi graf pada performa GNN.
- Jelajahi berbagai arsitektur GNN, seperti GCN, GAT, dan GraphSAGE.
- Jelaskan peran agregasi lingkungan dalam GNN.
- Diskusikan pentingnya interpretabilitas model dalam GNN.
- Analisis tantangan mengintegrasikan GNN dengan Model Bahasa Besar (LLM).
- Jelajahi berbagai strategi untuk mengintegrasikan GNN dan LLM (sekuensial, paralel, hibrida).
- Bagaimana teknik *transfer learning* dapat membantu dalam skenario data graf yang terbatas?

### 19.8.2. Latihan Praktis

#### Latihan Mandiri 19.1: Membangun Pipeline Data Graf
Tujuan: Merancang sistem untuk menyerap data graf mentah (misal: koneksi jejaring sosial) dan melakukan normalisasi fitur node.

#### Latihan Mandiri 19.2: Melatih Jaringan Saraf Graf (GNN)
Tujuan: Mengimplementasikan operasi konvolusi graf dasar dan mengevaluasi model pada tugas klasifikasi node sederhana.

#### Latihan Mandiri 19.3: Integrasi GNN dengan LLM
Tujuan: Membangun pipeline yang menggabungkan embedding dari GNN dengan output teks dari LLM untuk tugas penautan entitas.

#### Latihan Mandiri 19.4: Menyebarkan Model GNN-LLM Waktu Nyata
Tujuan: Mengimplementasikan pipeline inferensi yang mampu menangani perubahan struktur graf secara dinamis dengan latensi rendah.

#### Latihan Mandiri 19.5: Audit Etika dan Bias pada Model Graf
Tujuan: Mengembangkan modul untuk mendeteksi apakah model rekomendasi berbasis graf memberikan hasil yang bias terhadap kelompok demografis tertentu.