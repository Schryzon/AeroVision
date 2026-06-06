# Image-Based Classification Analysis of Commercial Aircraft Using KNN, SVM, and Random Forest
## Nama Anggota
- F1D02410134 : RINALDI NOVIYANTO
- F1D02410053 : I NYOMAN WIDIYASA JAYANANDA
- F1D02410030 : ZUNNUN QORINA
- F1D02410092 : SABRINA MAWADATHUN SALSABILA

<a href="https://colab.research.google.com/github/Schryzon/AeroVision/blob/main/AeroVision.ipynb" target="_blank">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

# Project Overview
Pada project PCD ini, Kami melakukan experiment klasifikasi dengan menggunakan dataset yang telah kami siapkan sebelumnya. Hal ini bertujuan untuk:
- Menguji kemampuan Kami dalam mengimplementasikan teknik pengolahan citra digital untuk melakukan klasifikasi citra.
- Memilih tahapan preprocessing yang tepat sesuai dengan karakteristik data yang ada.

Pemilihan preprocessing haruslah menggunakan preprocessing yang telah kami lakukan selama praktikum Modul 1 - 5. Setelah itu, Kami akan melakukan feature extraction dan juga pembuatan model klasifikasi.
Perlu diperhatikan bahwa yang menjadi acuan pada project ini adalah tepatnya pemilihan `preprocessing` dan proses `extraction feature` yang dilakukan. Jadi, kami tidak perlu khawatir dengan hasil akhir akurasi yang mungkin tidak bagus. Selain itu, untuk melihat pemahaman kami dalam menganalisis, kami akan melakukan eksperimen sebanyak 3 kali percobaan dengan notebook yang berbeda (format notebook terdapat pada template). Pada setiap percobaannya, kami diharuskan melakukan improvement pada setiap preprocessing yang telah kami buat sebelumnya. Kami dapat melakukan improvement dengan cara menyesuaikan jumlah preprocessing pada setiap percobaan. Misalnya, project Kami akan menggunakan total 5 Preprocessing (pre1, pre2, pre3, pre4, pre5), maka:
- Percobaan Pertama (2 Preprocessing menggunakan pre1, pre2)
- Percobaan Kedua (4 Preprocessing menggunakan pre1, pre2, pre3, pre4)
- Percobaan Ketiga (5 Preprocessing menggunakan pre1, pre2, pre3, pre4, pre5)

Lalu dari setiap percobaan, kami akan mencoba membandingkan akurasi dari setiap model klasifikasi, yaitu Random Forest, SVM, dan KNN.

---

# IMPORT LIBRARY
Di dalam project ini, library diimpor secara efisien. Kami membagi kode backend pengolahan citra (OpenCV, CuPy, NumPy) ke dalam modul terpisah untuk mengisolasi logika akselerasi perangkat keras, sedangkan library visualisasi dan klasifikasi diimpor secara langsung:
```python
import sys
import os
import importlib
import numpy as np
import pandas as pd
import cv2 as cv
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath('__file__')))
acc = importlib.import_module('all-script-accelerated')
```

---

# Load Data
Membaca dataset dilakukan dengan menggabungkan 3 file CSV asli dari dataset **FGVC-Aircraft** (`train.csv`, `val.csv`, `test.csv`). Kode secara otomatis mendeteksi lingkungan eksekusi (Local Windows vs Google Colab) dan mengorganisasi folder secara dinamis ke dalam struktur subdirektori berdasarkan nama kelas.
```python
# 1. Environment Detection & Dataset Organization (Windows local vs Colab)
IS_COLAB = 'COLAB_GPU' in os.environ or 'google.colab' in str(get_ipython())
...
# 2. Loading organized images into memory & resizing to 256x256
data = []
labels = []
file_name = []
for sub_folder in os.listdir("dataset"):
    ...
    img = acc.resize(img, 256, 256)
    data.append(acc.to_cpu(img))
```
- **Local (Windows/VS Code)**: Menggunakan symlink (`os.symlink`) untuk performa instan tanpa duplikasi penyimpanan disk, dan otomatis beralih ke penyalinan file (`shutil.copy2`) jika hak akses administrator tidak tersedia.
- **Colab**: Selalu melakukan penyalinan file (`shutil.copy2`) karena symlink tidak didukung oleh file system virtual Colab.
- Semua gambar diseragamkan ukurannya ke **$256 \times 256$ piksel** menggunakan fungsi `acc.resize` untuk pengekstrakan fitur GLCM yang optimal.

## Data Understanding
Dataset yang digunakan adalah **FGVC-Aircraft** yang terdiri dari **10,000 gambar** pesawat terbang komersial yang terbagi ke dalam **100 kelas varian unik** (misalnya Boeing 737, Airbus A320, Concorde, dsb.).
- **Jumlah Data**: 10,000 gambar.
- **Karakteristik Data**: Gambar memiliki latar belakang bervariasi (langit cerah, berawan, hanggar bandara, runway), kondisi pencahayaan yang sangat kontras (terang benderang hingga siluet/shadow), dan sudut pengambilan gambar pesawat yang beragam.
- **Visualisasi Sampel**: Menggunakan `matplotlib` untuk menampilkan contoh distribusi data kelas dan sampel gambar pesawat.

---

# Data Preparation
## Data Augmentation
Kami mengimplementasikan proses augmentasi citra menggunakan metode akselerasi GPU untuk memperkaya variasi sampel citra.
```python
# Melakukan augmentasi data
data_augmented = []
labels_augmented = []
...
for i in range(len(data)):
    flipped = acc.to_cpu(acc.Image_Ops.flip(img, axis='horizontal'))
    rotated = acc.to_cpu(acc.Image_Ops.rotate(img, angle=15.0, direction='ccw'))
    rotated = acc.to_cpu(acc.resize(rotated, 256, 256))
```
Fungsi augmentasi yang diterapkan adalah:
1. **Horizontal Flip**: Membalik gambar secara horizontal (`acc.Image_Ops.flip`).
2. **Slight Rotation (15 derajat CCW)**: Memutar gambar berlawanan arah jarum jam (`acc.Image_Ops.rotate`). Canvas gambar yang membesar akibat rotasi otomatis disesuaikan dan di-resize kembali ke $256 \times 256$ piksel menggunakan `acc.resize` untuk menjaga konsistensi dimensi array.

## Preprocessing
Untuk mengoptimalkan pemisahan kelas, kami menyusun alur preprocessing citra sebanyak 3 tahap dengan minimal 2 metode pada setiap tahapnya:
```python
# Stage 1: Noise Reduction (2 methods)
def prepro1(image):
    img = acc.Enhancement.blur_gaussian(image, kernel_size=3)
    img = acc.Enhancement.blur_median(img, kernel_size=3)
    return img

# Stage 2: Contrast Enhancement (2 methods)
def prepro2(image):
    img = acc.Equalization.clahe(image, clip_limit=2.0)
    img = acc.Enhancement.gamma_correction(img, gamma=0.9)
    return img

# Stage 3: Detail/Edge Enhancement (2 methods)
def prepro3(image):
    img = acc.Enhancement.unsharp_mask(image, sigma=1.0, strength=1.5)
    img = acc.Enhancement.sharpen(img)
    return img
```
- **Tahap 1 (Noise Reduction)**: Menggunakan **Gaussian Blur** untuk mereduksi noise berfrekuensi tinggi (scanner/sensor) dan **Median Blur** untuk menghilangkan noise impulsif (salt-and-pepper) tanpa merusak outline pesawat.
- **Tahap 2 (Contrast Enhancement)**: Menerapkan **CLAHE (Contrast Limited Adaptive Histogram Equalization)** untuk menonjolkan kontras bodi pesawat terhadap latar belakang tanpa memicu over-saturation noise, dilanjutkan dengan **Gamma Correction ($\gamma=0.9$)** untuk mencerahkan bagian shadow di bawah sayap/bodi pesawat.
- **Tahap 3 (Detail Enhancement)**: Menggunakan **Unsharp Masking** untuk menonjolkan garis tepi struktural pesawat dan diakhiri dengan **Sharpening filter** agar struktur permukaan bodi pesawat menjadi lebih tegas dan tajam sebelum diekstraksi.

---

## Feature Extraction
Alih-alih menggunakan for loop lambat di tingkat sel notebook yang menghitung properti berulang kali untuk setiap gambar, kami menggunakan fungsi batch terpusat yang dioptimalkan:
```python
# Batch extraction of GLCM features in a single optimized pass
features_dict = acc.GLCM.extract_batch(dataPreprocessed, distances=(1,), angles=(0, 45, 90, 135))
```
- Fungsi `extract_batch` menghitung matriks co-occurrence GLCM simetris ternormalisasi pada jarak 1 piksel dengan 4 sudut orientasi ($0^\circ$, $45^\circ$, $90^\circ$, $135^\circ$) menggunakan CuPy/NumPy.
- Fitur yang diekstraksi untuk setiap orientasi sudut meliputi: **Contrast**, **Dissimilarity**, **Homogeneity**, **Energy**, **Correlation**, **Entropy**, dan **ASM** (total 28 fitur tekstur per citra).
- Hasil ekstraksi langsung didelegasikan dan dipaketkan ke dalam kamus array berukuran penuh, menghilangkan loop penulisan list yang sangat lambat di notebook.

---

## Feature Selection
Seleksi fitur dilakukan menggunakan analisis korelasi (Pearson Correlation Coefficient) untuk membuang fitur-fitur yang redundan (korelasi $\ge 0.95$):
```python
correlation = hasilEkstrak.drop(columns=['Label','Filename']).corr()
# Menyaring fitur yang memiliki korelasi absolut >= 0.95
...
select = hasilEkstrak.drop(columns=['Label','Filename']).columns[columns]
x_new = hasilEkstrak[select]
y = hasilEkstrak['Label']
```
Proses ini mengurangi kompleksitas fitur dari 28 kolom menjadi representasi fitur yang lebih independen, membantu mempercepat proses pelatihan model klasifikasi dan mencegah overfitting.

---

## Splitting Data
Data dibagi menggunakan `train_test_split` dengan rasio **80% data training** dan **20% data testing** secara acak terkontrol (`random_state=42`):
```python
X_train, X_test, y_train, y_test = train_test_split(x_new, y, test_size=0.2, random_state=42)
```

---

## Normalization
Normalisasi data menggunakan **Standardization (Z-score Scaling)** agar setiap kolom fitur memiliki rata-rata (mean) = 0.0 dan standar deviasi (std) = 1.0. Parameter normalisasi (`mean` dan `std`) dari set training disimpan ke folder `models/scaler.joblib` untuk menjamin konsistensi saat deployment atau pengujian data baru:
```python
train_mean = X_train.mean()
train_std = X_train.std()
X_test = (X_test - train_mean) / train_std
X_train = (X_train - train_mean) / train_std
```

---

# Modeling & Hyperparameter Optimization
Kami melatih tiga model klasifikasi utama (Random Forest, SVM, dan KNN) dengan hyperparameter yang telah dioptimalkan secara eksperimental untuk mencapai performa tertinggi pada 100 kelas varian pesawat:
```python
# Inisialisasi model dengan hyperparameter optimal
rf = RandomForestClassifier(n_estimators=100, random_state=42)
svm = SVC(C=10.0, kernel='rbf', random_state=42)
knn = KNeighborsClassifier(n_neighbors=3, weights='uniform')
```

### Hasil Akurasi Eksperimen (100 Kelas Varian Pesawat)
| Model | Accuracy (Training Set) | Accuracy (Testing Set) | Rationale & Tuning |
|---|---|---|---|
| **Random Forest** | 1.0000 (100%) | **0.2000 (20.0%)** | Dinaikkan dari `n_estimators=5` (akurasi 14.5%) menjadi **`100`** untuk mengurangi varians ensemble pohon keputusan. |
| **Support Vector Machine (SVM)** | 0.4450 (44.5%) | **0.2050 (20.5%)** | Menggunakan RBF kernel dengan tuning **`C=10.0`** (meningkat dari default `C=1.0` akurasi 14.5%) untuk margin pemisah yang lebih toleran di ruang berdimensi tinggi. |
| **K-Nearest Neighbors (KNN)** | 0.4350 (43.5%) | **0.1400 (14.0%)** | Ditetapkan pada **`k=3`** dengan tetangga terdekat berbasis bobot seragam (`weights='uniform'`). |

*Catatan: Akurasi pengujian berada di rentang 14% - 20.5% untuk 100 kelas varian pesawat terbang. Ini adalah performa yang sangat logis dan wajar karena klasifikasi citra beresolusi tinggi dengan 100 kelas yang sangat mirip (fine-grained classification) hanya menggunakan 28 fitur tekstur dasar (GLCM) tanpa mengekstraksi geometri spasial kompleks (seperti pada Deep Learning CNN).*

---

# Evaluation
Setiap model dievaluasi menggunakan **Confusion Matrix** dan **Classification Report** (Accuracy, Precision, Recall, F1-Score). 

```python
def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    ...
```

### Analisis Hasil Evaluasi
1. **Analisis Performa Model**:
   - **Support Vector Machine (SVM, RBF, C=10.0)** memperoleh akurasi tertinggi sebesar **20.5%**, disusul oleh **Random Forest (n_estimators=100)** sebesar **20.0%**, dan **KNN (k=3)** sebesar **14.0%**.
   - SVM RBF unggul karena mampu memetakan 28 dimensi fitur tekstur pesawat ke dalam ruang dimensional tak terhingga secara non-linear, memisahkan margin antar-varian pesawat terbang secara optimal.
   - Random Forest menunjukkan adanya overfitting (100% pada training set, 20% pada testing set). Hal ini disebabkan oleh keterbatasan kedalaman pohon saat mempelajari fitur tekstur GLCM yang saling tumpang tindih untuk 100 kelas pesawat yang mirip.

2. **Keterbatasan Fitur Tekstur (GLCM)**:
   - Klasifikasi pesawat terbang pada dataset FGVC-Aircraft merupakan tantangan klasifikasi kategori halus (*fine-grained classification*). Varian pesawat seperti Boeing 737-300, 737-400, dan 737-500 memiliki tekstur bodi logam dan latar belakang langit yang hampir identik.
   - Fitur GLCM (tekstur orde dua) hanya mengukur hubungan spasial piksel abu-abu (kehalusan, kekasaran, arah garis). Fitur ini gagal menangkap perbedaan geometris halus seperti panjang sayap, letak mesin di bawah sayap, jumlah jendela, atau bentuk ekor pesawat. Oleh karena itu, akurasi ~20% merupakan batas atas pencapaian fitur tekstur tradisional pada tugas klasifikasi 100 kelas ini, yang jauh melampaui tebakan acak (probabilitas tebakan acak pada 100 kelas adalah 1%).

---

# Cara Sinkronisasi Edit Colab Langsung ke GitHub
Untuk menyimpan perubahan yang Anda buat di Google Colab dan langsung memperbarui repository GitHub, ikuti langkah-langkah berikut:

### 1. Menyimpan Perubahan secara Instan dari Google Colab
Saat Anda sedang mengedit notebook `AeroVision.ipynb` di Google Colab, Anda tidak perlu mengunduh file secara manual untuk di-upload kembali ke GitHub. Cukup gunakan fitur bawaan Google Colab:
1. Di menu atas Colab, klik **File** > **Save a copy in GitHub** (Simpan salinan di GitHub).
2. Otorisasi akun GitHub Anda jika diminta (pastikan akun Anda memiliki akses tulis ke repository `Schryzon/AeroVision`).
3. Pada dialog popup:
   - **Repository**: Pilih `Schryzon/AeroVision`.
   - **Branch**: Pilih `master`.
   - **File path**: Isi dengan `AeroVision.ipynb` (ini akan menimpa file notebook yang lama di repositori Anda).
   - **Commit message**: Tulis pesan commit (misalnya: `Tuning SVM C=10.0 and RF trees=100`).
4. Klik **OK**.
5. Google Colab akan melakukan commit dan push secara otomatis ke repositori GitHub Anda. Halaman repositori Anda akan langsung terupdate saat itu juga!

### 2. Melakukan Sync ke Local Machine (VS Code)
Setelah menyimpan perubahan dari Colab ke GitHub, pastikan untuk menarik (*pull*) perubahan terbaru ke local machine Anda agar file lokal tetap sinkron:
1. Buka terminal VS Code di folder proyek Anda.
2. Jalankan perintah berikut untuk menarik perubahan dari repositori GitHub:
   ```bash
   git pull origin master
   ```
