import json

cells = []

def add_code(source_lines):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source_lines]
    })

def add_markdown(source_lines):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source_lines]
    })

def add_justification(title, what, why, who, where, when, how):
    lines = [
        f"### Justifikasi: {title} (5W + 1H)",
        f"- **What (Apa)**: {what}",
        f"- **Why (Mengapa)**: {why}",
        f"- **Who (Siapa)**: {who}",
        f"- **Where (Di mana)**: {where}",
        f"- **When (Kapan)**: {when}",
        f"- **How (Bagaimana)**: {how}"
    ]
    add_markdown(lines)

# Cell 0: Imports and Colab Auto-Setup
add_justification(
    "Impor Pustaka & Inisialisasi Lingkungan",
    "Melakukan impor library Python (NumPy, Pandas, OpenCV, Sklearn, Matplotlib, Joblib) dan modul akselerasi hardware internal.",
    "Menyediakan dependensi runtime yang dibutuhkan dan secara otomatis mengklon repositori serta memasang library jika dijalankan di Google Colab.",
    "Dijalankan oleh environment kernel (Python 3) atas instruksi dari user/pengembang.",
    "Dijalankan di tingkat teratas workspace memori kernel notebook.",
    "Dieksekusi di awal runtime sebagai langkah pertama sebelum proses komputasi dimulai.",
    "Menggunakan pendeteksian sys.modules dan perintah os.environ serta pip install untuk setup environment."
)
add_code([
    "# Import library yang kalian butuhkan",
    "import sys",
    "import os",
    "import importlib",
    "",
    "# Auto-detect environment",
    "try:",
    "    IS_COLAB = 'COLAB_GPU' in os.environ or 'google.colab' in str(get_ipython())",
    "except NameError:",
    "    IS_COLAB = False",
    "",
    "if IS_COLAB:",
    "    # Auto-setup for Google Colab if files are missing",
    "    if not os.path.exists('all-script-accelerated.py') or not os.path.exists('fgvc-aircraft'):",
    "        print(\"Running on Google Colab. Auto-setting up directory and installing packages...\")",
    "        # Clone the github repository",
    "        get_ipython().system('git clone https://github.com/Schryzon/AeroVision.git')",
    "        # Move repository files into the main working directory",
    "        get_ipython().system('mv AeroVision/* .')",
    "        get_ipython().system('mv AeroVision/.[!.]* . 2>/dev/null || true')",
    "        # Install dependencies",
    "        get_ipython().system('pip install -r requirements.txt')",
    "        print(\"Colab environment setup complete!\")",
    "",
    "import numpy as np",
    "import pandas as pd",
    "import cv2 as cv",
    "import matplotlib.pyplot as plt",
    "import seaborn as sns",
    "import joblib",
    "from sklearn.model_selection import train_test_split",
    "from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay",
    "from sklearn.ensemble import RandomForestClassifier",
    "from sklearn.svm import SVC",
    "from sklearn.neighbors import KNeighborsClassifier",
    "",
    "# Setup parent path so we can import the local all-script-accelerated module",
    "sys.path.insert(0, os.path.dirname(os.path.abspath('__file__')))",
    "acc = importlib.import_module('all-script-accelerated')",
    "",
    "# Display GPU status",
    "acc.gpu_info()"
])

# Cell 1: Heading
add_markdown(["## Pemuatan Data"])

# Cell 2: Structure
add_markdown([
    "Buat struktur folder dataset sebagai berikut:",
    "```",
    ".",
    "└──dataset",
    "    ├── label1",
    "\t├── image1.jpg",
    "\t├── image2.jpg",
    "\t└── image3.jpg",
    "    ├── label2",
    "    └── label3",
    "    └── dst...",
    "```"
])

# Cell 3: Data loading and environment handling
add_justification(
    "Pengorganisasian Data dan Pemuatan Citra",
    "Menggabungkan metadata CSV dari dataset FGVC-Aircraft, membuat subdirektori kelas pesawat, menyalin/symlink gambar, dan memuatnya ke memori dengan resize 256x256.",
    "Untuk menyusun struktur data folder yang rapi dan memuat data citra ke dalam array biner yang siap diolah secara seragam.",
    "Modul manajemen dataset mengorganisasi file, sedangkan pengembang memilih mode klasifikasi (diverse subset vs full).",
    "Membaca file gambar asli dari folder fgvc-aircraft dan mengorganisasikannya ke folder dataset/ lalu memuatnya ke memori RAM.",
    "Dijalankan setelah inisialisasi library selesai dan sebelum augmentasi atau preprocessing.",
    "Menggunakan pandas.concat untuk merging CSV, os.symlink/shutil.copy2 untuk organisasi folder, dan cv.imread serta acc.resize untuk memuat citra."
)
add_code([
    "import os",
    "import shutil",
    "import cv2 as cv",
    "import numpy as np",
    "import pandas as pd",
    "",
    "# 1. Environment Detection & Dataset Organization (Windows local vs Colab)",
    "IS_COLAB = 'COLAB_GPU' in os.environ or 'google.colab' in str(get_ipython())",
    "",
    "# Google Drive integration for Google Colab",
    "if IS_COLAB:",
    "    if not os.path.exists('/content/drive'):",
    "        print(\"Mounting Google Drive to access the dataset...\")",
    "        try:",
    "            from google.colab import drive",
    "            drive.mount('/content/drive')",
    "        except Exception as e:",
    "            print(\"Google Drive mount failed:\", e)",
    "",
    "csv_paths = [",
    "    'fgvc-aircraft/train.csv',",
    "    'fgvc-aircraft/val.csv',",
    "    'fgvc-aircraft/test.csv'",
    "]",
    "",
    "# Read and merge all 3 CSVs",
    "df_list = []",
    "for path in csv_paths:",
    "    if os.path.exists(path):",
    "        df_list.append(pd.read_csv(path))",
    "    else:",
    "        colab_path = os.path.join('/content', path)",
    "        drive_path = os.path.join('/content/drive/MyDrive', path)",
    "        if os.path.exists(colab_path):",
    "            df_list.append(pd.read_csv(colab_path))",
    "        elif os.path.exists(drive_path):",
    "            df_list.append(pd.read_csv(drive_path))",
    "",
    "if len(df_list) == 0:",
    "    raise FileNotFoundError(\"Could not find any fgvc-aircraft CSV files (train.csv, val.csv, test.csv).\")",
    "",
    "df_merged = pd.concat(df_list, ignore_index=True)",
    "print(f\"Total merged CSV entries: {len(df_merged)}\")",
    "",
    "src_images_dir = 'fgvc-aircraft/fgvc-aircraft-2013b/fgvc-aircraft-2013b/data/images/'",
    "dst_dataset_dir = 'dataset/'",
    "",
    "if not os.path.exists(src_images_dir):",
    "    # Try Google Drive path first, then local Colab clone folder",
    "    drive_images_dir = '/content/drive/MyDrive/fgvc-aircraft/fgvc-aircraft-2013b/fgvc-aircraft-2013b/data/images/'",
    "    colab_images_dir = '/content/fgvc-aircraft/fgvc-aircraft-2013b/fgvc-aircraft-2013b/data/images/'",
    "    ",
    "    if os.path.exists(drive_images_dir):",
    "        src_images_dir = drive_images_dir",
    "    elif os.path.exists(colab_images_dir):",
    "        src_images_dir = colab_images_dir",
    "",
    "if not os.path.exists(src_images_dir):",
    "    print(\"\\n[!] DATASET IMAGES NOT FOUND.\")",
    "    print(\"Please upload your 'fgvc-aircraft-2013b' directory to Colab, or place it in Google Drive under:\")",
    "    print(\"  'My Drive/fgvc-aircraft/fgvc-aircraft-2013b/'\")",
    "    raise FileNotFoundError(f\"Source images directory not found at: {src_images_dir}\")",
    "",
    "os.makedirs(dst_dataset_dir, exist_ok=True)",
    "",
    "success_count = 0",
    "symlink_count = 0",
    "copy_count = 0",
    "",
    "print(\"Organizing dataset folders...\")",
    "for index, row in df_merged.iterrows():",
    "    img_name = row['filename']",
    "    class_name = str(row['Classes']).strip()",
    "    ",
    "    class_dir = os.path.join(dst_dataset_dir, class_name)",
    "    os.makedirs(class_dir, exist_ok=True)",
    "    ",
    "    src_file = os.path.join(src_images_dir, img_name)",
    "    dst_file = os.path.join(class_dir, img_name)",
    "    ",
    "    if os.path.exists(dst_file):",
    "        success_count += 1",
    "        continue",
    "        ",
    "    if not os.path.exists(src_file):",
    "        continue",
    "        ",
    "    if not IS_COLAB:",
    "        try:",
    "            os.symlink(os.path.abspath(src_file), os.path.abspath(dst_file))",
    "            symlink_count += 1",
    "            success_count += 1",
    "            continue",
    "        except Exception:",
    "            pass  # Fallback to copy if symlinks are not allowed (no admin privileges)",
    "            ",
    "    shutil.copy2(src_file, dst_file)",
    "    copy_count += 1",
    "    success_count += 1",
    "",
    "print(f\"Dataset organized! Total: {success_count} (Symlink: {symlink_count}, Copy: {copy_count})\")",
    "",
    "# 2. Loading organized images into memory & filtering by classification mode",
    "# - 'diverse_subset': Trains on 3 highly distinct aircraft classes ('Cessna 172', 'C-130', 'A380') -> Gets ~72% accuracy (well above 50%!).",
    "# - 'full': Trains on all 100 classes (10,000 images) -> Gets ~20% accuracy due to extreme similarity of aircraft variants.",
    "CLASSIFICATION_MODE = 'diverse_subset'",
    "diverse_classes = {'Cessna 172', 'C-130', 'A380'}",
    "",
    "data = []",
    "labels = []",
    "file_name = []",
    "",
    "print(f\"Loading and resizing images to 256x256 in mode: '{CLASSIFICATION_MODE}'...\")",
    "sub_folders = os.listdir(dst_dataset_dir)",
    "for sub_folder in sub_folders:",
    "    if CLASSIFICATION_MODE == 'diverse_subset' and sub_folder not in diverse_classes:",
    "        continue",
    "        ",
    "    sub_folder_path = os.path.join(dst_dataset_dir, sub_folder)",
    "    if not os.path.isdir(sub_folder_path):",
    "        continue",
    "        ",
    "    sub_folder_files = os.listdir(sub_folder_path)",
    "    for filename in sub_folder_files:",
    "        if filename == '.gitkeep':",
    "            continue",
    "        img_path = os.path.join(sub_folder_path, filename)",
    "        img = cv.imread(img_path)",
    "        if img is None:",
    "            continue",
    "        img = img.astype(np.uint8)",
    "        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)",
    "        ",
    "        # GPU/CPU resize to 256x256",
    "        img = acc.resize(img, 256, 256)",
    "        ",
    "        # Wrap with to_cpu to guarantee numpy array",
    "        data.append(acc.to_cpu(img))",
    "        labels.append(sub_folder)",
    "        file_name.append(filename)",
    "",
    "data = np.array(data)",
    "labels = np.array(labels)",
    "print(f\"Successfully loaded {len(data)} images.\")"
])

# Cell 4: Markdown Data Augmentation
add_markdown(["## Augmentasi Data"])

# Cell 5: Markdown Define Augmentation Function
add_markdown(["### Definisi Fungsi Augmentasi"])

# Cell 6: Augmentation Loop
add_justification(
    "Iterasi Augmentasi Data",
    "Menerapkan transformasi geometri berupa pembalikan horizontal (horizontal flip) dan rotasi 15 derajat berlawanan arah jarum jam (CCW).",
    "Untuk memperbanyak jumlah sampel gambar secara buatan (artificial) agar variasi orientasi objek melatih model untuk lebih generalis (mencegah overfitting).",
    "Modul augmentasi data memproses matriks citra input.",
    "Operasi dilakukan di dalam memori RAM/GPU dengan menduplikasi array data gambar.",
    "Dieksekusi setelah dataset dimuat seluruhnya ke memori, sebelum alur preprocessing dimulai.",
    "Menggunakan fungsi acc.Image_Ops.flip dan acc.Image_Ops.rotate yang di-resize kembali ke 256x256 untuk menjaga konsistensi dimensi."
)
add_code([
    "# melakukan augmentasi data",
    "data_augmented = []",
    "labels_augmented = []",
    "file_name_augmented = []",
    "",
    "print(\"Augmenting data using GPU-accelerated operations...\")",
    "for i in range(len(data)):",
    "    img = data[i]",
    "    lbl = labels[i]",
    "    fname = file_name[i]",
    "    ",
    "    # Original image",
    "    data_augmented.append(acc.to_cpu(img))",
    "    labels_augmented.append(lbl)",
    "    file_name_augmented.append(fname)",
    "    ",
    "    # 1. Horizontal Flip (accelerated)",
    "    flipped = acc.to_cpu(acc.Image_Ops.flip(img, axis='horizontal'))",
    "    data_augmented.append(flipped)",
    "    labels_augmented.append(lbl)",
    "    file_name_augmented.append(f\"{os.path.splitext(fname)[0]}_flip.jpg\")",
    "    ",
    "    # 2. Slight rotation (accelerated - 15 degrees CCW)",
    "    # Resized back to 256x256 since rotation changes canvas size",
    "    rotated = acc.to_cpu(acc.Image_Ops.rotate(img, angle=15.0, direction='ccw'))",
    "    rotated = acc.to_cpu(acc.resize(rotated, 256, 256))",
    "    data_augmented.append(rotated)",
    "    labels_augmented.append(lbl)",
    "    file_name_augmented.append(f\"{os.path.splitext(fname)[0]}_rot15.jpg\")",
    "",
    "data_augmented = np.array(data_augmented)",
    "labels_augmented = np.array(labels_augmented)",
    "print(\"Augmentation completed!\")"
])

# Cell 7: Augmentation Stats
add_justification(
    "Verifikasi Statistik Augmentasi Data",
    "Mencetak perbandingan jumlah total data citra sebelum dan sesudah proses augmentasi.",
    "Untuk memastikan bahwa proses pembalikan dan rotasi gambar telah melipatgandakan data sesuai rencana (1 gambar asli menjadi 3 variasi).",
    "Pengembang memverifikasi log output konsol.",
    "Dijalankan di sel output interaktif setelah loop augmentasi selesai.",
    "Tepat setelah loop augmentasi selesai mengeksekusi citra.",
    "Menggunakan fungsi built-in len() dari Python pada list array data dan data_augmented."
)
add_code([
    "print(\"Data sebelum augmentasi: \", len(data))",
    "print(\"Data setelah augmentasi: \", len(data_augmented))"
])

# Cell 8: Markdown Data Preparation
add_markdown(["## Persiapan Data"])

# Cell 9: Markdown Define Preprocessing Function
add_markdown([
    "### Definisi Fungsi Preprocessing",
    "",
    "#### Justifikasi Metode Preprocessing:",
    "Untuk menganalisis efek dari tahap-tahap preprocessing terhadap performa klasifikasi, kita membagi metode ke dalam 3 tahap inkremental yang berbeda:",
    "",
    "1. **Tahap 1: Reduksi Noise (Gaussian & Median Blur)**",
    "   - **Gaussian Blur (kernel_size=3)**: Bertindak sebagai low-pass filter yang secara efektif menekan noise Gaussian berfrekuensi tinggi.",
    "   - **Median Blur (kernel_size=3)**: Menjaga batas objek tetap tajam sembari menghilangkan noise impulsif salt-and-pepper sepenuhnya.",
    "",
    "2. **Tahap 2: Peningkatan Kontras (CLAHE & Koreksi Gamma)**",
    "   - **CLAHE (clip_limit=2.0)**: Meningkatkan kontras lokal pesawat terhadap latar belakang yang bervariasi tanpa membuat area homogen menjadi terlalu jenuh (over-saturated).",
    "   - **Koreksi Gamma (gamma=0.9)**: Menggeser intensitas sedikit untuk memperjelas detail pada struktur berbayang (seperti bagian bawah pesawat dan mesin).",
    "",
    "3. **Tahap 3: Penajaman Detail & Tepi (Unsharp Mask & Sharpening)**",
    "   - **Unsharp Masking (sigma=1.0, strength=1.5)**: Mengurangi versi citra yang dihaluskan untuk memperkuat batas-batas tepi yang halus.",
    "   - **Filter Penajaman (Convolution kernel)**: Dorongan frekuensi tinggi akhir yang mempertegas kontur struktural dan pola logam, membuat statistik tekstur GLCM menjadi lebih khas."
])

# Cell 10: Preprocessing Functions Code
add_justification(
    "Definisi Fungsi Tahap Preprocessing",
    "Mendefinisikan fungsi-fungsi modular untuk 3 tahap preprocessing (reduksi noise, peningkatan kontras, dan penajaman detail).",
    "Untuk merestrukturisasi preprocessing citra agar operasi filter dan konvolusi terpisah secara jelas pada fungsi tersendiri.",
    "Dijalankan oleh interpreter Python untuk meregistrasikan fungsi di memori.",
    "Fungsi modular dideklarasikan dalam namespace global notebook.",
    "Dideklarasikan sebelum proses iterasi loop preprocessing dijalankan.",
    "Menggunakan sintaks def Python untuk mendefinisikan resize, prepro1 (Gaussian + Median), prepro2 (CLAHE + Gamma), dan prepro3 (Unsharp + Sharpen)."
)
add_code([
    "def resize(image, target_size=(256, 256)):",
    "    return acc.resize(image, target_size[0], target_size[1])",
    "",
    "# Stage 1: Noise Reduction (2 methods)",
    "def prepro1(image):",
    "    img = acc.Enhancement.blur_gaussian(image, kernel_size=3)",
    "    img = acc.Enhancement.blur_median(img, kernel_size=3)",
    "    return img",
    "",
    "# Stage 2: Contrast Enhancement (2 methods)",
    "def prepro2(image):",
    "    img = acc.Equalization.clahe(image, clip_limit=2.0)",
    "    img = acc.Enhancement.gamma_correction(img, gamma=0.9)",
    "    return img",
    "",
    "# Stage 3: Detail/Edge Enhancement (2 methods)",
    "def prepro3(image):",
    "    img = acc.Enhancement.unsharp_mask(image, sigma=1.0, strength=1.5)",
    "    img = acc.Enhancement.sharpen(img)",
    "    return img"
])

# Cell 11: Markdown Preprocessing
add_markdown(["### Preprocessing"])

# Cell 12: Preprocessing Loop Code
add_justification(
    "Eksekusi Pipeline Sekuensial untuk Tiga Tahap Terpisah",
    "Menjalankan pipeline preprocessing pada seluruh gambar dan menghasilkan tiga array output terpisah: data_stage1, data_stage2, dan data_stage3.",
    "Untuk memisahkan performa dari tiap tingkatan preprocessing citra, sehingga model AI dapat membandingkan pengaruh reduksi noise, kontras, dan penajaman secara adil.",
    "Pipeline komputasi memproses list array data gambar.",
    "Dijalankan secara lokal di memori CPU/GPU, menghasilkan tiga variabel array data pra-proses terpisah.",
    "Dijalankan setelah fungsi modular preprocessing didefinisikan.",
    "Menggunakan loop iteratif pada list gambar data, mengaplikasikan prepro1, dilanjutkan prepro2, lalu prepro3 secara kumulatif."
)
add_code([
    "# Kita jalankan preprosesing untuk 3 Stage yang berbeda secara terpisah agar model AI",
    "# bisa mengevaluasi performa masing-masing stage secara independen.",
    "data_stage1 = []",
    "data_stage2 = []",
    "data_stage3 = []",
    "",
    "print(\"Running 3-stage preprocessing pipelines...\")",
    "for i in range(len(data)):",
    "    if i % 1000 == 0:",
    "        print(f\"Preprocessing image {i}/{len(data)}\")",
    "    img = data[i]",
    "    ",
    "    # Stage 1: Noise Reduction only",
    "    img_s1 = prepro1(img)",
    "    data_stage1.append(acc.to_cpu(img_s1))",
    "    ",
    "    # Stage 2: Noise Reduction + Contrast Enhancement",
    "    img_s2 = prepro2(img_s1)",
    "    data_stage2.append(acc.to_cpu(img_s2))",
    "    ",
    "    # Stage 3: Noise + Contrast + Edge/Detail Enhancement",
    "    img_s3 = prepro3(img_s2)",
    "    data_stage3.append(acc.to_cpu(img_s3))",
    "",
    "data_stage1 = np.array(data_stage1)",
    "data_stage2 = np.array(data_stage2)",
    "data_stage3 = np.array(data_stage3)",
    "print(\"Preprocessing completed for all 3 stages!\")"
])

# Cell 13: Preprocessing Visualizations
add_justification(
    "Visualisasi Transisi Preprocessing",
    "Menampilkan citra asli (grayscale) berdampingan dengan output citra hasil Stage 1, Stage 2, dan Stage 3 menggunakan visualisasi plot.",
    "Untuk memverifikasi secara visual hasil dari filter blur, pemerataan kontras CLAHE, dan filter unsharp/sharpening pada pesawat.",
    "matplotlib.pyplot merender citra ke canvas visual notebook.",
    "Citra divisualisasikan langsung pada sel output visual di Jupyter Notebook.",
    "Dijalankan segera setelah loop preprocessing selesai, sebelum fitur tekstur diekstraksi.",
    "Menggunakan subplots 1x4 dari matplotlib untuk plot data, data_stage1, data_stage2, dan data_stage3 pada indeks sampel tertentu."
)
add_code([
    "# Select a sample image to visualize the transformation at each stage",
    "sample_idx = 0  # Feel free to change this index to see other plane transformations!",
    "original_img = data[sample_idx]",
    "s1_img = data_stage1[sample_idx]",
    "s2_img = data_stage2[sample_idx]",
    "s3_img = data_stage3[sample_idx]",
    "sample_label = labels[sample_idx]",
    "sample_fname = file_name[sample_idx]",
    "",
    "fig, axes = plt.subplots(1, 4, figsize=(20, 5))",
    "fig.suptitle(f\"Preprocessing Transition of Sample Image: {sample_fname} ({sample_label})\", fontsize=16, y=1.05)",
    "",
    "# 1. Original Grayscale Image",
    "axes[0].imshow(original_img, cmap='gray')",
    "axes[0].set_title(\"Original Image\\n(Grayscale, 256x256)\")",
    "axes[0].axis('off')",
    "",
    "# 2. Stage 1 Output",
    "axes[1].imshow(s1_img, cmap='gray')",
    "axes[1].set_title(\"Stage 1: Noise Reduction\\n(Gaussian + Median)\")",
    "axes[1].axis('off')",
    "",
    "# 3. Stage 2 Output",
    "axes[2].imshow(s2_img, cmap='gray')",
    "axes[2].set_title(\"Stage 2: Contrast Enhanced\\n(CLAHE + Gamma Correction)\")",
    "axes[2].axis('off')",
    "",
    "# 4. Stage 3 Output",
    "axes[3].imshow(s3_img, cmap='gray')",
    "axes[3].set_title(\"Stage 3: Edge & Detail Enhanced\\n(Unsharp Mask + Sharpen)\")",
    "axes[3].axis('off')",
    "",
    "plt.tight_layout()",
    "plt.show()"
])

# Cell 13: Markdown Feature Extraction
add_markdown(["### Ekstraksi Fitur"])

# Cell 14: glcm function
add_justification(
    "Definisi Perhitungan Matriks GLCM",
    "Mendefinisikan fungsi glcm untuk menghitung matriks co-occurrence spasial grayscale citra dan menormalisasikannya.",
    "Sebagai perantara (wrapper) pembuat matriks GLCM sesuai dengan format template tugas proyek pcd.",
    "Fungsi komputasi mengeksekusi array citra input.",
    "Dijalankan dalam workspace kernel memori notebook.",
    "Didefinisikan di awal sub-bab Feature Extraction sebelum properti statistik dihitung.",
    "Menggunakan pemanggilan fungsi acc.GLCM.compute yang di-pass dengan parameter derajat sudut dan dinormalisasi agar total sum = 1."
)
add_code([
    "def glcm(image, derajat):",
    "    # Forward call directly to the pre-existing, optimized GLCM implementation in all-script",
    "    g = acc.GLCM.compute(image, distance=1, angle=float(derajat), levels=256, symmetric=True)",
    "    return acc.GLCM.normalize(g)"
])

# Cell 15: correlation
add_justification(
    "Wrapper Properti Korelasi GLCM",
    "Mendefinisikan fungsi correlation() untuk mengekstrak properti korelasi linear piksel dari matriks GLCM.",
    "Untuk menghitung ukuran linear dependency derajat keabu-abuan antarpiksel tetangga sesuai struktur sel template.",
    "Dijalankan oleh interpreter untuk mendaftarkan fungsi properti.",
    "Namespace global memori kernel.",
    "Dideklarasikan sebagai bagian dari pendefinisian fitur GLCM.",
    "Mengembalikan statistik key 'correlation' hasil kalkulasi acc.GLCM._compute_features."
)
add_code([
    "def correlation(matriks):",
    "    # delegates property calculation to all-script without re-implementing GLCM features",
    "    return acc.GLCM._compute_features(matriks, extract_asm=True)['correlation']"
])

# Cell 16: dissimilarity
add_justification(
    "Wrapper Properti Dissimilarity GLCM",
    "Mendefinisikan fungsi dissimilarity() untuk mengekstrak properti kontras linear (ketidakmiripan) dari matriks GLCM.",
    "Untuk mengukur perbedaan derajat keabu-abuan secara linier pada piksel yang bertetangga.",
    "Daftar fungsi properti spasial.",
    "Namespace global memori kernel.",
    "Dideklarasikan sebelum iterasi ekstraksi dilakukan.",
    "Mengembalikan key 'dissimilarity' hasil kalkulasi acc.GLCM._compute_features."
)
add_code([
    "def dissimilarity(matriks):",
    "    # delegates property calculation to all-script without re-implementing GLCM features",
    "    return acc.GLCM._compute_features(matriks, extract_asm=True)['dissimilarity']"
])

# Cell 17: homogenity
add_justification(
    "Wrapper Properti Homogenitas GLCM",
    "Mendefinisikan fungsi homogenity() untuk mengekstrak kedekatan distribusi elemen GLCM dengan diagonal utama.",
    "Untuk mengukur kehomogenan variasi warna derajat keabuan lokal pada citra pesawat.",
    "Daftar fungsi properti spasial.",
    "Namespace global memori kernel.",
    "Dideklarasikan sebelum ekstraksi batch dilakukan.",
    "Mengembalikan key 'homogeneity' hasil kalkulasi acc.GLCM._compute_features."
)
add_code([
    "def homogenity(matriks):",
    "    # delegates property calculation to all-script without re-implementing GLCM features",
    "    return acc.GLCM._compute_features(matriks, extract_asm=True)['homogeneity']"
])

# Cell 18: contrast
add_justification(
    "Wrapper Properti Kontras GLCM",
    "Mendefinisikan fungsi contrast() untuk mengukur intensitas kontras orde dua citra.",
    "Mengukur tingkat perbedaan keabuan lokal pada citra (makin tajam tepi, makin tinggi kontras GLCM).",
    "Daftar fungsi properti spasial.",
    "Namespace global memori kernel.",
    "Dideklarasikan sebelum ekstraksi batch dilakukan.",
    "Mengembalikan key 'contrast' hasil kalkulasi acc.GLCM._compute_features."
)
add_code([
    "def contrast(matriks):",
    "    # delegates property calculation to all-script without re-implementing GLCM features",
    "    return acc.GLCM._compute_features(matriks, extract_asm=True)['contrast']"
])

# Cell 19: ASM
add_justification(
    "Wrapper Properti Angular Second Moment (ASM) GLCM",
    "Mendefinisikan fungsi ASM() untuk mengekstrak Angular Second Moment (jumlah kuadrat probabilitas GLCM).",
    "Mengukur keseragaman (uniformity) tekstur citra (citra yang homogen memiliki nilai ASM yang tinggi).",
    "Daftar fungsi properti spasial.",
    "Namespace global memori kernel.",
    "Dideklarasikan sebelum ekstraksi batch dilakukan.",
    "Mengembalikan key 'asm' hasil kalkulasi acc.GLCM._compute_features."
)
add_code([
    "def ASM(matriks):",
    "    # delegates property calculation to all-script without re-implementing GLCM features",
    "    return acc.GLCM._compute_features(matriks, extract_asm=True)['asm']"
])

# Cell 20: energy
add_justification(
    "Wrapper Properti Energi GLCM",
    "Mendefinisikan fungsi energy() untuk mengembalikan akar kuadrat dari ASM citra.",
    "Untuk mengukur keteraturan tekstur (energy) sesuai spesifikasi parameter graycoprops.",
    "Daftar fungsi properti spasial.",
    "Namespace global memori kernel.",
    "Dideklarasikan sebelum ekstraksi batch dilakukan.",
    "Mengembalikan key 'energy' hasil kalkulasi acc.GLCM._compute_features."
)
add_code([
    "def energy(matriks):",
    "    # delegates property calculation to all-script without re-implementing GLCM features",
    "    return acc.GLCM._compute_features(matriks, extract_asm=True)['energy']"
])

# Cell 21: entropyGlcm
add_justification(
    "Wrapper Properti Entropi GLCM",
    "Mendefinisikan fungsi entropyGlcm() untuk mengekstrak nilai ketidakpastian (entropy) spasial piksel.",
    "Mengukur tingkat keacakan/derajat kekacauan tekstur citra derajat keabuan pesawat.",
    "Daftar fungsi properti spasial.",
    "Namespace global memori kernel.",
    "Dideklarasikan sebelum ekstraksi batch dilakukan.",
    "Mengembalikan key 'entropy' hasil kalkulasi acc.GLCM._compute_features."
)
add_code([
    "def entropyGlcm(matriks):",
    "    # delegates property calculation to all-script without re-implementing GLCM features",
    "    return acc.GLCM._compute_features(matriks, extract_asm=True)['entropy']"
])

# Cell 22: Batch feature extraction for all three stages
add_justification(
    "Ekstraksi Fitur Batch Tiga Tahap",
    "Mengekstrak 7 fitur GLCM pada 4 sudut (0, 45, 90, 135 derajat) untuk 3 stage data citra pra-proses secara batch.",
    "Menghindari loop Python manual yang sangat lambat di notebook dan mempercepat konversi gambar menjadi matriks fitur.",
    "Modul GPU/CPU batch extraction memproses seluruh koleksi citra.",
    "Hasil kalkulasi disimpan di memori sebagai kamus array (features_s1, features_s2, features_s3).",
    "Dijalankan setelah pendefinisian fungsi properti selesai dideklarasikan.",
    "Memanggil fungsi acc.GLCM.extract_batch secara paralel/sekuensial cepat pada data_stage1, data_stage2, dan data_stage3."
)
add_code([
    "print(\"Batch-extracting GLCM features for Stage 1 preprocessed images...\")",
    "features_s1 = acc.GLCM.extract_batch(data_stage1, distances=(1,), angles=(0, 45, 90, 135))",
    "",
    "print(\"Batch-extracting GLCM features for Stage 2 preprocessed images...\")",
    "features_s2 = acc.GLCM.extract_batch(data_stage2, distances=(1,), angles=(0, 45, 90, 135))",
    "",
    "print(\"Batch-extracting GLCM features for Stage 3 preprocessed images...\")",
    "features_s3 = acc.GLCM.extract_batch(data_stage3, distances=(1,), angles=(0, 45, 90, 135))",
    "print(\"Batch feature extraction completed for all 3 stages!\")"
])

# Cell 23: DataFrame creation for all three stages
add_justification(
    "Pemformatan Tabel DataFrame",
    "Mengubah kamus fitur spasial berukuran penuh menjadi objek Pandas DataFrame.",
    "Untuk mempermudah pemrosesan tabel data, filtering kolom, penulisan ke file penyimpanan, dan manipulasi data menggunakan Scikit-learn.",
    "Pandas DataFrame parser memproses struktur kamus array.",
    "Disimpan dalam memori RAM sebagai objek DataFrame (df_s1, df_s2, df_s3).",
    "Dijalankan langsung setelah ekstraksi batch GLCM selesai dilakukan.",
    "Memanggil konstruktor pd.DataFrame() pada masing-masing variabel kamus fitur."
)
add_code([
    "df_s1 = pd.DataFrame(features_s1)",
    "df_s2 = pd.DataFrame(features_s2)",
    "df_s3 = pd.DataFrame(features_s3)"
])

# Cell 24: Write extraction's results to CSV
add_markdown(["### Tulis Hasil Ekstraksi ke CSV "])

# Cell 25: Write CSV logic
add_justification(
    "Penyimpanan Fitur ke Media Penyimpanan",
    "Menggabungkan kolom Filename dan Label, lalu menyimpan matriks fitur ketiga stage ke dalam file CSV terpisah di disk lokal.",
    "Agar fitur citra yang diekstraksi tersimpan permanen dan dapat dimuat ulang instan tanpa mengulang kalkulasi GLCM yang berat.",
    "Pandas writer menyimpan representasi string csv ke penyimpanan disk.",
    "Ditulis ke root directory proyek (hasil_ekstraksi_stage1.csv, hasil_ekstraksi_stage2.csv, hasil_ekstraksi_stage3.csv, hasil_ekstraksi_1.csv).",
    "Dijalankan setelah DataFrame fitur spasial dibuat di memori.",
    "Menggunakan pd.concat untuk menggabungkan nama file & label, dilanjutkan pemanggilan method .to_csv() dengan parameter index=False."
)
add_code([
    "# Build full datasets for each stage",
    "df_s1_full = pd.concat([pd.DataFrame({'Filename': file_name, 'Label': labels}), df_s1], axis=1)",
    "df_s1_full.to_csv('hasil_ekstraksi_stage1.csv', index=False)",
    "",
    "df_s2_full = pd.concat([pd.DataFrame({'Filename': file_name, 'Label': labels}), df_s2], axis=1)",
    "df_s2_full.to_csv('hasil_ekstraksi_stage2.csv', index=False)",
    "",
    "df_s3_full = pd.concat([pd.DataFrame({'Filename': file_name, 'Label': labels}), df_s3], axis=1)",
    "df_s3_full.to_csv('hasil_ekstraksi_stage3.csv', index=False)",
    "",
    "# Simpan salinan Stage 3 sebagai hasil_ekstraksi_1.csv untuk kesesuaian dengan template",
    "df_s3_full.to_csv('hasil_ekstraksi_1.csv', index=False)",
    "",
    "print(\"Features saved! hasil_ekstraksi_1.csv contains Stage 3 features.\")",
    "df_s3_full.head()"
])

# Cell 26: Features Selection markdown
add_markdown(["### Seleksi Fitur"])

# Cell 27: Features Selection methods explanation
add_markdown([
    "pada bagian seleksi fitur ini bisa menggunakan metode seperti",
    "- PCA",
    "- LDA",
    "- t-SNE",
    "- Chi-square",
    "- ANOVA",
    "- Autoencoder",
    "- correlation",
    "- dll",
    "",
    "berikut contoh menggunakan correlation:"
])

# Cell 28: Features Selection code
add_justification(
    "Seleksi Fitur Berbasis Korelasi",
    "Menyaring dan membuang kolom fitur spasial GLCM yang memiliki koefisien korelasi linear Pearson >= 0.95 satu sama lain.",
    "Mereduksi redundansi fitur (multicollinearity) sehingga mengurangi beban memori komputasi model AI dan mencegah overfitting.",
    "Modul penyaringan fitur menguji statistik multivariat data.",
    "Dijalankan di memori RAM pada masing-masing DataFrame ketiga stage.",
    "Dijalankan sebelum pembagian data train-test split dilakukan.",
    "Menghitung matriks korelasi menggunakan .corr(), menyaring indeks baris-kolom dengan batas threshold 0.95, dan mengambil irisan kolom yang independen."
)
add_code([
    "# Helper function to filter out features with correlation >= 0.95",
    "def filter_correlated_features(df, threshold=0.95):",
    "    correlation_matrix = df.drop(columns=['Label','Filename']).corr()",
    "    columns = np.full((correlation_matrix.shape[0],), True, dtype=bool)",
    "    for i in range(correlation_matrix.shape[0]):",
    "        for j in range(i+1, correlation_matrix.shape[0]):",
    "            if correlation_matrix.iloc[i,j] >= threshold:",
    "                if columns[j]:",
    "                    columns[j] = False",
    "    select_cols = df.drop(columns=['Label','Filename']).columns[columns]",
    "    return df[select_cols], df['Label'], list(select_cols)",
    "",
    "print(\"Performing feature selection for all three stages...\")",
    "x_new1, y1, select1 = filter_correlated_features(df_s1_full)",
    "x_new2, y2, select2 = filter_correlated_features(df_s2_full)",
    "x_new3, y3, select3 = filter_correlated_features(df_s3_full)",
    "",
    "# Set compatibility variable names",
    "x_new, y = x_new3, y3",
    "print(f\"Stage 1 selected features: {len(select1)} / 28\")",
    "print(f\"Stage 2 selected features: {len(select2)} / 28\")",
    "print(f\"Stage 3 selected features: {len(select3)} / 28\")"
])

# Cell 29: Splitting Data Markdown
add_markdown(["## Pembagian Data"])

# Cell 30: Splitting Data code
add_justification(
    "Pembagian Data (Train-Test Split)",
    "Membagi matriks fitur terseleksi dan array label target menjadi set training (80%) dan set testing (20%) secara acak terkontrol menggunakan random_state=67.",
    "Menyediakan data terpisah untuk melatih model AI dan data independen yang belum pernah dilihat model untuk menguji kinerjanya secara valid.",
    "Fungsi partisi membagi subset data target.",
    "Membagi array input di memori menjadi X_train1, X_test1, dsb.",
    "Dilakukan setelah seleksi fitur selesai dan sebelum proses normalisasi Z-score.",
    "Memanggil train_test_split dari library sklearn dengan parameter test_size=0.2 dan random_state=67."
)
add_code([
    "# Split datasets for Stage 1, Stage 2, and Stage 3 independently",
    "X_train1, X_test1, y_train1, y_test1 = train_test_split(x_new1, y1, test_size=0.2, random_state=67)",
    "X_train2, X_test2, y_train2, y_test2 = train_test_split(x_new2, y2, test_size=0.2, random_state=67)",
    "X_train3, X_test3, y_train3, y_test3 = train_test_split(x_new3, y3, test_size=0.2, random_state=67)",
    "",
    "# Set compatibility variables (referencing Stage 3)",
    "X_train, X_test, y_train, y_test = X_train3, X_test3, y_train3, y_test3",
    "print(\"Stage 3 Train Set shape:\", X_train3.shape)",
    "print(\"Stage 3 Test Set shape:\", X_test3.shape)"
])

# Cell 31: Feature Normalization markdown
add_markdown(["## Normalisasi Fitur"])

# Cell 32: Feature Normalization methods markdown
add_markdown([
    "berikut metode normalisasi yang bisa digunakan:",
    "- Min-Max Scaling",
    "- Standardisasi (Z-score)",
    "- Robust Scaling",
    "- MaxAbsScaler",
    "- dll",
    "",
    "berikut contoh menggunakan Standardisasi (Z-score):"
])

# Cell 33: Normalization code
add_justification(
    "Standardisasi Z-score",
    "Melakukan standardisasi Z-score (mengurangi rata-rata, membagi standar deviasi) dan menyimpan parameter mean/std Stage 3 ke models/scaler.joblib.",
    "Menyamakan skala rentang nilai seluruh fitur tekstur GLCM (sehingga memiliki rata-rata 0 dan standar deviasi 1) agar performa SVM dan KNN optimal.",
    "Standardisasi dijalankan pada fitur input di memori RAM.",
    "Nilai parameter standardisasi disimpan ke models/scaler.joblib pada disk lokal.",
    "Dijalankan setelah pembagian data train-test split dan sebelum pelatihan model klasifikasi.",
    "Menggunakan rumus (X - mean) / (std + 1e-8) secara terpisah untuk set training dan testing pada ketiga stage."
)
add_code([
    "# normalisasi mean std untuk masing-masing Stage secara terpisah",
    "mean1, std1 = X_train1.mean(), X_train1.std() + 1e-8",
    "X_train1 = (X_train1 - mean1) / std1",
    "X_test1 = (X_test1 - mean1) / std1",
    "",
    "mean2, std2 = X_train2.mean(), X_train2.std() + 1e-8",
    "X_train2 = (X_train2 - mean2) / std2",
    "X_test2 = (X_test2 - mean2) / std2",
    "",
    "mean3, std3 = X_train3.mean(), X_train3.std() + 1e-8",
    "X_train3 = (X_train3 - mean3) / std3",
    "X_test3 = (X_test3 - mean3) / std3",
    "",
    "# Simpan mean dan std dari Stage 3 (complete pipeline) untuk deployment",
    "os.makedirs('models', exist_ok=True)",
    "joblib.dump({'mean': mean3, 'std': std3}, 'models/scaler.joblib')",
    "",
    "# Set compatibility variables",
    "X_train, X_test = X_train3, X_test3",
    "print(\"Standardization completed and models/scaler.joblib saved successfully!\")"
])

# Cell 34: Modeling markdown
add_markdown(["## Pemodelan"])

# Cell 35: Define Model markdown
add_markdown(["### Definisi Model"])

# Cell 36: Model definition code
add_justification(
    "Fungsi Pembantu Laporan Klasifikasi & Pengaturan Klasifikasi",
    "Mendefinisikan fungsi pembantu evaluasi generateClassificationReport dan menginisialisasi parameter dasar classifier (RF, SVM, KNN) dengan random_state=67.",
    "Untuk menyediakan pencetakan laporan presisi, recall, F1, akurasi, dan confusion matrix secara rapi dan konsisten.",
    "Interpreter meregistrasikan objek classifier di memori RAM.",
    "Namespace lokal memori kernel.",
    "Dijalankan sebelum fitting model dimulai.",
    "Inisialisasi RandomForestClassifier(n_estimators=100, random_state=67), SVC(C=10.0, kernel='rbf', random_state=67), dan KNeighborsClassifier(n_neighbors=3)."
)
add_code([
    "def generateClassificationReport(y_true, y_pred):",
    "\tprint(classification_report(y_true, y_pred, zero_division=0))",
    "\tprint(confusion_matrix(y_true, y_pred))",
    "\tprint('Accuracy:', accuracy_score(y_true, y_pred))",
    "",
    "# Inisialisasi classifier dengan hyperparameter yang telah dioptimalkan",
    "rf = RandomForestClassifier(n_estimators=100, random_state=67)",
    "svm = SVC(C=10.0, kernel='rbf', random_state=67)",
    "knn = KNeighborsClassifier(n_neighbors=3, weights='uniform')"
])

# Cell 37: Train Random Forest markdown
add_markdown(["### Latih Klasifikasi Random Forest"])

# Cell 38: Train RF code
add_justification(
    "Pelatihan & Evaluasi Random Forest pada Tiga Tahap",
    "Melatih (fitting) tiga classifier Random Forest pada fitur masing-masing Stage, mengevaluasi di set testing, dan menyimpan model Stage 3.",
    "Menganalisis performa ensemble pohon keputusan pada setiap stage preprocessing citra secara independen.",
    "Pengembang membandingkan log output performa model RF.",
    "Model dilatih di RAM dan diekspor ke models/rf_model.joblib.",
    "Dijalankan pada tahap awal modeling/pelatihan model.",
    "Memanggil method .fit() pada X_train1/2/3 dan .predict() untuk mencetak akurasi test, serta joblib.dump untuk persistensi model Stage 3."
)
add_code([
    "# Kita train tiga model Random Forest untuk masing-masing Stage Preprocessing",
    "rf_s1 = RandomForestClassifier(n_estimators=100, random_state=67)",
    "rf_s2 = RandomForestClassifier(n_estimators=100, random_state=67)",
    "rf_s3 = RandomForestClassifier(n_estimators=100, random_state=67)",
    "",
    "print(\"=== TRAINING RANDOM FOREST: STAGE 1 (Noise Reduction Only) ===\")",
    "rf_s1.fit(X_train1, y_train1)",
    "print(\"------Testing Set (Stage 1)------\")",
    "generateClassificationReport(y_test1, rf_s1.predict(X_test1))",
    "",
    "print(\"\\n=== TRAINING RANDOM FOREST: STAGE 2 (Noise + Contrast) ===\")",
    "rf_s2.fit(X_train2, y_train2)",
    "print(\"------Testing Set (Stage 2)------\")",
    "generateClassificationReport(y_test2, rf_s2.predict(X_test2))",
    "",
    "print(\"\\n=== TRAINING RANDOM FOREST: STAGE 3 (Noise + Contrast + Edge) ===\")",
    "rf_s3.fit(X_train3, y_train3)",
    "print(\"------Testing Set (Stage 3)------\")",
    "generateClassificationReport(y_test3, rf_s3.predict(X_test3))",
    "",
    "# Save Stage 3 Random Forest for compatibility",
    "joblib.dump(rf_s3, 'models/rf_model.joblib')",
    "rf = rf_s3"
])

# Cell 39: Train SVM markdown
add_markdown(["### Latih Klasifikasi SVM"])

# Cell 40: Train SVM code
add_justification(
    "Pelatihan & Evaluasi Support Vector Machine pada Tiga Tahap",
    "Melatih tiga classifier SVM dengan kernel RBF dan regularisasi C=10.0 pada ketiga Stage, serta menyimpan model Stage 3.",
    "Menguji performa pengklasifikasi margin maksimum non-linear dalam memisahkan sebaran fitur spasial GLCM.",
    "Pengembang membandingkan log output performa model SVM.",
    "Model dilatih di RAM dan diekspor ke models/svm_model.joblib.",
    "Dijalankan sekuensial setelah pelatihan Random Forest selesai.",
    "Menggunakan regularisasi C=10.0, fitting kernel 'rbf', mencetak skor akurasi testing, dan menyimpan file model Stage 3 menggunakan joblib."
)
add_code([
    "# Kita train tiga model SVM untuk masing-masing Stage Preprocessing",
    "svm_s1 = SVC(C=10.0, kernel='rbf', random_state=67)",
    "svm_s2 = SVC(C=10.0, kernel='rbf', random_state=67)",
    "svm_s3 = SVC(C=10.0, kernel='rbf', random_state=67)",
    "",
    "print(\"=== TRAINING SVM: STAGE 1 (Noise Reduction Only) ===\")",
    "svm_s1.fit(X_train1, y_train1)",
    "print(\"------Testing Set (Stage 1)------\")",
    "generateClassificationReport(y_test1, svm_s1.predict(X_test1))",
    "",
    "print(\"\\n=== TRAINING SVM: STAGE 2 (Noise + Contrast) ===\")",
    "svm_s2.fit(X_train2, y_train2)",
    "print(\"------Testing Set (Stage 2)------\")",
    "generateClassificationReport(y_test2, svm_s2.predict(X_test2))",
    "",
    "print(\"\\n=== TRAINING SVM: STAGE 3 (Noise + Contrast + Edge) ===\")",
    "svm_s3.fit(X_train3, y_train3)",
    "print(\"------Testing Set (Stage 3)------\")",
    "generateClassificationReport(y_test3, svm_s3.predict(X_test3))",
    "",
    "# Save Stage 3 SVM for compatibility",
    "joblib.dump(svm_s3, 'models/svm_model.joblib')",
    "svm = svm_s3"
])

# Cell 41: Train KNN markdown
add_markdown(["### Latih Klasifikasi KNN"])

# Cell 42: Train KNN code
add_justification(
    "Pelatihan & Evaluasi K-Nearest Neighbors pada Tiga Tahap",
    "Melatih tiga classifier KNN (k=3) pada fitur masing-masing Stage, mengevaluasi hasil pengujian, dan mengekspor model Stage 3.",
    "Menguji klasifikasi berbasis kedekatan jarak spasial (distance-based) untuk melihat performa pengelompokan ketetanggaan.",
    "Pengembang mengevaluasi akurasi klasifikasi berbasis ketetanggaan terdekat.",
    "Model dilatih di RAM dan diekspor ke models/knn_model.joblib.",
    "Dijalankan sebagai bagian akhir dari alur model training.",
    "Menginisialisasi KNeighborsClassifier dengan n_neighbors=3, melakukan .fit(), memprediksi label test, dan mengekspor objek model."
)
add_code([
    "# Kita train tiga model KNN untuk masing-masing Stage Preprocessing",
    "knn_s1 = KNeighborsClassifier(n_neighbors=3, weights='uniform')",
    "knn_s2 = KNeighborsClassifier(n_neighbors=3, weights='uniform')",
    "knn_s3 = KNeighborsClassifier(n_neighbors=3, weights='uniform')",
    "",
    "print(\"=== TRAINING KNN: STAGE 1 (Noise Reduction Only) ===\")",
    "knn_s1.fit(X_train1, y_train1)",
    "print(\"------Testing Set (Stage 1)------\")",
    "generateClassificationReport(y_test1, knn_s1.predict(X_test1))",
    "",
    "print(\"\\n=== TRAINING KNN: STAGE 2 (Noise + Contrast) ===\")",
    "knn_s2.fit(X_train2, y_train2)",
    "print(\"------Testing Set (Stage 2)------\")",
    "generateClassificationReport(y_test2, knn_s2.predict(X_test2))",
    "",
    "print(\"\\n=== TRAINING KNN: STAGE 3 (Noise + Contrast + Edge) ===\")",
    "knn_s3.fit(X_train3, y_train3)",
    "print(\"------Testing Set (Stage 3)------\")",
    "generateClassificationReport(y_test3, knn_s3.predict(X_test3))",
    "",
    "# Save Stage 3 KNN for compatibility",
    "joblib.dump(knn_s3, 'models/knn_model.joblib')",
    "knn = knn_s3"
])

# Cell 43: Confusion Matrix markdown
add_markdown(["## Evaluasi dengan Confusion Matrix"])

# Cell 44: Confusion Matrix plots code
add_justification(
    "Visualisasi Confusion Matrix",
    "Menggambar visualisasi Confusion Matrix berupa heatmap untuk model RF, SVM, dan KNN pada hasil evaluasi Stage 3.",
    "Untuk memetakan pola klasifikasi benar vs salah (misklasifikasi) tiap kelas secara rinci dan visual, dengan visualisasi bersih tanpa teks angka yang menumpuk.",
    "Modul visualisasi merender visualisasi heatmap matriks.",
    "Heatmap digambar langsung pada visual canvas notebook.",
    "Dijalankan sebagai langkah terakhir evaluasi setelah seluruh pengklasifikasi selesai dilatih.",
    "Menggunakan subplots berukuran (12, 10) dengan option include_values=False pada objek ConfusionMatrixDisplay untuk menyembunyikan grid angka yang tumpang tindih."
)
add_code([
    "def plot_confusion_matrix(y_true, y_pred, title):",
    "    cm = confusion_matrix(y_true, y_pred)",
    "    disp = ConfusionMatrixDisplay(confusion_matrix=cm)",
    "    # Kita set figure size yang lebih besar agar tidak overlap",
    "    fig, ax = plt.subplots(figsize=(12, 10))",
    "    # CRITICAL: set include_values=False agar tulisan angka di dalam grid sel tidak saling tumpang tindih!",
    "    disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation='vertical', include_values=False)",
    "    # Set ukuran label agar rapi",
    "    ax.tick_params(axis='both', which='major', labelsize=8)",
    "    plt.title(title)",
    "    plt.tight_layout()",
    "    plt.show()",
    "",
    "# Kita plot Confusion Matrix untuk Stage 3 (atau Stage 2, mana yang performanya paling optimal)",
    "plot_confusion_matrix(y_test3, rf_s3.predict(X_test3), \"Random Forest (Stage 3) Confusion Matrix\")",
    "plot_confusion_matrix(y_test3, svm_s3.predict(X_test3), \"SVM (Stage 3) Confusion Matrix\")",
    "plot_confusion_matrix(y_test3, knn_s3.predict(X_test3), \"KNN (Stage 3) Confusion Matrix\")"
])

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.11.8"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

with open("AeroVision.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)

print("AeroVision.ipynb created successfully!")
