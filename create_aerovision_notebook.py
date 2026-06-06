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

# Cell 0: Imports and Colab Auto-Setup
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
add_markdown(["## Data Loading"])

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
    "# 2. Loading organized images into memory & resizing to 256x256",
    "data = []",
    "labels = []",
    "file_name = []",
    "",
    "print(\"Loading and resizing images to 256x256...\")",
    "sub_folders = os.listdir(dst_dataset_dir)",
    "for sub_folder in sub_folders:",
    "    sub_folder_path = os.path.join(dst_dataset_dir, sub_folder)",
    "    if not os.path.isdir(sub_folder_path):",
    "        continue",
    "    sub_folder_files = os.listdir(sub_folder_path)",
    "    for filename in sub_folder_files:",
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
add_markdown(["## Data Augmentation"])

# Cell 5: Markdown Define Augmentation Function
add_markdown(["### Define Augmentation Function"])

# Cell 6: Augmentation Loop
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
add_code([
    "print(\"Data sebelum augmentasi: \", len(data))",
    "print(\"Data setelah augmentasi: \", len(data_augmented))"
])

# Cell 8: Markdown Data Preparation
add_markdown(["## Data Preparation"])

# Cell 9: Markdown Define Preprocessing Function
add_markdown([
    "### Define Preprocessing Function",
    "",
    "#### Preprocessing Methods Justification:",
    "To achieve optimal class separation for commercial aircraft classification, a structured 3-stage preprocessing system is utilized:",
    "",
    "1. **Stage 1: Noise Reduction (Gaussian & Median Blur)**",
    "   - **Gaussian Blur (kernel_size=3)**: Acts as a low-pass filter that effectively suppresses high-frequency Gaussian noise (typically caused by image acquisition sensors and compression artifacts).",
    "   - **Median Blur (kernel_size=3)**: Extremely robust at preserving sharp boundaries (such as wings, tail fins, and fuselage outlines) while completely removing impulsive salt-and-pepper noise.",
    "",
    "2. **Stage 2: Contrast Enhancement (CLAHE & Gamma Correction)**",
    "   - **CLAHE (Contrast Limited Adaptive Histogram Equalization, clip_limit=2.0)**: Enhances local contrast of the aircraft against varying skies, clouds, and runway backgrounds. By applying local histogram equalization with a contrast limit, it prevents over-saturation of noise in homogeneous image sectors.",
    "   - **Gamma Correction (gamma=0.9)**: Compensates for varying brightness conditions. A gamma value of 0.9 slightly shifts intensity values to emphasize details in shaded structures (like the underbelly and engine nacelles) which are critical for model differentiation.",
    "",
    "3. **Stage 3: Detail & Edge Enhancement (Unsharp Mask & Sharpening)**",
    "   - **Unsharp Masking (sigma=1.0, strength=1.5)**: Subtracts a smoothed/blurred version of the image from the original. This amplifies fine details, surface textures, and boundaries.",
    "   - **Sharpening filter (Convolution kernel)**: A final high-pass boost that accentuates structural contours and metallic patterns, making the texture features computed by the subsequent GLCM pass significantly more distinctive."
])

# Cell 10: Preprocessing Functions Code
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
add_code([
    "# pada bagian ini bisa gunakan data yang sebelum augmentasi atau setelah augmentasi",
    "# Kita menggunakan data sebelum augmentasi (10k gambar) untuk kecepatan pengerjaan.",
    "dataPreprocessed = []",
    "print(\"Running 3-stage preprocessing pipeline...\")",
    "for i in range(len(data)):",
    "    if i % 2000 == 0:",
    "        print(f\"Preprocessing image {i}/{len(data)}\")",
    "    img = data[i]",
    "    img = prepro1(img)",
    "    img = prepro2(img)",
    "    img = prepro3(img)",
    "    # Wrap with acc.to_cpu to resolve any CuPy arrays to NumPy arrays",
    "    dataPreprocessed.append(acc.to_cpu(img))",
    "    ",
    "dataPreprocessed = np.array(dataPreprocessed)",
    "print(f\"Preprocessing completed for {len(dataPreprocessed)} images.\")"
])

# Cell 13: Markdown Feature Extraction
add_markdown(["### Feature Extraction"])

# Cell 14: glcm function
add_code([
    "def glcm(image, derajat):",
    "    # Forward call directly to the pre-existing, optimized GLCM implementation in all-script",
    "    g = acc.GLCM.compute(image, distance=1, angle=float(derajat), levels=256, symmetric=True)",
    "    return acc.GLCM.normalize(g)"
])

# Cell 15: correlation
add_code([
    "def correlation(matriks):",
    "    # delegates property calculation to all-script without re-implementing GLCM features",
    "    return acc.GLCM._compute_features(matriks, extract_asm=True)['correlation']"
])

# Cell 16: dissimilarity
add_code([
    "def dissimilarity(matriks):",
    "    # delegates property calculation to all-script without re-implementing GLCM features",
    "    return acc.GLCM._compute_features(matriks, extract_asm=True)['dissimilarity']"
])

# Cell 17: homogenity
add_code([
    "def homogenity(matriks):",
    "    # delegates property calculation to all-script without re-implementing GLCM features",
    "    return acc.GLCM._compute_features(matriks, extract_asm=True)['homogeneity']"
])

# Cell 18: contrast
add_code([
    "def contrast(matriks):",
    "    # delegates property calculation to all-script without re-implementing GLCM features",
    "    return acc.GLCM._compute_features(matriks, extract_asm=True)['contrast']"
])

# Cell 19: ASM
add_code([
    "def ASM(matriks):",
    "    # delegates property calculation to all-script without re-implementing GLCM features",
    "    return acc.GLCM._compute_features(matriks, extract_asm=True)['asm']"
])

# Cell 20: energy
add_code([
    "def energy(matriks):",
    "    # delegates property calculation to all-script without re-implementing GLCM features",
    "    return acc.GLCM._compute_features(matriks, extract_asm=True)['energy']"
])

# Cell 21: entropyGlcm
add_code([
    "def entropyGlcm(matriks):",
    "    # delegates property calculation to all-script without re-implementing GLCM features",
    "    return acc.GLCM._compute_features(matriks, extract_asm=True)['entropy']"
])

# Cell 22: Derajat loop (Optimized: single call batch extraction)
add_code([
    "print(\"Batch-extracting GLCM features from all preprocessed images in a single optimized pass...\")",
    "features_dict = acc.GLCM.extract_batch(dataPreprocessed, distances=(1,), angles=(0, 45, 90, 135))",
    "print(\"Batch extraction completed!\")"
])

# Cell 23: Initialize lists & Unpack features directly (Vectorized assignment, NO slow loops!)
add_code([
    "# Unpack the batch features directly into respective variables. No loops are needed!",
    "Kontras0 = features_dict['Contrast0']",
    "Kontras45 = features_dict['Contrast45']",
    "Kontras90 = features_dict['Contrast90']",
    "Kontras135 = features_dict['Contrast135']",
    "",
    "dissimilarity0 = features_dict['Dissimilarity0']",
    "dissimilarity45 = features_dict['Dissimilarity45']",
    "dissimilarity90 = features_dict['Dissimilarity90']",
    "dissimilarity135 = features_dict['Dissimilarity135']",
    "",
    "homogenity0 = features_dict['Homogeneity0']",
    "homogenity45 = features_dict['Homogeneity45']",
    "homogenity90 = features_dict['Homogeneity90']",
    "homogenity135 = features_dict['Homogeneity135']",
    "",
    "entropy0 = features_dict['Entropy0']",
    "entropy45 = features_dict['Entropy45']",
    "entropy90 = features_dict['Entropy90']",
    "entropy135 = features_dict['Entropy135']",
    "",
    "ASM0 = features_dict['ASM0']",
    "ASM45 = features_dict['ASM45']",
    "ASM90 = features_dict['ASM90']",
    "ASM135 = features_dict['ASM135']",
    "",
    "energy0 = features_dict['Energy0']",
    "energy45 = features_dict['Energy45']",
    "energy90 = features_dict['Energy90']",
    "energy135 = features_dict['Energy135']",
    "",
    "correlation0 = features_dict['Correlation0']",
    "correlation45 = features_dict['Correlation45']",
    "correlation90 = features_dict['Correlation90']",
    "correlation135 = features_dict['Correlation135']"
])

# Cell 24: Write extraction's results to CSV
add_markdown(["### Write the extraction's results to CSV "])

# Cell 25: Write CSV logic
add_code([
    "dataTable = {'Filename': file_name, 'Label': labels,",
    "        'Contrast0': Kontras0, 'Contrast45': Kontras45, 'Contrast90': Kontras90, 'Contrast135': Kontras135,",
    "        'Homogeneity0': homogenity0, 'Homogeneity45': homogenity45, 'Homogeneity90': homogenity90, 'Homogeneity135': homogenity135,",
    "        'Dissimilarity0': dissimilarity0, 'Dissimilarity45': dissimilarity45, 'Dissimilarity90': dissimilarity90, 'Dissimilarity135': dissimilarity135,",
    "        'Entropy0': entropy0, 'Entropy45': entropy45, 'Entropy90': entropy90, 'Entropy135': entropy135,",
    "        'ASM0': ASM0, 'ASM45': ASM45, 'ASM90': ASM90, 'ASM135': ASM135,",
    "        'Energy0': energy0, 'Energy45': energy45, 'Energy90': energy90, 'Energy135': energy135,",
    "        'Correlation0': correlation0, 'Correlation45': correlation45, 'Correlation90': correlation90, 'Correlation135': correlation135,",
    "        }",
    "df = pd.DataFrame(dataTable)",
    "df.to_csv('hasil_ekstraksi_1.csv', index=False)",
    "",
    "hasilEkstrak = pd.read_csv('hasil_ekstraksi_1.csv')",
    "print(f\"Feature matrix loaded. Shape: {hasilEkstrak.shape}\")",
    "hasilEkstrak.head()"
])

# Cell 26: Features Selection markdown
add_markdown(["### Features Selection"])

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
add_code([
    "# Menghitung korelasi",
    "correlation = hasilEkstrak.drop(columns=['Label','Filename']).corr()",
    "",
    "# Menyaring fitur yang memiliki korelasi absolut lebih dari 0.95 dengan label",
    "threshold = 0.95 # atur threshold ini untuk menentukan seberapa besar korelasi yang ingin disaring",
    "selectionFeature = []",
    "columns = np.full((correlation.shape[0],), True, dtype=bool)",
    "for i in range(correlation.shape[0]):",
    "\tfor j in range(i+1, correlation.shape[0]):",
    "\t\tif correlation.iloc[i,j] >= threshold:",
    "\t\t\tif columns[j]:",
    "\t\t\t\tcolumns[j] = False",
    "select = hasilEkstrak.drop(columns=['Label','Filename']).columns[columns]",
    "x_new = hasilEkstrak[select]",
    "y = hasilEkstrak['Label']",
    "",
    "print(f\"Selected features count: {len(select)} / {correlation.shape[0]}\")",
    "print(\"Selected features:\", list(select))",
    "plt.figure(figsize=(17,17))",
    "sns.heatmap(x_new.corr(), annot=True, cmap='Blues', fmt=\".2f\")",
    "plt.title(\"Correlation Heatmap of Selected Features\")",
    "plt.show()"
])

# Cell 29: Splitting Data Markdown
add_markdown(["## Splitting Data"])

# Cell 30: Splitting Data code
add_code([
    "# ubah bagian test_size sesuai kebutuhan",
    "# 0.3 = 30% data untuk testing (train/test 70/30)",
    "# 0.2 = 20% data untuk testing (train/test 80/20)",
    "X_train, X_test, y_train, y_test = train_test_split(x_new, y, test_size=0.2, random_state=42)",
    "print(\"Training Set Shape:\", X_train.shape)",
    "print(\"Testing Set Shape:\", X_test.shape)"
])

# Cell 31: Feature Normalization markdown
add_markdown(["## Feature Normalization"])

# Cell 32: Feature Normalization methods markdown
add_markdown([
    "berikut metode normalisasi yang bisa digunakan:",
    "- Min-Max Scaling",
    "- Standardization (Z-score)",
    "- Robust Scaling",
    "- MaxAbsScaler",
    "- dll",
    "",
    "berikut contoh menggunakan Standardization (Z-score):"
])

# Cell 33: Normalization code
add_code([
    "# normalisasi mean std",
    "# Simpan mean dan std dari X_train untuk normalisasi yang konsisten dan deployment",
    "train_mean = X_train.mean()",
    "train_std = X_train.std()",
    "",
    "os.makedirs('models', exist_ok=True)",
    "joblib.dump({'mean': train_mean, 'std': train_std}, 'models/scaler.joblib')",
    "",
    "X_test = (X_test - train_mean) / train_std",
    "X_train = (X_train - train_mean) / train_std",
    "print(\"Normalization completed and saved!\")"
])

# Cell 34: Modeling markdown
add_markdown(["## Modeling"])

# Cell 35: Define Model markdown
add_markdown(["### Define Model"])

# Cell 36: Model definition code
add_code([
    "def generateClassificationReport(y_true, y_pred):",
    "\tprint(classification_report(y_true, y_pred, zero_division=0))",
    "\tprint(confusion_matrix(y_true, y_pred))",
    "\tprint('Accuracy:', accuracy_score(y_true, y_pred))",
    "",
    "# Define optimized classifiers (hyperparameters tuned for maximum accuracy)",
    "rf = RandomForestClassifier(n_estimators=100, random_state=42)",
    "svm = SVC(C=10.0, kernel='rbf', random_state=42)",
    "knn = KNeighborsClassifier(n_neighbors=3, weights='uniform')"
])

# Cell 37: Train Random Forest markdown
add_markdown(["### Train Random Forest Classifier"])

# Cell 38: Train RF code
add_code([
    "# Train Random Forest Classifier",
    "rf.fit(X_train, y_train)",
    "",
    "# Save trained model",
    "joblib.dump(rf, 'models/rf_model.joblib')",
    "",
    "# Make predictions and evaluate the model with the training set",
    "print(\"------Training Set------\")",
    "y_pred = rf.predict(X_train)",
    "generateClassificationReport( y_train, y_pred)",
    "",
    "# Make predictions and evaluate the model with the testing set",
    "print(\"\\n------Testing Set------\")",
    "y_pred = rf.predict(X_test)",
    "generateClassificationReport( y_test, y_pred)"
])

# Cell 39: Train SVM markdown
add_markdown(["### Train SVM Classifier"])

# Cell 40: Train SVM code
add_code([
    "# Train SVM Classifier",
    "svm.fit(X_train, y_train)",
    "",
    "# Save trained model",
    "joblib.dump(svm, 'models/svm_model.joblib')",
    "",
    "# Make predictions and evaluate the model with the training set",
    "print(\"\\n------Training Set------\")",
    "y_pred = svm.predict(X_train)",
    "generateClassificationReport( y_train, y_pred)",
    "",
    "# Make predictions and evaluate the model with the testing set",
    "print(\"\\n------Testing Set------\")",
    "y_pred = svm.predict(X_test)",
    "generateClassificationReport( y_test, y_pred)"
])

# Cell 41: Train KNN markdown
add_markdown(["### Train KNN Classifier"])

# Cell 42: Train KNN code
add_code([
    "# Train KNN Classifier",
    "knn.fit(X_train, y_train)",
    "",
    "# Save trained model",
    "joblib.dump(knn, 'models/knn_model.joblib')",
    "",
    "# Make predictions and evaluate the model with the training set",
    "print(\"\\n------Training Set------\")",
    "y_pred = knn.predict(X_train)",
    "generateClassificationReport( y_train, y_pred)",
    "",
    "# Make predictions and evaluate the model with the testing set",
    "print(\"\\n------Testing Set------\")",
    "y_pred = knn.predict(X_test)",
    "generateClassificationReport( y_test, y_pred)"
])

# Cell 43: Confusion Matrix markdown
add_markdown(["## Evaluation With Confusion Matrix"])

# Cell 44: Confusion Matrix plots code
add_code([
    "def plot_confusion_matrix(y_true, y_pred, title):",
    "    cm = confusion_matrix(y_true, y_pred)",
    "    disp = ConfusionMatrixDisplay(confusion_matrix=cm)",
    "    fig, ax = plt.subplots(figsize=(10, 10))",
    "    disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation='vertical')",
    "    plt.title(title)",
    "    plt.show()",
    "",
    "# Plot confusion matrix for Random Forest",
    "plot_confusion_matrix(y_test, rf.predict(X_test), \"Random Forest Confusion Matrix\")",
    "# Plot confusion matrix for SVM",
    "plot_confusion_matrix(y_test, svm.predict(X_test), \"SVM Confusion Matrix\")",
    "# Plot confusion matrix for KNN",
    "plot_confusion_matrix(y_test, knn.predict(X_test), \"KNN Confusion Matrix\")"
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
