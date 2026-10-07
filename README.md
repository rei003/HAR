# Human Activity Recognition using Raw IMU Data

A machine learning project for classifying human activities using raw smartphone accelerometer and gyroscope data.

Instead of using the dataset's pre-built 561-feature representation, this project starts from the raw sensor signals and builds a small custom feature representation using simple statistical features.

---

## Overview

The goal of this project is to classify human activities from smartphone-based inertial measurement unit (IMU) data.

The raw dataset contains:

- Accelerometer data — X, Y, Z axes
- Gyroscope data — X, Y, Z axes
- Sampling frequency: **50 Hz**

The raw sensor streams are divided into fixed-size windows. Statistical features are then extracted from each window and used to train and compare several machine learning models.

The complete pipeline is:

```text
Raw Accelerometer + Gyroscope Data
                ↓
        128-sample Windows
                ↓
          Window Labeling
                ↓
      Custom Feature Extraction
                ↓
        Exploratory Analysis
                ↓
         Train / Test Split
                ↓
          Feature Scaling
                ↓
         Multiple ML Models
                ↓
          Model Comparison
                ↓
       Feature Engineering Experiment
                ↓
        Final SVM Classifier
```

---

## Dataset

This project uses the **UCI Human Activity Recognition Using Smartphones Dataset**.

The raw sensor signals are sampled at **50 Hz** and contain three-axis accelerometer and gyroscope measurements.

The project works directly with the raw sensor recordings rather than using the dataset's pre-computed feature vectors.

### Activities

The original dataset contains 12 activity labels:

| Label | Activity |
|------:|----------|
| 1 | WALKING |
| 2 | WALKING_UPSTAIRS |
| 3 | WALKING_DOWNSTAIRS |
| 4 | SITTING |
| 5 | STANDING |
| 6 | LAYING |
| 7 | STAND_TO_SIT |
| 8 | SIT_TO_STAND |
| 9 | SIT_TO_LIE |
| 10 | LIE_TO_SIT |
| 11 | STAND_TO_LIE |
| 12 | LIE_TO_STAND |

The transition activities (7–12) contained very few usable windows compared with the six main activities. They were therefore removed from the final classification dataset.

The final dataset contains:

- **5,000 windows**
- **6 activity classes**
- **18 input features**

---

## Data Processing

### 1. Windowing the Raw Sensor Data

Each accelerometer and gyroscope recording is a continuous sequence of X/Y/Z measurements.

The signals are divided into windows of:

```text
128 samples
```

At 50 Hz, this corresponds to:

```text
128 / 50 = 2.56 seconds
```

The windows are non-overlapping.

Any remaining samples that do not form a complete 128-sample window are discarded.

### 2. Window Labeling

The activity labels are provided as intervals in the original label file.

For each 128-sample window, the project checks whether exactly one labeled activity interval completely contains that window.

Windows that:

- cross an activity transition, or
- do not belong to a clearly defined activity interval

are not assigned an activity label.

This prevents transition windows from being incorrectly assigned to one of the neighboring activities.

---

## Feature Engineering

Rather than using the original 561 engineered features from the dataset, this project creates a custom representation from the raw signals.

For each sensor axis, three statistical features are calculated:

- Mean
- Standard deviation
- Maximum

These are calculated separately for X, Y and Z.

### Accelerometer

```text
a_mean_x
a_mean_y
a_mean_z

a_std_x
a_std_y
a_std_z

a_max_x
a_max_y
a_max_z
```

### Gyroscope

```text
g_mean_x
g_mean_y
g_mean_z

g_std_x
g_std_y
g_std_z

g_max_x
g_max_y
g_max_z
```

This gives:

**9 accelerometer features + 9 gyroscope features = 18 features**

### Why these features?

- **Mean** captures the typical signal level within the window.
- **Standard deviation** captures how much the signal varies.
- **Maximum** captures the largest positive value in the window.
- Keeping X/Y/Z separate preserves directional information.
- Accelerometer and gyroscope measurements capture different aspects of movement.

The result is one row per sensor window.

---

## Exploratory Data Analysis

Several checks were performed before model training.

### Dataset Structure

The generated feature dataset contains numeric floating-point features and no missing values after removing unlabeled windows.

### Class Distribution

The six main activities contained substantially more samples than the transition activities.

This was the reason for removing classes 7–12 before the final classification experiment.

### Feature Distributions

Feature distributions showed differences between static activities such as:

- SITTING
- STANDING
- LAYING

and dynamic activities such as:

- WALKING
- WALKING_UPSTAIRS
- WALKING_DOWNSTAIRS

### Outliers

Some features contained unusually large values.

These were checked against the underlying samples rather than being automatically removed. The inspected values corresponded to genuine movement signals, so they were retained.

### Correlation

A correlation heatmap was used to inspect relationships between the engineered features.

Some gyroscope standard-deviation and maximum features showed relatively strong correlation with each other.

---

## Model Training

After filtering the dataset to the six main activities, the data is separated into:

```text
X = input features
y = activity label
```

The data is divided into:

```text
80% training
20% testing
```

using:

```text
random_state = 42
```

### Feature Scaling

The features are standardized using `StandardScaler`.

The scaler is fitted on the training data only and then used to transform both the training and test data.

This prevents information from the test set from being used while learning the preprocessing parameters.

The fitted scaler is saved as:

```text
scaler.pkl
```

---

## Models

Five classification algorithms were compared:

1. Logistic Regression
2. Support Vector Machine (SVM)
3. Decision Tree
4. Random Forest
5. SGD Classifier

Hyperparameters were explored using `GridSearchCV`.

---

## Model Comparison

| Model | Test Accuracy |
|---|---:|
| Logistic Regression | **91.8%** |
| SVM | **96.3%** |
| Decision Tree | **88.2%** |
| Random Forest | **95.2%** |
| SGD Classifier | **91.9%** |

The SVM produced the highest test accuracy among the models evaluated.

---

## Final Model — SVM

The final classifier is an RBF-kernel Support Vector Machine.

The selected configuration was:

```text
C = 100
kernel = RBF
gamma = scale
```

Performance:

```text
Training Accuracy: 97.275%
Test Accuracy:     96.3%
```

The trained model is saved as:

```text
svm_model.pkl
```

The corresponding fitted scaler is saved as:

```text
scaler.pkl
```

---

## Error Analysis

The SVM confusion matrix shows that the model performs very well across the six activities.

`LAYING` was classified correctly for all test samples.

The main remaining confusion occurs between:

**SITTING ↔ STANDING**

The SVM confusion matrix contained:

- 9 SITTING samples classified as STANDING
- 19 STANDING samples classified as SITTING

This suggests that the statistical features used in this project capture movement differences effectively, while distinguishing activities with similar static sensor characteristics remains more difficult.

---

## Feature Engineering Experiment

After establishing the original 18-feature representation, an additional feature-engineering experiment was performed.

### Magnitude Features

For each timestep, the magnitude of the three-axis sensor signal was calculated:

```text
magnitude = sqrt(x² + y² + z²)
```

Mean, standard deviation and maximum were then calculated from the magnitude signal.

This added six additional features:

```text
a_mag_mean
a_mag_std
a_mag_max

g_mag_mean
g_mag_std
g_mag_max
```

The resulting representation contained:

**18 original features + 6 magnitude features = 24 features**

The same SVM training procedure was used to make the comparison controlled.

### Results

| Representation | Features | Train Accuracy | Test Accuracy |
|---|---:|---:|---:|
| Original | 18 | 97.275% | **96.3%** |
| + Magnitude | 24 | 97.125% | **96.1%** |

The magnitude features did not improve performance in this experiment.

Therefore, the original 18-feature representation was retained as the final representation.

---

## Project Structure

```text
HAR/
│
├── RawData/
├── Train/
├── Test/
│
├── activity_labels.txt
├── features.txt
├── features_info.txt
│
├── main.ipynb
│
├── svm_model.pkl
├── scaler.pkl
├── svm_magnitude_model.pkl
├── scaler_mag.pkl
├── forest_model.pkl
│
├── har_tree.dot
│
└── README.md
```

### Important Files

| File | Purpose |
|---|---|
| `main.ipynb` | Complete data processing, EDA, model training and evaluation |
| `svm_model.pkl` | Saved final SVM model |
| `scaler.pkl` | Saved fitted `StandardScaler` |
| `forest_model.pkl` | Saved Random Forest model |
| `activity_labels.txt` | Activity interval labels |
| `features.txt` | Dataset feature information |
| `features_info.txt` | Dataset feature documentation |
| `RawData/` | Raw accelerometer and gyroscope recordings |
| `svm_magnitude_model.pkl` | SVM model trained with the 24-feature magnitude representation |
| `scaler_magnitude.pkl` | Fitted scaler for the 24-feature magnitude representation |

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Jupyter Notebook
- Graphviz

---

## Running the Project

Install the required Python packages:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib graphviz
```

Then open:

```text
main.ipynb
```

and run the notebook.

### Dataset Path

The notebook expects the raw dataset files inside the project's `RawData/` directory.

The notebook uses a relative path, so no machine-specific absolute path is required as long as the project structure is preserved.
---

## Saved Models

The repository contains the trained model artifacts:

```text
svm_model.pkl
scaler.pkl
svm_magnitude_model.pkl
scaler_magnitude.pkl
forest_model.pkl
```

The SVM and scaler can be loaded using:

```python
import joblib

model = joblib.load("svm_model.pkl")
scaler = joblib.load("scaler.pkl")
```

For new data, the same feature extraction process used during training must be applied before scaling and prediction.

---

## Limitations

### 1. Window-level train/test split

The train/test split is performed at the window level. Therefore, windows from the same experiment/user may occur in both training and testing sets.

The reported accuracy should therefore not be interpreted as a measurement of completely unseen-user generalization.

### 2. Handcrafted Features

The model uses only 18 statistical features per window.

This simplifies the problem but also removes much of the detailed temporal structure contained in the original sensor signals.

### 3. Transition Activities

The six transition activities were excluded because they had substantially fewer samples than the main activities.

Therefore, the final model only predicts six activity classes.

### 4. Dataset Configuration

The notebook currently uses a local dataset path, so the path needs to be adjusted when running the project on another machine.

---

## Results & Conclusion

This project demonstrates a complete machine learning workflow starting from raw smartphone IMU signals rather than relying on the dataset's pre-engineered feature vectors.

The main workflow was:

```text
Raw IMU Signals
      ↓
128-sample Windows
      ↓
Activity Labeling
      ↓
18 Custom Statistical Features
      ↓
EDA
      ↓
Train/Test Split
      ↓
Standardization
      ↓
Model Comparison
      ↓
SVM
      ↓
96.4% Test Accuracy
```

Among the evaluated models, the RBF SVM achieved a **96.3% test accuracy** using the original 18-feature representation.

A subsequent magnitude-feature experiment increased the feature count from 18 to 24, but reduced test accuracy slightly to **96.1%**. The original feature representation was therefore retained.

The project focuses on understanding the complete pipeline from **raw sensor data → feature engineering → model comparison → evaluation**, rather than simply using the pre-built feature vectors supplied with the dataset.