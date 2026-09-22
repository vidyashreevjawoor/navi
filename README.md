# Binary Classification: Naive Bayes vs. K-Nearest Neighbors (KNN)

An end-to-end implementation comparing **Naive Bayes (GaussianNB)** and **K-Nearest Neighbors (KNN)** algorithms on the MAGIC Gamma Telescope dataset. 

---

## 📌 Machine Learning Pipeline

1. **Data Collection & Load:** Load dataset (`magic04.data`) and assign column headers.
2. **Data Exploration:** Check dataset shape (19,020 samples × 11 columns), column data types, missing values, and descriptive statistics.
3. **Target Encoding:** Binary encode the target column (`'g'` $\rightarrow 1$, `'h'` $\rightarrow 0$).
4. **Data Splitting & Preprocessing:** Split dataset into Train (60%), Validation (20%), and Test (20%) sets[cite: 5]. Apply `StandardScaler` to normalize features and `RandomOverSampler` to balance training classes[cite: 5].
5. **Model Building & Evaluation:** Train Gaussian Naive Bayes and KNN classifiers, and compare performance using classification reports[cite: 5].

---

## 📊 Dataset Overview

- **Source:** `magic04.data` (MAGIC Gamma Telescope Dataset)[cite: 5]
- **Shape:** 19,020 rows × 11 columns[cite: 5]
- **Features ($X$):** `fLength`, `fWidth`, `fSize`, `fConc`, `fConc1`, `fAsym`, `fM3Long`, `fM3Trans`, `fAlpha`, `fDist`[cite: 5]
- **Target ($y$):** `class` (`g` = gamma/signal $\rightarrow 1$, `h` = hadron/background $\rightarrow 0$)[cite: 5]

---

## 💻 Implementation & Code Workflow

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

# 1. Load Dataset & Define Columns
columns = ['fLength', 'fWidth', 'fSize', 'fConc', 'fConc1', 'fAsym', 'fM3Long', 'fM3Trans', 'fAlpha', 'fDist', 'class']
df = pd.read_csv("magic04.data", names=columns)

# 2. Binary Encoding Target Column
df['class'] = (df['class'] == 'g').astype(int)

# 3. Train / Validation / Test Split (60% Train, 20% Valid, 20% Test)
train, valid, test = np.split(df.sample(frac=1, random_state=42), [int(0.6 * len(df)), int(0.8 * len(df))])

# 4. Feature Scaling & Oversampling Function
def scale_data(dataframe, oversample=False):
    x = dataframe[dataframe.columns[:-1]].values
    y = dataframe[dataframe.columns[-1]].values

    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    if oversample:
        ros = RandomOverSampler()
        x, y = ros.fit_resample(x, y)

    data = np.hstack((x, np.reshape(y, (-1, 1))))
    return data, x, y

# 5. Apply Preprocessing
train, X_train, y_train = scale_data(pd.DataFrame(train), oversample=True)
valid, X_valid, y_valid = scale_data(pd.DataFrame(valid), oversample=False)
test, X_test, y_test = scale_data(pd.DataFrame(test), oversample=False)

# 6. Gaussian Naive Bayes Classifier
nb = GaussianNB()
nb.fit(X_train, y_train)
nb_predict = nb.predict(X_valid)

print("--- Naive Bayes Performance (Validation Set) ---")
print(classification_report(y_valid, nb_predict))

# 7. K-Nearest Neighbors Classifier (K=3)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
knn_predict = knn.predict(X_valid)

print("--- KNN Performance (Validation Set) ---")
print(classification_report(y_valid, knn_predict))

### 📈 Performance Comparison & Model Evaluation

| Metric | Naive Bayes (GaussianNB) | K-Nearest Neighbors (KNN, K=3) |
| :--- | :---: | :---: |
| **Accuracy** | **~59%** | **~81% - 83%** |
| **Macro F1-Score** | ~0.49 | ~0.79 |
| **Weighted F1-Score**| ~0.56 | ~0.82 || Metric | Naive Bayes (GaussianNB) | K-Nearest Neighbors (KNN, K=3) |
| :--- | :---: | :---: |
| **Accuracy** | **~59%** | **~81% - 83%** |
| **Macro F1-Score** | ~0.49 | ~0.79 |
| **Weighted F1-Score**| ~0.56 | ~0.82 |
