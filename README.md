# Support Vector Machines (SVM) – Classification, Kernels & Regression

## Overview

This repository contains Jupyter Notebooks demonstrating **Support Vector Machine (SVM)** techniques for both **classification** and **regression** tasks. The notebooks focus on understanding how SVMs work, how different kernels affect decision boundaries, and how Support Vector Regression (SVR) can be used for continuous target prediction.

---

## Table of Contents

1. Installation  
2. Project Structure  
3. Support Vector Classification (SVC)  
4. SVM Kernels & Decision Boundaries  
5. Support Vector Regression (SVR)

---

## Installation

Install the required Python libraries:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

---

## Project Structure

- `SVC.ipynb` – Support Vector Classification on synthetic data  
- `SVMKernels.ipynb` – Visualization of kernel-based decision boundaries  
- `SVR.ipynb` – Support Vector Regression using a real dataset  

---

## Support Vector Classification (SVC)

### `SVC.ipynb`

This notebook demonstrates **Support Vector Classification** using synthetically generated data.

Key points:
- Uses synthetic datasets to understand class separation
- Demonstrates how SVM finds the optimal separating hyperplane
- Focuses on classification with margin maximization

Basic commands used:
```python
from sklearn.datasets import make_classification
from sklearn.svm import SVC

X, y = make_classification(n_samples=1000, n_features=2, n_classes=2)
model = SVC(kernel='linear')
model.fit(X, y)
```

---

## SVM Kernels & Decision Boundaries

### `SVMKernels.ipynb`

This notebook focuses on **kernel methods** and how they transform data into higher dimensions.

Key points:
- Visualizes non-linear class boundaries
- Demonstrates how kernels help separate complex patterns
- Uses geometric shapes to illustrate kernel behavior

Common commands used:
```python
from sklearn.svm import SVC

model = SVC(kernel='rbf')
model.fit(X, y)
```

Visualizations help show how kernels allow SVMs to handle non-linear data.

---

## Support Vector Regression (SVR)

### `SVR.ipynb`

This notebook applies **Support Vector Regression** to predict continuous values using the Tips dataset.

Key points:
- Uses real-world data for regression
- Explores relationships between features
- Demonstrates epsilon-insensitive loss

Basic commands used:
```python
from sklearn.svm import SVR

svr = SVR(kernel='rbf')
svr.fit(X_train, y_train)
y_pred = svr.predict(X_test)
```

---


## Author

**Manasa Vijayendra Gokak**  
Graduate Student – Data Science  
DePaul University
