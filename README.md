# 💰 Income Prediction System
An interactive Machine Learning web application built using **Flask, Python, and Scikit-learn** for predicting income levels and visualizing multiple Decision Tree algorithms.

---

## 📌 Project Overview
This project demonstrates how different **Decision Tree algorithms** can be used to predict whether a person's income is **<=50K or >50K** based on given features.

The system also provides visualizations and comparisons of various tree-based models to help understand their behavior.

---

## ✨ Features
- 🔮 Income Prediction based on user input
- 🌳 Decision Tree visualization
- 📊 Accuracy comparison graphs
- 📖 Detailed algorithm descriptions
- 🎨 Modern dark-themed UI
- ⚡ Flask-based dynamic web application

---

## 🚀 Algorithms Implemented

### 1. ID3 (Iterative Dichotomiser 3)
- Uses **Entropy & Information Gain**
- Best for categorical data
- Simple and interpretable

### 2. C4.5
- Improved ID3
- Uses **Gain Ratio**
- Handles continuous values

### 3. CART (Classification and Regression Trees)
- Uses **Gini Index**
- Produces binary trees

### 4. Decision Tree (Entropy)
- Implemented using **Scikit-learn**
- Uses entropy for splitting

### 5. Decision Tree (Gini)
- Uses **Gini Impurity**
- Faster than entropy

### 6. Random Forest
- Ensemble method
- Combines multiple trees for better accuracy

### 7. Extra Trees Classifier
- Uses random splits
- Faster and more diverse trees

---

## 🛠️ Technologies Used
- Python
- Flask
- HTML5
- CSS3
- Scikit-learn
- Pandas
- NumPy
- Matplotlib

---

## 📂 Project Structure

AML-Assignment-2/
│
├── static/
│ ├── style.css
│ ├── id3.png
│ ├── c45.png
│ ├── cart.png
│ ├── entropy.png
│ ├── gini.png
│
├── templates/
│ ├── index.html
│ ├── tree.html
│ ├── compare.html
│ ├── predict.html
│
├── app.py
├── id3.py
├── c45.py
├── cart.py
├── sklearn_models.py
├── dataset.csv
└── README.md


---

## ⚙️ Installation & Setup

### 1️⃣ Clone Repository
```bash
git clone https://github.com/Navya1370/AML-Assignment-2.git
2️⃣ Navigate to Project Folder
cd AML-Assignment-2
3️⃣ Install Dependencies
pip install flask pandas numpy matplotlib scikit-learn
4️⃣ Generate Tree Images

Run once:

python id3.py
python c45.py
python cart.py
python sklearn_models.py
5️⃣ Run Application
python app.py
🌐 Open in Browser
http://127.0.0.1:5000

📚 Learning Outcomes
Understanding Decision Tree algorithms
Entropy, Gini, Information Gain concepts
Machine Learning model comparison
Flask web development
Data preprocessing techniques
⭐ Future Improvements
Add interactive tree visualization
Upload custom datasets
Add confusion matrix & reports
Deploy as a live website

👩‍💻 Author

Navya

🔗 GitHub:
https://github.com/Navya1370

📁 Project Repo:
https://github.com/Navya1370/AML-Assignment-2
