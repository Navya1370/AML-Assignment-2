# 💰 Income Prediction System

An interactive Machine Learning web application built using **Flask, Python, and Scikit-learn** for predicting income levels and visualizing multiple Decision Tree algorithms.

---

## 📌 Project Overview

This project demonstrates how different **Decision Tree algorithms** can be used to predict whether a person's income is **<=50K or >50K** based on given features.

It also provides **tree visualizations, model comparison, and explanations** to help understand how each algorithm works.

---

## ✨ Features

- 🔮 Income Prediction using user input  
- 🌳 Decision Tree Visualization  
- 📊 Model Accuracy Comparison  
- 📖 Detailed Algorithm Descriptions  
- 🎨 Dark-Themed Modern UI  
- ⚡ Flask-Based Dynamic Web App  

---

## 🚀 Algorithms Implemented

### 1. ID3 (Iterative Dichotomiser 3)
- Uses Entropy & Information Gain  
- Best for categorical data  
- Simple and interpretable  

### 2. C4.5
- Improved version of ID3  
- Uses Gain Ratio  
- Handles continuous values  

### 3. CART (Classification and Regression Trees)
- Uses Gini Index  
- Produces binary trees  

### 4. Decision Tree (Entropy)
- Implemented using Scikit-learn  
- Uses entropy for splitting  

### 5. Decision Tree (Gini)
- Uses Gini Impurity  
- Faster than entropy  

### 6. Random Forest
- Ensemble method  
- Combines multiple trees  
- Improves accuracy  

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

1️⃣ Clone the Repository
git clone https://github.com/Navya1370/AML-Assignment-2.git

2️⃣ Navigate to the Project Folder
cd AML-Assignment-2

3️⃣ Install Required Libraries
pip install flask pandas numpy matplotlib scikit-learn

4️⃣ Generate Decision Tree Images (Run Once)
python id3.py
python c45.py
python cart.py
python sklearn_models.py

5️⃣ Run the Flask Application
python app.py

🌐 Open in Browser
http://127.0.0.1:5000

---

📚 Learning Outcomes
Understanding Decision Tree algorithms
Entropy, Gini, and Information Gain
Machine Learning model comparison
Flask web development
Data preprocessing techniques

---

⭐ Future Improvements
Add interactive tree visualization
Upload custom datasets
Add confusion matrix & reports
Deploy application online
Improve UI animations

---

👩‍💻 Author

Navya

🔗 GitHub:
https://github.com/Navya1370

📁 Project Repository:
https://github.com/Navya1370/AML-Assignment-2

---

📄 License

This project is developed for academic and educational purposes.

