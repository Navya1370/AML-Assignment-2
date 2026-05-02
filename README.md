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

Income-Prediction-System/
│
├── app.py                     # Main Flask application
├── dataset.csv               # Adult dataset (cleaned/used for training)
│
├── models/                   # (Optional but recommended)
│   ├── id3.py                # Custom ID3 implementation
│   ├── c45.py                # Custom C4.5 implementation
│   ├── cart.py               # Custom CART implementation
│   ├── sklearn_models.py     # Entropy, Gini, RF, ExtraTrees
│
├── static/                   # Static files (CSS, images)
│   ├── style.css             # Dark theme UI styling
│   ├── id3.png               # (optional – if you pre-save trees)
│   ├── c45.png
│   ├── cart.png
│   ├── entropy.png
│   ├── gini.png
│
├── templates/                # HTML pages
│   ├── index.html            # Home page (model selection UI)
│   ├── tree.html             # Tree visualization page
│   ├── compare.html          # Model comparison page
│   ├── predict.html          # Prediction form page
│
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
│
└── .gitignore                # Ignore venv, cache files

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

