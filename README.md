# 💰 Income Prediction System

An interactive Machine Learning web application built using **Flask, Python, and Scikit-learn** for predicting income levels and visualizing multiple Decision Tree algorithms.

---

## 📌 Project Overview

This project demonstrates how different **Decision Tree algorithms** can be used to predict whether a person's income is **<=50K or >50K** based on given features.

The system also provides:
- 🌳 Decision tree visualizations  
- 📊 Model comparison graphs  
- 📖 Algorithm explanations  
- 🔮 Income prediction using user input  

---

## ✨ Features

- 🔮 Income Prediction  
- 🌳 Decision Tree Visualization  
- 📊 Accuracy Comparison  
- 📖 Algorithm Descriptions  
- 🎨 Dark-Themed UI  
- ⚡ Flask Web Application  

---

## 🚀 Algorithms Implemented

- ID3 (Information Gain)  
- C4.5 (Gain Ratio)  
- CART (Gini Index)  
- Decision Tree (Entropy - sklearn)  
- Decision Tree (Gini - sklearn)  
- Random Forest  
- Extra Trees Classifier  

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
├── app.py
├── dataset.csv
│
├── id3.py
├── c45.py
├── cart.py
├── sklearn_models.py
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
│ ├── result.html
│
├── README.md
└── requirements.txt


---

## ⚙️ Installation & Setup

* Clone Repository

git clone https://github.com/Navya1370/AML-Assignment-2.git

* Open Project Folder

cd AML-Assignment-2

* Install Dependencies

pip install flask pandas numpy matplotlib scikit-learn

* Generate Tree Images (Run Once)

python id3.py
python c45.py
python cart.py
python sklearn_models.py

* Run Application

python app.py


🌐 Open in Browser

http://127.0.0.1:5000

---

📚 Learning Outcomes

* Decision Tree Algorithms
* Entropy & Gini Concepts
* Model Comparison
* Flask Web Development
* Data Preprocessing

---

⭐ Future Improvements

* Interactive tree visualization
* Upload custom datasets
* Add confusion matrix
* Deploy project online

---

👩‍💻 Author

Navya
GitHub: https://github.com/Navya1370

Project Link:
https://github.com/Navya1370/AML-Assignment-2

---

📄 License

This project is developed for academic purposes.
