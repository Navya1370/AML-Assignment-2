# Decision Tree Explorer 🌳

An interactive Machine Learning web application built using **Flask**, **Python**, and **Scikit-learn** for visualizing and comparing multiple Decision Tree algorithms.

---

# 📌 Project Overview

This project demonstrates the implementation and visualization of different Decision Tree algorithms using the same dataset.

The web application provides:

- 🌳 Interactive Decision Tree visualizations
- 📊 Accuracy comparison graphs
- 📖 Detailed algorithm descriptions
- 🎨 Modern dark-themed UI
- ⚡ Flask-based dynamic web interface

---

# 🚀 Algorithms Implemented

## 1. ID3 (Iterative Dichotomiser 3)

- Uses **Entropy** and **Information Gain**
- Best suited for categorical datasets
- Simple and easy to understand

---

## 2. C4.5

- Improved version of ID3
- Uses **Gain Ratio**
- Handles continuous and missing values

---

## 3. CART (Classification and Regression Trees)

- Uses **Gini Index**
- Produces binary trees
- Widely used in practical ML systems

---

## 4. Decision Tree (Entropy)

- Implemented using `sklearn`
- Uses entropy criterion for node splitting

---

## 5. Decision Tree (Gini)

- Implemented using `sklearn`
- Uses Gini impurity for classification

---

## 6. Random Forest

- Ensemble learning method
- Combines multiple decision trees
- Improves prediction accuracy

---

## 7. Extra Trees Classifier

- Similar to Random Forest
- Uses randomized splits
- Faster and more diverse trees

---

# 🛠️ Technologies Used

- Python
- Flask
- HTML5
- CSS3
- Scikit-learn
- Pandas
- NumPy
- Matplotlib

---

# 📂 Project Structure

```plaintext
AML-Assignment-2/
│
├── static/
│   ├── style.css
│   ├── id3.png
│   ├── c45.png
│   ├── cart.png
│   ├── entropy.png
│   ├── gini.png
│
├── templates/
│   ├── index.html
│   ├── tree.html
│   ├── compare.html
│
├── app.py
├── id3.py
├── c45.py
├── cart.py
├── sklearn_models.py
├── dataset.csv
└── README.md
```

---

# ⚙️ Installation & Setup

## 1️⃣ Clone Repository

```bash
git clone https://github.com/Navya1370/AML-Assignment-2.git
```

---

## 2️⃣ Open Project Folder

```bash
cd AML-Assignment-2
```

---

## 3️⃣ Install Dependencies

```bash
pip install flask pandas numpy matplotlib scikit-learn
```

---

## 4️⃣ Generate Tree Images

Run these files once:

```bash
python id3.py
python c45.py
python cart.py
python sklearn_models.py
```

---

## 5️⃣ Run Flask Application

```bash
python app.py
```

---

# 🌐 Open in Browser

```plaintext
http://127.0.0.1:5000
```

---

# ✨ Features

- 🌳 Dynamic decision tree visualization
- 📊 Model accuracy comparison
- 📖 Detailed explanations for every algorithm
- 🎨 Responsive dark-themed UI
- ⚡ Easy navigation between models
- 📈 Machine learning model comparison

---

# 📷 Screenshots

Add screenshots of:
- Home page
- Tree visualization page
- Comparison graph page

---

# 📚 Learning Outcomes

Through this project, we learned:

- Implementation of Decision Tree algorithms
- Entropy and Gini calculations
- Information Gain and Gain Ratio
- Flask web development
- Machine Learning visualization techniques
- Model comparison and evaluation

---

# 👩‍💻 Author

## Navya

GitHub Profile:  
https://github.com/Navya1370

Project Repository:  
https://github.com/Navya1370/AML-Assignment-2

---

# ⭐ Future Improvements

- Add fully interactive tree visualizations
- Add dataset upload functionality
- Add confusion matrix and classification reports
- Deploy application online
- Improve animations and UI effects

---

# 📄 License

This project is developed for educational and academic purposes.
