import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import matplotlib.pyplot as plt

# ================= LOAD DATA =================
df = pd.read_csv("dataset.csv")

df.replace('?', pd.NA, inplace=True)
df.dropna(inplace=True)

# Encode
le = LabelEncoder()
for col in df.columns:
    df[col] = le.fit_transform(df[col])

# Split
X = df.drop('income', axis=1)
y = df['income']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ================= MODELS =================

# 1. Decision Tree (Entropy)
dt_entropy = DecisionTreeClassifier(criterion='entropy', max_depth=3)
dt_entropy.fit(X_train, y_train)
pred1 = dt_entropy.predict(X_test)

# 2. Decision Tree (Gini)
dt_gini = DecisionTreeClassifier(criterion='gini', max_depth=3)
dt_gini.fit(X_train, y_train)
pred2 = dt_gini.predict(X_test)

# 3. Random Forest
rf = RandomForestClassifier(n_estimators=50)
rf.fit(X_train, y_train)
pred3 = rf.predict(X_test)

# 4. Extra Trees
et = ExtraTreesClassifier(n_estimators=50)
et.fit(X_train, y_train)
pred4 = et.predict(X_test)

# ================= ACCURACY =================

print("\nModel Accuracies:\n")

print("Decision Tree (Entropy):", accuracy_score(y_test, pred1))
print("Decision Tree (Gini):", accuracy_score(y_test, pred2))
print("Random Forest:", accuracy_score(y_test, pred3))
print("Extra Trees:", accuracy_score(y_test, pred4))


# ================= VISUALIZATION =================

# ================= VISUALIZATION =================

# ENTROPY TREE
plt.figure(figsize=(20,10))

plot_tree(
    dt_entropy,
    filled=True,
    feature_names=X.columns,
    class_names=["<=50K", ">50K"]
)

plt.title("Decision Tree - Entropy")

plt.savefig("static/entropy.png")

plt.close()


# GINI TREE
plt.figure(figsize=(20,10))

plot_tree(
    dt_gini,
    filled=True,
    feature_names=X.columns,
    class_names=["<=50K", ">50K"]
)

plt.title("Decision Tree - Gini")

plt.savefig("static/gini.png")

plt.close()


# RANDOM FOREST TREE
plt.figure(figsize=(20,10))

plot_tree(
    rf.estimators_[0],
    filled=True,
    feature_names=X.columns,
    class_names=["<=50K", ">50K"]
)

plt.title("Random Forest - One Tree")

plt.savefig("static/randomforest.png")

plt.close()