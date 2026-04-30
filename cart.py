import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

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

X_np = X.values
y_np = y.values


# ================= GINI =================

def gini(y):
    unique, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()
    
    g = 1
    for p in probs:
        g -= p**2
    
    return g


def gini_index(X, y, feature):
    values, counts = np.unique(X[:, feature], return_counts=True)
    
    gini_total = 0
    for i in range(len(values)):
        subset_y = y[X[:, feature] == values[i]]
        gini_total += (counts[i] / np.sum(counts)) * gini(subset_y)
    
    return gini_total


# ================= CART =================

def cart(X, y, features):
    
    if len(np.unique(y)) == 1:
        return y[0]
    
    if len(features) == 0:
        return np.bincount(y).argmax()
    
    gini_values = [gini_index(X, y, f) for f in features]
    best_feature = features[np.argmin(gini_values)]  # MIN gini
    
    tree = {best_feature: {}}
    
    values = np.unique(X[:, best_feature])
    
    for value in values:
        subset_X = X[X[:, best_feature] == value]
        subset_y = y[X[:, best_feature] == value]
        
        new_features = [f for f in features if f != best_feature]
        subtree = cart(subset_X, subset_y, new_features)
        
        tree[best_feature][value] = subtree
    
    return tree


# ================= PRINT =================

def print_tree(tree, indent=""):
    if not isinstance(tree, dict):
        print(indent + "→ Class:", tree)
        return
    
    for feature, branches in tree.items():
        for value, subtree in branches.items():
            print(indent + f"Feature {feature} = {value}:")
            print_tree(subtree, indent + "   ")


# ================= RUN =================

features = list(range(3))

tree = cart(X_np, y_np, features)

print("\nCART TREE:\n")
print_tree(tree)

from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

model = DecisionTreeClassifier(criterion='gini', max_depth=3)
model.fit(X, y)

plt.figure(figsize=(15,10))
plot_tree(model, filled=True, feature_names=X.columns, class_names=["<=50K", ">50K"])
plt.title("CART (Gini) Tree")
plt.savefig("static/cart.png")
plt.close()