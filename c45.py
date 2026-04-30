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


# ================= FUNCTIONS =================

def entropy(y):
    unique, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()
    e = 0
    for p in probs:
        e -= p * np.log2(p)
    return e


def information_gain(X, y, feature):
    total_entropy = entropy(y)
    values, counts = np.unique(X[:, feature], return_counts=True)
    
    weighted_entropy = 0
    for i in range(len(values)):
        subset_y = y[X[:, feature] == values[i]]
        weighted_entropy += (counts[i] / np.sum(counts)) * entropy(subset_y)
    
    return total_entropy - weighted_entropy


# 🔥 NEW PART (C4.5)

def split_info(X, feature):
    values, counts = np.unique(X[:, feature], return_counts=True)
    probs = counts / counts.sum()
    
    si = 0
    for p in probs:
        si -= p * np.log2(p)
    
    return si


def gain_ratio(X, y, feature):
    ig = information_gain(X, y, feature)
    si = split_info(X, feature)
    
    if si == 0:
        return 0
    
    return ig / si


def c45(X, y, features):
    if len(np.unique(y)) == 1:
        return y[0]
    
    if len(features) == 0:
        return np.bincount(y).argmax()
    
    gains = [gain_ratio(X, y, f) for f in features]
    best_feature = features[np.argmax(gains)]
    
    tree = {best_feature: {}}
    
    values = np.unique(X[:, best_feature])
    
    for value in values:
        subset_X = X[X[:, best_feature] == value]
        subset_y = y[X[:, best_feature] == value]
        
        new_features = [f for f in features if f != best_feature]
        subtree = c45(subset_X, subset_y, new_features)
        
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

tree = c45(X_np, y_np, features)

print("\nC4.5 TREE:\n")
print_tree(tree)

from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Visualization (approximation)
model = DecisionTreeClassifier(criterion='entropy', max_depth=3)
model.fit(X, y)

plt.figure(figsize=(15,10))
plot_tree(model, filled=True, feature_names=X.columns, class_names=["<=50K", ">50K"])
plt.title("Approximate Visualization of C4.5 (using entropy)")
plt.savefig("static/c45.png")
plt.close()