from flask import Flask, render_template, Response
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io

app = Flask(__name__)

# ===== LOAD DATA =====
df = pd.read_csv("dataset.csv")
df.replace('?', pd.NA, inplace=True)
df.dropna(inplace=True)

le = LabelEncoder()
for col in df.columns:
    df[col] = le.fit_transform(df[col])

X = df.drop('income', axis=1)
y = df['income']


# ===== HOME =====
@app.route('/')
def home():
    return render_template('index.html')


# ===== TREE PAGE =====
@app.route('/tree/<model>')
def tree(model):

    descriptions = {

        "id3": """
        ID3 (Iterative Dichotomiser 3) is a decision tree algorithm developed by Ross Quinlan.
        It uses Information Gain and Entropy to determine the best feature for splitting data.

        Features:
        - Uses Entropy
        - Uses Information Gain
        - Works mainly with categorical data
        - Can overfit on training data

        Advantages:
        - Simple to understand
        - Fast for small datasets
        - Easy to implement

        Disadvantages:
        - Overfitting problem
        - Cannot handle continuous values efficiently
        - Biased toward attributes with many values
        """,

        "c45": """
        C4.5 is an improved version of the ID3 algorithm developed by Ross Quinlan.

        It uses Gain Ratio instead of Information Gain, which reduces bias.
        It can also handle continuous and missing values.

        Features:
        - Uses Gain Ratio
        - Handles continuous data
        - Handles missing values
        - Performs pruning

        Advantages:
        - Better accuracy than ID3
        - Reduces overfitting
        - More practical for real datasets

        Disadvantages:
        - More computationally expensive
        - More complex than ID3
        """,

        "cart": """
        CART (Classification and Regression Trees) is a binary decision tree algorithm.

        It uses the Gini Index to split nodes and creates binary trees.

        Features:
        - Uses Gini Index
        - Creates binary splits
        - Supports classification and regression

        Advantages:
        - Simple and efficient
        - Handles numerical data well
        - Widely used in machine learning

        Disadvantages:
        - Can overfit
        - Sensitive to small data changes
        """,

        "entropy": """
        Entropy-based Decision Tree uses entropy as the impurity measure.

        Lower entropy means better purity in the dataset split.

        Features:
        - Uses entropy criterion
        - Implemented using sklearn
        - Produces accurate splits

        Advantages:
        - High interpretability
        - Good classification performance

        Disadvantages:
        - Slightly slower than Gini
        """,

        "gini": """
        Gini Decision Tree uses Gini Impurity to evaluate splits.

        Lower Gini value indicates purer nodes.

        Features:
        - Uses Gini Index
        - Faster than entropy
        - Common in CART

        Advantages:
        - Fast computation
        - Efficient for large datasets

        Disadvantages:
        - Sometimes less informative than entropy
        """
    }

    images = {
        "id3": "id3.png",
        "c45": "c45.png",
        "cart": "cart.png",
        "entropy": "entropy.png",
        "gini": "gini.png"
    }

    return render_template(
        "tree.html",
        model=model,
        description=descriptions.get(model, ""),
        image=images.get(model, "")
    )

# ===== DYNAMIC TREE IMAGE =====
@app.route('/plot/<model>')
def plot(model):

    if model == "entropy":
        clf = DecisionTreeClassifier(criterion='entropy', max_depth=3)

    elif model == "gini":
        clf = DecisionTreeClassifier(criterion='gini', max_depth=3)

    elif model == "cart":
        clf = DecisionTreeClassifier(criterion='gini', max_depth=3)

    else:
        # fallback (ID3, C4.5 → approximate using entropy)
        clf = DecisionTreeClassifier(criterion='entropy', max_depth=3)

    clf.fit(X, y)

    fig = plt.figure(figsize=(12, 8))
    plot_tree(clf, filled=True, feature_names=X.columns)

    img = io.BytesIO()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)

    return Response(img.getvalue(), mimetype='image/png')


# ===== COMPARISON PAGE =====
@app.route('/compare')
def compare():
    return render_template("compare.html")


# ===== ACCURACY GRAPH =====
@app.route('/accuracy')
def accuracy():

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    models = {
        "Entropy": DecisionTreeClassifier(criterion='entropy'),
        "Gini": DecisionTreeClassifier(criterion='gini'),
        "Random Forest": RandomForestClassifier(),
        "Extra Trees": ExtraTreesClassifier()
    }

    scores = []

    for m in models.values():
        m.fit(X_train, y_train)
        pred = m.predict(X_test)
        scores.append(accuracy_score(y_test, pred))

    plt.figure()
    plt.bar(models.keys(), scores)

    img = io.BytesIO()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)

    return Response(img.getvalue(), mimetype='image/png')


if __name__ == "__main__":
    app.run(debug=True)