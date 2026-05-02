from flask import Flask, render_template, Response
from flask import request   # add this at top (important)
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

categorical_cols = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country']

encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

target_encoder = LabelEncoder()
df['income'] = target_encoder.fit_transform(df['income'])

X = df.drop('income', axis=1)
y = df['income']

# Train model globally for prediction
global_model = DecisionTreeClassifier(criterion='entropy', max_depth=5)
global_model.fit(X, y)

# Defaults for excluded form fields
defaults = {
    'fnlwgt': X['fnlwgt'].median(),
    'education.num': X['education.num'].mode()[0],
    'capital.gain': 0,
    'capital.loss': 0,
    'native.country': encoders['native.country'].transform(['United-States'])[0] if 'United-States' in encoders['native.country'].classes_ else X['native.country'].mode()[0]
}


# ===== HOME =====
@app.route('/')
def home():
    return render_template('index.html')


#----

@app.route('/predict-page')
def predict_page():
    form_data = {
        col: encoders[col].classes_.tolist() for col in ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex']
    }
    return render_template('predict.html', form_data=form_data)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        sample = [
            int(request.form.get('age', 30)),
            encoders['workclass'].transform([request.form.get('workclass')])[0],
            defaults['fnlwgt'],
            encoders['education'].transform([request.form.get('education')])[0],
            defaults['education.num'],
            encoders['marital.status'].transform([request.form.get('marital.status')])[0],
            encoders['occupation'].transform([request.form.get('occupation')])[0],
            encoders['relationship'].transform([request.form.get('relationship')])[0],
            encoders['race'].transform([request.form.get('race')])[0],
            encoders['sex'].transform([request.form.get('sex')])[0],
            defaults['capital.gain'],
            defaults['capital.loss'],
            int(request.form.get('hours.per.week', 40)),
            defaults['native.country']
        ]
        
        prediction = global_model.predict([sample])
        result = target_encoder.inverse_transform(prediction)[0]
    except Exception as e:
        result = f"Error: {str(e)}"

    return render_template('result.html', result=result)


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