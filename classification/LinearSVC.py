"""Intent classification model: sklearn SVC with linear kernel."""
import pandas as pd
from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC


def get_data(data_path, x_col="sample", y_col="label"):
    """Get x, y columns from the given csv file."""
    df = pd.read_csv(data_path)
    x = df[x_col]
    y = df[y_col]
    return x, y


def train(data_path, model_path):
    """Train a vectorizer-classifier pipeline with scikit-learn's SVC."""
    x, y = get_data(data_path)

    pipe = Pipeline(
        [
            ("vect", TfidfVectorizer()),
            ("clf", SVC(kernel="linear", probability=True)),
        ]
    )

    pipe.fit(x, y)
    dump(pipe, model_path)
    print("Training complete.")
    return pipe
