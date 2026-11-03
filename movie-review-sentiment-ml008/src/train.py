"""Model training: Logistic Regression, Multinomial NB, Linear SVM (notebook Step 6, Member 2)."""
import os
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

RANDOM_STATE = 42


def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Naive Bayes": MultinomialNB(),
        "Linear SVM": LinearSVC(random_state=RANDOM_STATE),
    }


def train_all(X_train, y_train, X_test, y_test):
    models, preds, rows = get_models(), {}, []
    for name, model in models.items():
        model.fit(X_train, y_train)
        p = model.predict(X_test)
        preds[name] = p
        rows.append({"Model": name,
                     "Accuracy": accuracy_score(y_test, p),
                     "Precision": precision_score(y_test, p),
                     "Recall": recall_score(y_test, p),
                     "F1-Score": f1_score(y_test, p)})
    results = pd.DataFrame(rows).set_index("Model").round(4)
    return models, preds, results


def save_artifacts(models, tfidf, results, best_name, out_dir="models"):
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(tfidf, f"{out_dir}/tfidf.pkl")
    joblib.dump(models[best_name], f"{out_dir}/best_model.pkl")
    joblib.dump(models["Logistic Regression"], f"{out_dir}/logistic_regression.pkl")
    os.makedirs("results", exist_ok=True)
    results.to_csv("results/model_results.csv")


if __name__ == "__main__":
    import argparse
    from vectorize import split_data, build_tfidf   # run as: python src/train.py
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/processed/cleaned_movie_reviews.csv")
    a = p.parse_args()
    df = pd.read_csv(a.data)
    Xtr_t, Xte_t, ytr, yte = split_data(df)
    tfidf, Xtr, Xte = build_tfidf(Xtr_t, Xte_t)
    models, preds, results = train_all(Xtr, ytr, Xte, yte)
    best = results["F1-Score"].idxmax()
    save_artifacts(models, tfidf, results, best)
    print(results, "\nBest model:", best)
