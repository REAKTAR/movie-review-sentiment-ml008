"""Metrics, confusion matrix, coefficient analysis, error analysis (notebook Steps 7-10, Member 3)."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

sns.set_theme(style="whitegrid")


def plot_confusion(y_test, y_pred, name, path="figures/confusion_matrix.png"):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title(f"Confusion Matrix - {name}")
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    return cm


def plot_model_comparison(results, path="figures/model_comparison.png"):
    results.plot(kind="bar", figsize=(9, 4), colormap="viridis")
    plt.title("Model comparison"); plt.ylim(0.6, 1.0); plt.xticks(rotation=0)
    plt.legend(loc="lower right"); plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()


def top_words(lr_model, tfidf, k=15, path="figures/feature_importance.png"):
    """Largest +/- Logistic Regression coefficients = words driving the prediction."""
    feat, coef = np.array(tfidf.get_feature_names_out()), lr_model.coef_[0]
    order = np.argsort(coef)
    top_neg, top_pos = feat[order[:k]], feat[order[-k:]][::-1]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].barh(top_pos[::-1], np.sort(coef)[-k:], color="#2e8b57"); ax[0].set_title("Top positive words")
    ax[1].barh(top_neg[::-1], np.sort(coef)[:k][::-1], color="#d9534f"); ax[1].set_title("Top negative words")
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    return list(top_pos), list(top_neg)


def misclassified(df, X_test_text, y_test, y_pred):
    err = X_test_text.to_frame("clean_review").copy()
    err["actual"] = np.where(y_test == 1, "Positive", "Negative")
    err["predicted"] = np.where(y_pred == 1, "Positive", "Negative")
    err["original"] = df.loc[err.index, "review"]
    wrong = err[err["actual"] != err["predicted"]][["original", "actual", "predicted"]].copy()
    wrong["reason"] = ""   # fill by hand: sarcasm / negation / mixed opinion / quoted criticism
    return wrong


def predict_sentiment(review, tfidf, model, clean_fn):
    return "Positive" if model.predict(tfidf.transform([clean_fn(review)]))[0] == 1 else "Negative"


if __name__ == "__main__":
    import os
    from preprocess import clean_text          # run as: python src/evaluate.py
    from vectorize import split_data, build_tfidf
    from train import train_all
    df = pd.read_csv("data/processed/cleaned_movie_reviews.csv")
    Xtr_t, Xte_t, ytr, yte = split_data(df)
    tfidf, Xtr, Xte = build_tfidf(Xtr_t, Xte_t)
    models, preds, results = train_all(Xtr, ytr, Xte, yte)
    best = results["F1-Score"].idxmax()
    os.makedirs("figures", exist_ok=True); os.makedirs("results", exist_ok=True)
    print(classification_report(yte, preds[best], target_names=["Negative", "Positive"]))
    print("Confusion matrix:\n", plot_confusion(yte, preds[best], best))
    plot_model_comparison(results)
    pos, neg = top_words(models["Logistic Regression"], tfidf)
    print("Top positive:", pos, "\nTop negative:", neg)
    misclassified(df, Xte_t, yte, preds[best]).to_csv("results/misclassified_reviews.csv", index=False)
