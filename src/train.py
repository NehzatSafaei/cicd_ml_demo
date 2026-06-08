import os
import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier


def main():
    X, y = load_iris(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = XGBClassifier(
        n_estimators=10,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
        eval_metric="mlogloss"
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print(f"Accuracy: {acc:.4f}")

    # -------------------------
    # OUTPUT DIRECTORY
    # -------------------------
    os.makedirs("output", exist_ok=True)

    # save metrics
    with open("output/metrics.txt", "w") as f:
        f.write(f"accuracy={acc:.4f}\n")
    with open("output/test.txt", "w") as f:
        f.write("ci ok\n")

    # save model
    joblib.dump(model, "output/model.pkl")
    print("Model saved to output/model.pkl")


if __name__ == "__main__":
    main()
    print("CI TEST RUNNING")
