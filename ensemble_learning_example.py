from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def run_bagging_example(X, y):
    """Train and evaluate a bagging ensemble using RandomForest."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=50,
        max_depth=4,
        random_state=42,
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print("Bagging example: RandomForestClassifier")
    print(f"Accuracy: {accuracy:.3f}")
    print()


def run_boosting_example(X, y):
    """Train and evaluate a boosting ensemble using AdaBoost."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    model = AdaBoostClassifier(
        n_estimators=50,
        learning_rate=1.0,
        random_state=42,
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print("Boosting example: AdaBoostClassifier")
    print(f"Accuracy: {accuracy:.3f}")
    print()


def main():
    data = load_iris()
    X, y = data.data, data.target

    run_bagging_example(X, y)
    run_boosting_example(X, y)


if __name__ == "__main__":
    main()
