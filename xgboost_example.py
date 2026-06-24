import xgboost as xgb
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

print("Loading the Iris dataset...")
data = load_iris()
X = data.data
y = data.target

print(f"Features: {data.feature_names}")
print(f"Classes:  {list(data.target_names)}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTraining data size: {X_train.shape[0]} samples")
print(f"Testing data size:  {X_test.shape[0]} samples")

print("\nInitializing XGBoost Classifier...")
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    objective="multi:softmax",
    num_class=3,
    random_state=42
)

print("Training the model...")
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
print("\nDetailed Report:")
print(classification_report(y_test, predictions, target_names=data.target_names))
