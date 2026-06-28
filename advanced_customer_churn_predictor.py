import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, f1_score, precision_score, recall_score
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore')

def load_and_merge_data():
    print("Loading data...")
    customers = pd.read_csv('customers 2.csv')
    orders = pd.read_csv('orders 2.csv')
    sales = pd.read_csv('sales 2.csv')
    
    df = pd.merge(sales, orders, on='order_id', how='inner')
    df = pd.merge(df, customers, on='customer_id', how='inner')
    df['order_date'] = pd.to_datetime(df['order_date'])
    return df

def build_leakage_free_clusters(pre_df):
    product_features = pre_df.groupby('product_id').agg(
        Total_Quantity_Sold=('quantity', 'sum'),
        Total_Revenue=('total_price', 'sum'),
        Unique_Orders=('order_id', 'nunique'),
        Unique_Customers=('customer_id', 'nunique')
    ).reset_index()
    
    scaler_prod = StandardScaler()
    scaled_prod = scaler_prod.fit_transform(product_features[['Total_Quantity_Sold', 'Total_Revenue', 'Unique_Orders']])
    kmeans_prod = KMeans(n_clusters=4, random_state=42, n_init=10)
    product_features['Product_Cluster_ID'] = kmeans_prod.fit_predict(scaled_prod)
    
    pre_df = pd.merge(pre_df, product_features[['product_id', 'Product_Cluster_ID']], on='product_id', how='left')
    return pre_df

def advanced_feature_engineering(df):
    print("Performing advanced feature engineering (Leakage-Free)...")
    max_date = df['order_date'].max()
    cutoff_date = max_date - pd.Timedelta(days=90)
    
    pre_df = df[df['order_date'] <= cutoff_date].copy()
    post_df = df[df['order_date'] > cutoff_date].copy()
    
    pre_df = build_leakage_free_clusters(pre_df)
    
    customer_features = pre_df.groupby('customer_id').agg(
        Recency=('order_date', lambda x: (cutoff_date - x.max()).days),
        Tenure=('order_date', lambda x: (cutoff_date - x.min()).days),
        Frequency=('order_id', 'nunique'),
        Monetary=('total_price', 'sum'),
        Distinct_Products=('product_id', 'nunique'),
        Age=('age', 'first'),
        Gender=('gender', 'first'),
        State=('state', 'first')
    ).reset_index()
    
    first_order_info = pre_df.sort_values(['order_date', 'order_id']).groupby('customer_id')[['order_id']].first().reset_index()
    first_order_rows = pd.merge(pre_df, first_order_info, on=['customer_id', 'order_id'])
    first_order_features = first_order_rows.groupby('customer_id').agg(
        First_Order_Value=('total_price', 'sum'),
        First_Order_Items=('quantity', 'sum')
    ).reset_index()
    customer_features = pd.merge(customer_features, first_order_features, on='customer_id', how='left')
    
    momentum_cutoff_30 = cutoff_date - pd.Timedelta(days=30)
    momentum_30 = pre_df[pre_df['order_date'] > momentum_cutoff_30].groupby('customer_id')['total_price'].sum().reset_index(name='Momentum_30d_Spend')
    customer_features = pd.merge(customer_features, momentum_30, on='customer_id', how='left')
    customer_features['Momentum_30d_Spend'] = customer_features['Momentum_30d_Spend'].fillna(0)
    
    prod_spend = pre_df.groupby(['customer_id', 'Product_Cluster_ID'])['total_price'].sum().unstack(fill_value=0)
    prod_spend = prod_spend.div(prod_spend.sum(axis=1), axis=0)
    prod_spend.columns = [f'Pct_Spend_ProdCluster_{int(c)}' for c in prod_spend.columns]
    prod_spend.reset_index(inplace=True)
    
    affinity_cols = [col for col in prod_spend.columns if 'Pct_Spend_ProdCluster' in col]
    if len(affinity_cols) > 1:
        prod_spend = prod_spend.drop(columns=[affinity_cols[-1]])
        print(f"  Dropped '{affinity_cols[-1]}' to fix multicollinearity ({len(affinity_cols)-1} affinities kept)")
    
    customer_features = pd.merge(customer_features, prod_spend, on='customer_id', how='left')
    
    remaining_affinity_cols = [col for col in customer_features.columns if 'Pct_Spend_ProdCluster' in col]
    customer_features[remaining_affinity_cols] = customer_features[remaining_affinity_cols].fillna(0)
    
    customer_features = pd.get_dummies(customer_features, columns=['Gender', 'State'], drop_first=True)
    
    post_behavior = post_df.groupby('customer_id').agg(Future_Spend=('total_price', 'sum')).reset_index()
    customer_data = pd.merge(customer_features, post_behavior, on='customer_id', how='left')
    customer_data['Future_Spend'] = customer_data['Future_Spend'].fillna(0)
    
    customer_data['Churn'] = (customer_data['Future_Spend'] == 0).astype(int)
    
    churn_rate = customer_data['Churn'].mean()
    print(f"  Churn Rate: {churn_rate:.2%} ({customer_data['Churn'].sum()} churned / {len(customer_data)} total)")
    
    return customer_data

def apply_customer_kmeans(customer_data):
    cluster_features = customer_data[['Recency', 'Frequency', 'Monetary', 'Age']]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(cluster_features)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    customer_data['Cluster_ID'] = kmeans.fit_predict(scaled_features)
    return customer_data

def train_persona_classifier(customer_data):
    print("\n--- 1. PERSONA CLASSIFIER (Day-1 Behavior: Hyper-Tuned) ---")
    
    feature_cols = ['Age', 'First_Order_Value', 'First_Order_Items'] + \
                   [col for col in customer_data.columns if col.startswith('Gender_') or col.startswith('State_')]
    
    X = customer_data[feature_cols]
    y = customer_data['Cluster_ID']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    def print_persona_metrics(model_name, preds, best_params=None):
        print(f"\nModel: {model_name}")
        if best_params:
            print(f"  Best Params: {best_params}")
        print(f"  -> Accuracy: {accuracy_score(y_test, preds):.2%}")
        print("  -> Classification Report:")
        print(classification_report(y_test, preds, zero_division=0))
    
    lr_base = LogisticRegression(random_state=42, max_iter=2000, class_weight='balanced')
    param_grid_lr = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
    lr_tuned = RandomizedSearchCV(lr_base, param_grid_lr, n_iter=6, cv=3, random_state=42, scoring='f1_macro', n_jobs=-1)
    lr_tuned.fit(X_train_scaled, y_train)
    print_persona_metrics("Tuned Logistic Regression", lr_tuned.predict(X_test_scaled), lr_tuned.best_params_)
    
    svm_base = SVC(class_weight='balanced', random_state=42)
    param_grid_svm = {'C': [0.1, 1, 10, 50], 'gamma': ['scale', 'auto', 0.1, 0.01], 'kernel': ['rbf']}
    svm_tuned = RandomizedSearchCV(svm_base, param_grid_svm, n_iter=8, cv=3, random_state=42, scoring='f1_macro', n_jobs=-1)
    svm_tuned.fit(X_train_scaled, y_train)
    print_persona_metrics("Tuned SVM (RBF)", svm_tuned.predict(X_test_scaled), svm_tuned.best_params_)
    
    rf_base = RandomForestClassifier(class_weight='balanced', random_state=42)
    param_grid_rf = {'n_estimators': [100, 200], 'max_depth': [None, 5, 10], 'min_samples_split': [2, 5]}
    rf_tuned = RandomizedSearchCV(rf_base, param_grid_rf, n_iter=8, cv=3, random_state=42, scoring='f1_macro', n_jobs=-1)
    rf_tuned.fit(X_train, y_train)
    print_persona_metrics("Tuned Random Forest", rf_tuned.predict(X_test), rf_tuned.best_params_)

def train_churn_predictor(customer_data):
    print("\n--- 2. CHURN PREDICTOR (90-Day Window: Core Features) ---")
    
    feature_cols = [
        'Recency', 'Frequency', 'Monetary', 'Age', 'Momentum_30d_Spend',
        'Distinct_Products'
    ] + [col for col in customer_data.columns if 'Pct_Spend_ProdCluster' in col]
    
    customer_data = customer_data.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    X = customer_data[feature_cols]
    y = customer_data['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    print(f"Class Distribution -> Stay (0): {y_train.value_counts()[0]}, Churn (1): {y_train.value_counts()[1]}")
    print(f"Features used: {len(feature_cols)} features\n")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("\n--- Tuning Hyperparameters (Optimizing for F1-Score) ---")
    
    def print_metrics(model_name, y_true, preds, probs, best_params=None):
        print(f"\nModel: {model_name}")
        if best_params:
            print(f"  Best Params: {best_params}")
        print(f"  -> Accuracy:  {accuracy_score(y_true, preds):.2%}")
        print(f"  -> Precision: {precision_score(y_true, preds, zero_division=0):.2%}")
        print(f"  -> Recall:    {recall_score(y_true, preds, zero_division=0):.2%}")
        print(f"  -> F1-Score:  {f1_score(y_true, preds, zero_division=0):.2%}")
        print(f"  -> ROC-AUC:   {roc_auc_score(y_true, probs):.2%}")
    
    lr_base = LogisticRegression(random_state=42, max_iter=2000)
    param_grid_lr = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'penalty': ['l2']}
    lr_tuned = RandomizedSearchCV(lr_base, param_grid_lr, n_iter=6, cv=3, random_state=42, scoring='f1', n_jobs=-1)
    lr_tuned.fit(X_train_scaled, y_train)
    lr_preds = lr_tuned.predict(X_test_scaled)
    lr_probs = lr_tuned.predict_proba(X_test_scaled)[:, 1]
    print_metrics("Tuned Logistic Regression", y_test, lr_preds, lr_probs, lr_tuned.best_params_)
    
    svm_base = SVC(probability=True, random_state=42, kernel='rbf')
    param_grid_svm = {'C': [0.1, 1, 10, 50], 'gamma': ['scale', 'auto', 0.1, 0.01]}
    svm_tuned = RandomizedSearchCV(svm_base, param_grid_svm, n_iter=8, cv=3, random_state=42, scoring='f1', n_jobs=-1)
    svm_tuned.fit(X_train_scaled, y_train)
    svm_preds = svm_tuned.predict(X_test_scaled)
    svm_probs = svm_tuned.predict_proba(X_test_scaled)[:, 1]
    print_metrics("Tuned SVM (RBF)", y_test, svm_preds, svm_probs, svm_tuned.best_params_)
    
    rf_base = RandomForestClassifier(random_state=42)
    param_grid_rf = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 5, 10, 15],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    rf_tuned = RandomizedSearchCV(rf_base, param_grid_rf, n_iter=10, cv=3, random_state=42, scoring='f1', n_jobs=-1)
    rf_tuned.fit(X_train, y_train)
    rf_preds = rf_tuned.predict(X_test)
    rf_probs = rf_tuned.predict_proba(X_test)[:, 1]
    print_metrics("Tuned Random Forest", y_test, rf_preds, rf_probs, rf_tuned.best_params_)

if __name__ == "__main__":
    df = load_and_merge_data()
    customer_data = advanced_feature_engineering(df)
    customer_data = apply_customer_kmeans(customer_data)
    
    train_persona_classifier(customer_data)
    train_churn_predictor(customer_data)
