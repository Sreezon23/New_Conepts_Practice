import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, f1_score, precision_score, recall_score, silhouette_score
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).resolve().parent


def _data_path(filename):
    return BASE_DIR / filename


def load_and_merge_data():
    print("Loading data...")
    customers = pd.read_csv(_data_path('customers 2.csv'))
    orders = pd.read_csv(_data_path('orders 2.csv'))
    sales = pd.read_csv(_data_path('sales 2.csv'))
    
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
        State=('state', 'first'),
        Customer_Name=('customer_name', 'first')
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
    cluster_features = customer_data[['Recency', 'Frequency', 'Monetary', 'Age', 'Distinct_Products',
                                      'First_Order_Value', 'First_Order_Items', 'Momentum_30d_Spend']].copy()

    for col in ['Recency', 'Frequency', 'Monetary', 'Age', 'Distinct_Products',
                'First_Order_Value', 'First_Order_Items', 'Momentum_30d_Spend']:
        cluster_features[col] = np.log1p(cluster_features[col].fillna(0))

    scaler = RobustScaler()
    scaled_features = scaler.fit_transform(cluster_features)

    best_score = -1
    best_labels = None
    best_n_clusters = 2
    best_model_name = 'kmeans'

    for n_clusters in range(2, 7):
        candidate_models = [
            ('kmeans', KMeans(n_clusters=n_clusters, random_state=42, n_init=50)),
            ('minibatch', MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=256, n_init=20))
        ]

        for model_name, model in candidate_models:
            labels = model.fit_predict(scaled_features)
            score = silhouette_score(scaled_features, labels)
            if score > best_score:
                best_score = score
                best_labels = labels
                best_n_clusters = n_clusters
                best_model_name = model_name

    customer_data['Cluster_ID'] = best_labels
    print(f"  Selected customer clustering: {best_model_name} with {best_n_clusters} clusters (Silhouette Score: {best_score:.4f})")
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
        return f1_score(y_test, preds, average='macro', zero_division=0)
    
    persona_results = []

    lr_base = LogisticRegression(random_state=42, max_iter=2000, class_weight='balanced')
    param_grid_lr = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
    lr_tuned = RandomizedSearchCV(lr_base, param_grid_lr, n_iter=6, cv=3, random_state=42, scoring='f1_macro', n_jobs=-1)
    lr_tuned.fit(X_train_scaled, y_train)
    lr_f1 = print_persona_metrics("Tuned Logistic Regression", lr_tuned.predict(X_test_scaled), lr_tuned.best_params_)
    persona_results.append({'name': 'Tuned Logistic Regression', 'model': lr_tuned, 'scaler': scaler, 'use_scaled': True, 'score': lr_f1})
    
    svm_base = SVC(class_weight='balanced', random_state=42)
    param_grid_svm = {'C': [0.1, 1, 10, 50], 'gamma': ['scale', 'auto', 0.1, 0.01], 'kernel': ['rbf']}
    svm_tuned = RandomizedSearchCV(svm_base, param_grid_svm, n_iter=8, cv=3, random_state=42, scoring='f1_macro', n_jobs=-1)
    svm_tuned.fit(X_train_scaled, y_train)
    svm_f1 = print_persona_metrics("Tuned SVM (RBF)", svm_tuned.predict(X_test_scaled), svm_tuned.best_params_)
    persona_results.append({'name': 'Tuned SVM (RBF)', 'model': svm_tuned, 'scaler': scaler, 'use_scaled': True, 'score': svm_f1})
    
    rf_base = RandomForestClassifier(class_weight='balanced', random_state=42)
    param_grid_rf = {'n_estimators': [100, 200], 'max_depth': [None, 5, 10], 'min_samples_split': [2, 5]}
    rf_tuned = RandomizedSearchCV(rf_base, param_grid_rf, n_iter=8, cv=3, random_state=42, scoring='f1_macro', n_jobs=-1)
    rf_tuned.fit(X_train, y_train)
    rf_preds = rf_tuned.predict(X_test)
    rf_f1 = print_persona_metrics("Tuned Random Forest", rf_preds, rf_tuned.best_params_)
    persona_results.append({'name': 'Tuned Random Forest', 'model': rf_tuned, 'scaler': None, 'use_scaled': False, 'score': rf_f1})

    best_persona = max(persona_results, key=lambda x: x['score'])
    print(f"\nBest persona classifier: {best_persona['name']} (Macro F1: {best_persona['score']:.2%})")
    return best_persona, feature_cols

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

    churn_results = []

    lr_base = LogisticRegression(random_state=42, max_iter=2000)
    param_grid_lr = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'penalty': ['l2']}
    lr_tuned = RandomizedSearchCV(lr_base, param_grid_lr, n_iter=6, cv=3, random_state=42, scoring='f1', n_jobs=-1)
    lr_tuned.fit(X_train_scaled, y_train)
    lr_preds = lr_tuned.predict(X_test_scaled)
    lr_probs = lr_tuned.predict_proba(X_test_scaled)[:, 1]
    print_metrics("Tuned Logistic Regression", y_test, lr_preds, lr_probs, lr_tuned.best_params_)
    churn_results.append({'name': 'Tuned Logistic Regression', 'model': lr_tuned, 'scaler': scaler, 'use_scaled': True, 'score': f1_score(y_test, lr_preds, zero_division=0)})
    
    svm_base = SVC(probability=True, random_state=42, kernel='rbf')
    param_grid_svm = {'C': [0.1, 1, 10, 50], 'gamma': ['scale', 'auto', 0.1, 0.01]}
    svm_tuned = RandomizedSearchCV(svm_base, param_grid_svm, n_iter=8, cv=3, random_state=42, scoring='f1', n_jobs=-1)
    svm_tuned.fit(X_train_scaled, y_train)
    svm_preds = svm_tuned.predict(X_test_scaled)
    svm_probs = svm_tuned.predict_proba(X_test_scaled)[:, 1]
    print_metrics("Tuned SVM (RBF)", y_test, svm_preds, svm_probs, svm_tuned.best_params_)
    churn_results.append({'name': 'Tuned SVM (RBF)', 'model': svm_tuned, 'scaler': scaler, 'use_scaled': True, 'score': f1_score(y_test, svm_preds, zero_division=0)})
    
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
    churn_results.append({'name': 'Tuned Random Forest', 'model': rf_tuned, 'scaler': None, 'use_scaled': False, 'score': f1_score(y_test, rf_preds, zero_division=0)})

    best_churn = max(churn_results, key=lambda x: x['score'])
    print(f"\nBest churn model: {best_churn['name']} (F1: {best_churn['score']:.2%})")
    return best_churn, feature_cols
    
    churn_results = []

    lr_base = LogisticRegression(random_state=42, max_iter=2000)
    param_grid_lr = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'penalty': ['l2']}
    lr_tuned = RandomizedSearchCV(lr_base, param_grid_lr, n_iter=6, cv=3, random_state=42, scoring='f1', n_jobs=-1)
    lr_tuned.fit(X_train_scaled, y_train)
    lr_preds = lr_tuned.predict(X_test_scaled)
    lr_probs = lr_tuned.predict_proba(X_test_scaled)[:, 1]
    print_metrics("Tuned Logistic Regression", y_test, lr_preds, lr_probs, lr_tuned.best_params_)
    churn_results.append({'name': 'Tuned Logistic Regression', 'model': lr_tuned, 'scaler': scaler, 'use_scaled': True, 'score': f1_score(y_test, lr_preds, zero_division=0)})
    
    svm_base = SVC(probability=True, random_state=42, kernel='rbf')
    param_grid_svm = {'C': [0.1, 1, 10, 50], 'gamma': ['scale', 'auto', 0.1, 0.01]}
    svm_tuned = RandomizedSearchCV(svm_base, param_grid_svm, n_iter=8, cv=3, random_state=42, scoring='f1', n_jobs=-1)
    svm_tuned.fit(X_train_scaled, y_train)
    svm_preds = svm_tuned.predict(X_test_scaled)
    svm_probs = svm_tuned.predict_proba(X_test_scaled)[:, 1]
    print_metrics("Tuned SVM (RBF)", y_test, svm_preds, svm_probs, svm_tuned.best_params_)
    churn_results.append({'name': 'Tuned SVM (RBF)', 'model': svm_tuned, 'scaler': scaler, 'use_scaled': True, 'score': f1_score(y_test, svm_preds, zero_division=0)})
    
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
    churn_results.append({'name': 'Tuned Random Forest', 'model': rf_tuned, 'scaler': None, 'use_scaled': False, 'score': f1_score(y_test, rf_preds, zero_division=0)})

    best_churn = max(churn_results, key=lambda x: x['score'])
    print(f"\nBest churn model: {best_churn['name']} (F1: {best_churn['score']:.2%})")
    return best_churn, feature_cols

def predict_model_probability(model_info, X):
    X_input = X.copy()
    if model_info['use_scaled']:
        X_input = model_info['scaler'].transform(X_input)
    if hasattr(model_info['model'], 'predict_proba'):
        return model_info['model'].predict_proba(X_input)[:, 1]
    if hasattr(model_info['model'], 'decision_function'):
        scores = model_info['model'].decision_function(X_input)
        return (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
    return model_info['model'].predict(X_input)


def build_business_insight_tables(df, products_df, customer_data, churn_model_info, churn_feature_cols):
    product_sales = df.groupby('product_id').agg(
        Total_Quantity_Sold=('quantity', 'sum'),
        Total_Revenue=('total_price', 'sum')
    ).reset_index()
    product_sales = pd.merge(product_sales, products_df.rename(columns={'quantity': 'Stock'}), on='product_id', how='right')
    product_sales['Total_Quantity_Sold'] = product_sales['Total_Quantity_Sold'].fillna(0)
    product_sales['Total_Revenue'] = product_sales['Total_Revenue'].fillna(0)
    product_sales['Sell_Through'] = product_sales['Total_Quantity_Sold'] / (product_sales['Stock'].replace(0, np.nan))
    product_sales['Sell_Through'] = product_sales['Sell_Through'].fillna(0)

    best_products = product_sales.sort_values('Total_Revenue', ascending=False).head(5).copy()
    attention_products = product_sales[product_sales['Stock'] > 0].sort_values(['Total_Revenue', 'Total_Quantity_Sold']).head(5).copy()

    customer_business = customer_data.copy()
    if 'customer_name' not in customer_business.columns:
        customer_business = pd.merge(customer_business, df[['customer_id', 'customer_name']].drop_duplicates(), on='customer_id', how='left')
    customer_business['Predicted_Churn_Prob'] = predict_model_probability(churn_model_info, customer_business[churn_feature_cols])
    customer_business['Predicted_Churn_Label'] = (customer_business['Predicted_Churn_Prob'] >= 0.5).astype(int)

    top_customers = customer_business.sort_values('Monetary', ascending=False).head(5).copy()
    risky_customers = customer_business[customer_business['Predicted_Churn_Prob'] >= 0.6].sort_values(['Predicted_Churn_Prob', 'Monetary'], ascending=[False, False]).head(5).copy()

    cluster_summary = customer_business.groupby('Cluster_ID').agg(
        Customers=('customer_id', 'count'),
        Avg_Monetary=('Monetary', 'mean'),
        Avg_Future_Spend=('Future_Spend', 'mean'),
        Avg_Predicted_Churn=('Predicted_Churn_Prob', 'mean'),
        Actual_Churn_Rate=('Churn', 'mean')
    ).reset_index().sort_values(['Avg_Future_Spend', 'Avg_Monetary'], ascending=False)

    age_bins = [0, 25, 35, 45, 55, 100]
    customer_business['Age_Band'] = pd.cut(customer_business['Age'], age_bins)
    age_summary = customer_business.groupby('Age_Band').agg(
        Customers=('customer_id', 'count'),
        Avg_Monetary=('Monetary', 'mean'),
        Churn_Rate=('Churn', 'mean')
    ).reset_index().sort_values('Avg_Monetary', ascending=False)

    summary = {
        'customer_count': int(customer_business['customer_id'].nunique()),
        'churn_rate': float(customer_business['Churn'].mean()),
        'cluster_count': int(customer_business['Cluster_ID'].nunique()),
        'average_monetary': float(customer_business['Monetary'].mean()),
        'high_risk_count': int((customer_business['Predicted_Churn_Prob'] >= 0.7).sum()),
        'low_risk_count': int((customer_business['Predicted_Churn_Prob'] <= 0.3).sum()),
    }

    return {
        'summary': pd.DataFrame([summary]),
        'customers': top_customers.reset_index(drop=True),
        'risky_customers': risky_customers.reset_index(drop=True),
        'clusters': cluster_summary.reset_index(drop=True),
        'products': best_products.reset_index(drop=True),
        'attention_products': attention_products.reset_index(drop=True),
        'age_segments': age_summary.reset_index(drop=True),
    }


def print_business_insights(df, products_df, customer_data, churn_model_info, churn_feature_cols):
    print("\n--- 3. BUSINESS INSIGHTS ---")
    insight_tables = build_business_insight_tables(df, products_df, customer_data, churn_model_info, churn_feature_cols)
    summary = insight_tables['summary'].iloc[0]
    top_customers = insight_tables['customers']
    risky_customers = insight_tables['risky_customers']
    cluster_summary = insight_tables['clusters']
    best_products = insight_tables['products']
    attention_products = insight_tables['attention_products']
    age_summary = insight_tables['age_segments']

    print("\nTop 5 customers by historical revenue:")
    for _, row in top_customers.iterrows():
        print(f"  {row.get('Customer_Name', row.get('customer_name','ID '+str(int(row['customer_id']))))}: ${row['Monetary']:.2f} historical spend, churn prob {row['Predicted_Churn_Prob']:.2%}")

    if not risky_customers.empty:
        print("\nPotentially harmful customers (high churn probability + high spend):")
        for _, row in risky_customers.iterrows():
            print(f"  {row.get('Customer_Name', row.get('customer_name','ID '+str(int(row['customer_id']))))}: ${row['Monetary']:.2f} spend, churn prob {row['Predicted_Churn_Prob']:.2%}")
    else:
        print("\nNo high-risk revenue customers were detected above the 60% churn-probability threshold.")

    best_cluster = cluster_summary.iloc[0]
    print(f"\nBest customer group for business: Cluster {int(best_cluster['Cluster_ID'])}")
    print(f"  Customers: {int(best_cluster['Customers'])}")
    print(f"  Avg Monetary Spend: ${best_cluster['Avg_Monetary']:.2f}")
    print(f"  Avg Future Spend: ${best_cluster['Avg_Future_Spend']:.2f}")
    print(f"  Avg Predicted Churn Probability: {best_cluster['Avg_Predicted_Churn']:.2%}")
    print(f"  Actual Churn Rate: {best_cluster['Actual_Churn_Rate']:.2%}")

    print("\nTop 5 product sellers by revenue:")
    for _, row in best_products.iterrows():
        print(f"  {row['product_name']} ({row['product_type']}): ${row['Total_Revenue']:.2f} revenue, {int(row['Total_Quantity_Sold'])} units sold")

    print("\nProducts needing attention (low sales, stock remaining):")
    for _, row in attention_products.iterrows():
        print(f"  {row['product_name']} ({row['product_type']}): ${row['Total_Revenue']:.2f} revenue, {int(row['Total_Quantity_Sold'])} sold, {int(row['Stock'])} stock")

    top_age = age_summary.iloc[0]
    print(f"\nHighest value age segment: {top_age['Age_Band']} with avg spend ${top_age['Avg_Monetary']:.2f} and churn {top_age['Churn_Rate']:.2%}")

    print(f"\nPredicted at-risk segment: {int(summary['high_risk_count'])} customers with churn probability >= 70%.")
    print(f"Predicted safest segment: {int(summary['low_risk_count'])} customers with churn probability <= 30%.")


def save_business_insight_excel(output_path, insight_tables):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet_names = {
        'summary': 'Summary',
        'customers': 'Customers',
        'risky_customers': 'Risky Customers',
        'clusters': 'Clusters',
        'products': 'Products',
        'attention_products': 'Attention Products',
        'age_segments': 'Age Segments',
    }
    with pd.ExcelWriter(output_path) as writer:
        for key, dataframe in insight_tables.items():
            dataframe.to_excel(writer, sheet_name=sheet_names[key], index=False)


def run_analysis_pipeline(output_path=None):
    df = load_and_merge_data()
    products_df = pd.read_csv(_data_path('products 2.csv'))
    customer_data = advanced_feature_engineering(df)
    customer_data = apply_customer_kmeans(customer_data)

    _, _ = train_persona_classifier(customer_data)
    churn_model_info, churn_feature_cols = train_churn_predictor(customer_data)
    insight_tables = build_business_insight_tables(df, products_df, customer_data, churn_model_info, churn_feature_cols)

    if output_path:
        save_business_insight_excel(output_path, insight_tables)

    summary = insight_tables['summary'].iloc[0].to_dict()
    return {
        'summary': {
            'customer_count': int(summary['customer_count']),
            'churn_rate': float(summary['churn_rate']),
            'cluster_count': int(summary['cluster_count']),
            'average_monetary': float(summary['average_monetary']),
            'high_risk_count': int(summary['high_risk_count']),
            'low_risk_count': int(summary['low_risk_count']),
        },
        'customers': insight_tables['customers'].to_dict(orient='records'),
        'risky_customers': insight_tables['risky_customers'].to_dict(orient='records'),
        'clusters': insight_tables['clusters'].to_dict(orient='records'),
        'products': insight_tables['products'].to_dict(orient='records'),
        'attention_products': insight_tables['attention_products'].to_dict(orient='records'),
        'age_segments': insight_tables['age_segments'].to_dict(orient='records'),
    }

if __name__ == "__main__":
    payload = run_analysis_pipeline(output_path=_data_path('analysis_results.xlsx'))
    print(payload['summary'])
