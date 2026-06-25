import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

def feature_engineering(df):
    print("Performing advanced feature engineering...")
    max_date = df['order_date'].max()
    cutoff_date = max_date - pd.Timedelta(days=90)
    
    pre_df = df[df['order_date'] <= cutoff_date]
    post_df = df[df['order_date'] > cutoff_date]
    
    order_totals = pre_df.groupby(['customer_id', 'order_id'])['total_price'].sum().reset_index()
    max_order_values = order_totals.groupby('customer_id')['total_price'].max().reset_index(name='Max_Order_Value')
    
    momentum_cutoff = cutoff_date - pd.Timedelta(days=30)
    momentum_df = pre_df[pre_df['order_date'] > momentum_cutoff]
    momentum = momentum_df.groupby('customer_id')['total_price'].sum().reset_index(name='Momentum_30d_Spend')
    
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
    
    customer_features = pd.merge(customer_features, max_order_values, on='customer_id', how='left')
    customer_features = pd.merge(customer_features, momentum, on='customer_id', how='left')
    customer_features['Momentum_30d_Spend'] = customer_features['Momentum_30d_Spend'].fillna(0)
    
    customer_features['AOV'] = customer_features['Monetary'] / customer_features['Frequency']
    customer_features['Velocity'] = customer_features['Tenure'] / customer_features['Frequency']
    
    customer_features = pd.get_dummies(customer_features, columns=['Gender', 'State'], drop_first=True)
    
    post_behavior = post_df.groupby('customer_id').agg(Future_Spend=('total_price', 'sum')).reset_index()
    customer_data = pd.merge(customer_features, post_behavior, on='customer_id', how='left')
    customer_data['Future_Spend'] = customer_data['Future_Spend'].fillna(0)
    
    p95 = np.percentile(customer_data[customer_data['Future_Spend'] > 0]['Future_Spend'], 95)
    customer_data['Future_Spend_Clipped'] = np.clip(customer_data['Future_Spend'], 0, p95)
    
    customer_data['Churn'] = (customer_data['Future_Spend'] == 0).astype(int)
    
    return customer_data

def apply_kmeans(customer_data):
    cluster_features = customer_data[['Recency', 'Frequency', 'Monetary', 'Age']]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(cluster_features)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    customer_data['Cluster_ID'] = kmeans.fit_predict(scaled_features)
    return customer_data

def product_engineering(df):
    product_features = df.groupby('product_id').agg(
        Total_Quantity_Sold=('quantity', 'sum'),
        Total_Revenue=('total_price', 'sum'),
        Unique_Orders=('order_id', 'nunique'),
        Unique_Customers=('customer_id', 'nunique'),
        Price_Per_Unit=('price_per_unit', 'mean')
    ).reset_index()
    return product_features

def apply_product_kmeans(product_data):
    features = product_data[['Total_Quantity_Sold', 'Total_Revenue', 'Unique_Orders', 'Price_Per_Unit']]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    product_data['Product_Cluster_ID'] = kmeans.fit_predict(scaled)
    return product_data

def order_engineering(df):
    order_features = df.groupby('order_id').agg(
        Cart_Total_Value=('total_price', 'sum'),
        Total_Items=('quantity', 'sum'),
        Unique_Products=('product_id', 'nunique')
    ).reset_index()
    return order_features

def apply_order_kmeans(order_data):
    features = order_data[['Cart_Total_Value', 'Total_Items', 'Unique_Products']]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    order_data['Order_Cluster_ID'] = kmeans.fit_predict(scaled)
    return order_data

def load_and_merge_data():
    print("Loading data...")
    customers = pd.read_csv('customers 2.csv')
    orders = pd.read_csv('orders 2.csv')
    sales = pd.read_csv('sales 2.csv')
    
    df = pd.merge(sales, orders, on='order_id', how='inner')
    df = pd.merge(df, customers, on='customer_id', how='inner')
    df['order_date'] = pd.to_datetime(df['order_date'])
    return df

df = load_and_merge_data()
df.to_csv('merged_dataset.csv', index=False)
print("Saved merged_dataset.csv successfully.")

customer_data = feature_engineering(df)
customer_data = apply_kmeans(customer_data)

product_data = product_engineering(df)
product_data = apply_product_kmeans(product_data)

order_data = order_engineering(df)
order_data = apply_order_kmeans(order_data)

print("\n--- CUSTOMER PERSONAS (K-MEANS CLUSTERING) ---\n")
customers = pd.read_csv('customers 2.csv')
merged = pd.merge(customer_data[['customer_id', 'Cluster_ID', 'Recency', 'Frequency', 'Monetary', 'Age']], customers, on='customer_id', how='left')

max_date = df['order_date'].max()
cutoff_date = max_date - pd.Timedelta(days=90)
pre_df = df[df['order_date'] <= cutoff_date]
transactions = pd.merge(pre_df, merged[['customer_id', 'Cluster_ID']], on='customer_id', how='left')

for cluster in sorted(merged['Cluster_ID'].unique()):
    cluster_df = merged[merged['Cluster_ID'] == cluster]
    size = len(cluster_df)
    pct = (size / len(merged)) * 100
    
    avg_spend = cluster_df['Monetary'].mean()
    avg_freq = cluster_df['Frequency'].mean()
    avg_age = cluster_df['Age'].mean()
    
    top_state = cluster_df['state'].mode()[0] if not cluster_df['state'].mode().empty else "Unknown"
    top_gender = cluster_df['gender'].mode()[0] if not cluster_df['gender'].mode().empty else "Unknown"
    
    cluster_tx = transactions[transactions['Cluster_ID'] == cluster]
    if not cluster_tx.empty:
        top_products = cluster_tx['product_id'].value_counts().head(3).index.tolist()
    else:
        top_products = []
        
    print(f"CUSTOMER CLUSTER {cluster} ({size} customers, {pct:.1f}% of base)")
    print(f"  - Avg Spend:     ${avg_spend:.2f}")
    print(f"  - Avg Frequency: {avg_freq:.2f} orders")
    print(f"  - Avg Age:       {avg_age:.1f} years old")
    print(f"  - Top State:     {top_state}")
    print(f"  - Top Gender:    {top_gender}")
    print(f"  - Top Products:  {top_products}\n")

print("--- PRODUCT CATEGORIES (K-MEANS CLUSTERING) ---\n")
for cluster in sorted(product_data['Product_Cluster_ID'].unique()):
    cluster_df = product_data[product_data['Product_Cluster_ID'] == cluster]
    size = len(cluster_df)
    pct = (size / len(product_data)) * 100
    
    avg_qty = cluster_df['Total_Quantity_Sold'].mean()
    avg_rev = cluster_df['Total_Revenue'].mean()
    avg_price = cluster_df['Price_Per_Unit'].mean()
    avg_orders = cluster_df['Unique_Orders'].mean()
    
    print(f"PRODUCT CLUSTER {cluster} ({size} products, {pct:.1f}% of catalog)")
    print(f"  - Avg Total Revenue:   ${avg_rev:.2f}")
    print(f"  - Avg Total Qty Sold:  {avg_qty:.2f} units")
    print(f"  - Avg Price Per Unit:  ${avg_price:.2f}")
    print(f"  - Avg Unique Orders:   {avg_orders:.2f} orders\n")

print("--- ORDER TYPES (K-MEANS CLUSTERING) ---\n")
for cluster in sorted(order_data['Order_Cluster_ID'].unique()):
    cluster_df = order_data[order_data['Order_Cluster_ID'] == cluster]
    size = len(cluster_df)
    pct = (size / len(order_data)) * 100
    
    avg_value = cluster_df['Cart_Total_Value'].mean()
    avg_items = cluster_df['Total_Items'].mean()
    avg_unique_prods = cluster_df['Unique_Products'].mean()
    
    print(f"ORDER CLUSTER {cluster} ({size} orders, {pct:.1f}% of total orders)")
    print(f"  - Avg Cart Value:      ${avg_value:.2f}")
    print(f"  - Avg Total Items:     {avg_items:.2f} items")
    print(f"  - Avg Unique Products: {avg_unique_prods:.2f} distinct items\n")

def train_persona_classifier(customer_data):
    print("\n--- 1. PERSONA CLASSIFIER (Advanced) ---")
    
    feature_cols = ['Age', 'Monetary'] + [col for col in customer_data.columns if col.startswith('Gender_') or col.startswith('State_')]
    
    X = customer_data[feature_cols]
    y = customer_data['Cluster_ID']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    acc_rf = accuracy_score(y_test, rf.predict(X_test))
    
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }
    rf_tuned = RandomizedSearchCV(RandomForestClassifier(random_state=42), param_grid, n_iter=5, cv=3, random_state=42, n_jobs=-1)
    rf_tuned.fit(X_train, y_train)
    acc_rf_tuned = accuracy_score(y_test, rf_tuned.predict(X_test))
    
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    xgb.fit(X_train, y_train)
    acc_xgb = accuracy_score(y_test, xgb.predict(X_test))
    
    print("Features used: Age, Gender, State, and Initial Spend (Monetary)")
    print(f"  -> Baseline Random Forest: {acc_rf:.2%}")
    print(f"  -> Tuned Random Forest:    {acc_rf_tuned:.2%}")
    print(f"  -> XGBoost Classifier:     {acc_xgb:.2%}")

def train_churn_predictor(customer_data):
    print("\n--- 2. CHURN PREDICTOR (Advanced) ---")
    feature_cols = [
        'Recency', 'Tenure', 'Frequency', 'Monetary', 'Distinct_Products', 
        'Age', 'Momentum_30d_Spend', 'AOV', 'Velocity', 'Cluster_ID'
    ] + [col for col in customer_data.columns if col.startswith('Gender_') or col.startswith('State_')]
    
    customer_data = customer_data.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    X = customer_data[feature_cols]
    y = customer_data['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    acc_rf = accuracy_score(y_test, rf.predict(X_test))
    
    param_grid = {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [None, 5, 10, 15],
        'min_samples_split': [2, 5, 10]
    }
    rf_tuned = RandomizedSearchCV(RandomForestClassifier(random_state=42), param_grid, n_iter=10, cv=3, random_state=42, n_jobs=-1)
    rf_tuned.fit(X_train, y_train)
    acc_rf_tuned = accuracy_score(y_test, rf_tuned.predict(X_test))
    
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    xgb.fit(X_train, y_train)
    acc_xgb = accuracy_score(y_test, xgb.predict(X_test))
    
    print("Predicting Churn WITH Persona (Cluster_ID)")
    print(f"  -> Baseline Random Forest: {acc_rf:.2%}")
    print(f"  -> Tuned Random Forest:    {acc_rf_tuned:.2%}")
    print(f"  -> XGBoost Classifier:     {acc_xgb:.2%}")

train_persona_classifier(customer_data)
train_churn_predictor(customer_data)
