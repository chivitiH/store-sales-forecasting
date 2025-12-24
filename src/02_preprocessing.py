"""
Store Sales Forecasting - Data Preprocessing & Feature Engineering
Prépare les données pour le modeling
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Paths
DATA_PATH = Path("data/raw")
PROCESSED_PATH = Path("data/processed")
PROCESSED_PATH.mkdir(parents=True, exist_ok=True)

print("="*60)
print("🔧 DATA PREPROCESSING & FEATURE ENGINEERING")
print("="*60)

# ============================================
# 1. LOAD DATA
# ============================================
print("\n📦 Chargement des données...")

train = pd.read_csv(DATA_PATH / "train.csv", parse_dates=['date'])
test = pd.read_csv(DATA_PATH / "test.csv", parse_dates=['date'])
stores = pd.read_csv(DATA_PATH / "stores.csv")
oil = pd.read_csv(DATA_PATH / "oil.csv", parse_dates=['date'])
holidays = pd.read_csv(DATA_PATH / "holidays_events.csv", parse_dates=['date'])
transactions = pd.read_csv(DATA_PATH / "transactions.csv", parse_dates=['date'])

print(f"✅ Données chargées")

# ============================================
# 2. HANDLE MISSING VALUES
# ============================================
print("\n" + "="*60)
print("🧹 NETTOYAGE DES DONNÉES")
print("="*60)

# Oil prices - Forward fill puis backward fill
print("\n📊 Traitement des valeurs manquantes du pétrole...")
print(f"   Avant: {oil['dcoilwtico'].isnull().sum()} NaN")
oil['dcoilwtico'] = oil['dcoilwtico'].fillna(method='ffill').fillna(method='bfill')
print(f"   Après: {oil['dcoilwtico'].isnull().sum()} NaN")

# ============================================
# 3. MERGE ALL TABLES
# ============================================
print("\n" + "="*60)
print("🔗 MERGE DES TABLES")
print("="*60)

# Merge train with stores
print("\n1️⃣ Train + Stores...")
df = train.merge(stores, on='store_nbr', how='left')
print(f"   Shape: {df.shape}")

# Merge with oil
print("\n2️⃣ + Oil prices...")
df = df.merge(oil, on='date', how='left')
print(f"   Shape: {df.shape}")

# Merge with transactions
print("\n3️⃣ + Transactions...")
df = df.merge(transactions, on=['date', 'store_nbr'], how='left')
print(f"   Shape: {df.shape}")

# Process holidays - créer flag binaire
print("\n4️⃣ + Holidays...")
holidays_national = holidays[holidays['locale'] == 'National'][['date', 'type']].copy()
holidays_national['is_holiday'] = 1
holidays_national = holidays_national.drop_duplicates('date')
df = df.merge(holidays_national[['date', 'is_holiday']], on='date', how='left')
df['is_holiday'] = df['is_holiday'].fillna(0).astype(int)
print(f"   Shape: {df.shape}")

print(f"\n✅ Dataset final mergé: {df.shape}")

# ============================================
# 4. TEMPORAL FEATURES
# ============================================
print("\n" + "="*60)
print("📅 FEATURE ENGINEERING - TEMPORAL")
print("="*60)

print("\n🕐 Création features temporelles...")
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['dayofweek'] = df['date'].dt.dayofweek
df['dayofyear'] = df['date'].dt.dayofyear
df['weekofyear'] = df['date'].dt.isocalendar().week.astype(int)
df['quarter'] = df['date'].dt.quarter
df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
df['is_month_end'] = df['date'].dt.is_month_end.astype(int)

# Payday features (15th and end of month)
df['is_payday'] = ((df['day'] == 15) | (df['is_month_end'] == 1)).astype(int)

print(f"   ✅ {11} features temporelles créées")

# ============================================
# 5. LAG FEATURES
# ============================================
print("\n" + "="*60)
print("🔄 FEATURE ENGINEERING - LAG FEATURES")
print("="*60)

print("\n⏮️ Création lag features (par store + family)...")

# Sort by date
df = df.sort_values(['store_nbr', 'family', 'date']).reset_index(drop=True)

# Lag features: 1, 7, 14, 30 days
lags = [1, 7, 14, 30]
for lag in lags:
    print(f"   Lag {lag} jours...")
    df[f'sales_lag_{lag}'] = df.groupby(['store_nbr', 'family'])['sales'].shift(lag)

print(f"   ✅ {len(lags)} lag features créées")

# ============================================
# 6. ROLLING FEATURES
# ============================================
print("\n" + "="*60)
print("📊 FEATURE ENGINEERING - ROLLING FEATURES")
print("="*60)

print("\n📈 Création rolling means...")

windows = [7, 14, 30]
for window in windows:
    print(f"   Rolling mean {window} jours...")
    df[f'sales_rolling_mean_{window}'] = df.groupby(['store_nbr', 'family'])['sales'].transform(
        lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
    )
    
    print(f"   Rolling std {window} jours...")
    df[f'sales_rolling_std_{window}'] = df.groupby(['store_nbr', 'family'])['sales'].transform(
        lambda x: x.shift(1).rolling(window=window, min_periods=1).std()
    )

print(f"   ✅ {len(windows)*2} rolling features créées")

# ============================================
# 7. PROMOTION FEATURES
# ============================================
print("\n" + "="*60)
print("🎯 FEATURE ENGINEERING - PROMOTIONS")
print("="*60)

print("\n🛍️ Création features promotions...")

# Binary promotion flag
df['has_promotion'] = (df['onpromotion'] > 0).astype(int)

# Promotion intensity
df['promo_intensity'] = df['onpromotion']

print(f"   ✅ 2 promotion features créées")

# ============================================
# 8. CATEGORICAL ENCODING
# ============================================
print("\n" + "="*60)
print("🏷️ ENCODING CATÉGORIELLES")
print("="*60)

print("\n📋 Label encoding...")

from sklearn.preprocessing import LabelEncoder

categorical_cols = ['family', 'city', 'state', 'type']
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[f'{col}_encoded'] = le.fit_transform(df[col])
    label_encoders[col] = le
    print(f"   ✅ {col}: {len(le.classes_)} classes")

# ============================================
# 9. HANDLE REMAINING MISSING VALUES
# ============================================
print("\n" + "="*60)
print("🧹 GESTION FINALES DES NaN")
print("="*60)

print("\nNaN avant remplissage:")
nan_cols = df.isnull().sum()
nan_cols = nan_cols[nan_cols > 0]
if len(nan_cols) > 0:
    print(nan_cols)
    
    # Fill transactions NaN with 0
    if 'transactions' in df.columns:
        df['transactions'] = df['transactions'].fillna(0)
    
    # Fill lag/rolling features NaN with 0 (début de série)
    lag_rolling_cols = [col for col in df.columns if 'lag' in col or 'rolling' in col]
    for col in lag_rolling_cols:
        df[col] = df[col].fillna(0)
    
    print("\nNaN après remplissage:")
    nan_cols_after = df.isnull().sum()
    nan_cols_after = nan_cols_after[nan_cols_after > 0]
    if len(nan_cols_after) > 0:
        print(nan_cols_after)
    else:
        print("   ✅ Plus de NaN !")
else:
    print("   ✅ Pas de NaN détecté")

# ============================================
# 10. PREPARE TEST SET
# ============================================
print("\n" + "="*60)
print("🧪 PRÉPARATION TEST SET")
print("="*60)

print("\n📊 Application des mêmes transformations au test set...")

# Merge test with stores
test_df = test.merge(stores, on='store_nbr', how='left')
test_df = test_df.merge(oil, on='date', how='left')
test_df = test_df.merge(transactions, on=['date', 'store_nbr'], how='left')
test_df = test_df.merge(holidays_national[['date', 'is_holiday']], on='date', how='left')
test_df['is_holiday'] = test_df['is_holiday'].fillna(0).astype(int)

# Temporal features
test_df['year'] = test_df['date'].dt.year
test_df['month'] = test_df['date'].dt.month
test_df['day'] = test_df['date'].dt.day
test_df['dayofweek'] = test_df['date'].dt.dayofweek
test_df['dayofyear'] = test_df['date'].dt.dayofyear
test_df['weekofyear'] = test_df['date'].dt.isocalendar().week.astype(int)
test_df['quarter'] = test_df['date'].dt.quarter
test_df['is_weekend'] = (test_df['dayofweek'] >= 5).astype(int)
test_df['is_month_start'] = test_df['date'].dt.is_month_start.astype(int)
test_df['is_month_end'] = test_df['date'].dt.is_month_end.astype(int)
test_df['is_payday'] = ((test_df['day'] == 15) | (test_df['is_month_end'] == 1)).astype(int)

# Categorical encoding
for col in categorical_cols:
    test_df[f'{col}_encoded'] = label_encoders[col].transform(test_df[col])

# Promotion features
test_df['has_promotion'] = (test_df['onpromotion'] > 0).astype(int)
test_df['promo_intensity'] = test_df['onpromotion']

# Pour le test, on doit calculer les lags à partir des dernières valeurs du train
# Créer un dataset combiné temporaire
combined = pd.concat([
    df[['date', 'store_nbr', 'family', 'sales']],
    test_df[['date', 'store_nbr', 'family']].assign(sales=np.nan)
], ignore_index=True).sort_values(['store_nbr', 'family', 'date'])

# Recalculer les lags sur le dataset combiné
for lag in lags:
    combined[f'sales_lag_{lag}'] = combined.groupby(['store_nbr', 'family'])['sales'].shift(lag)

for window in windows:
    combined[f'sales_rolling_mean_{window}'] = combined.groupby(['store_nbr', 'family'])['sales'].transform(
        lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
    )
    combined[f'sales_rolling_std_{window}'] = combined.groupby(['store_nbr', 'family'])['sales'].transform(
        lambda x: x.shift(1).rolling(window=window, min_periods=1).std()
    )

# Extraire les lags/rolling du test
test_lag_cols = [col for col in combined.columns if 'lag' in col or 'rolling' in col]
test_lags = combined[combined['sales'].isna()][['date', 'store_nbr', 'family'] + test_lag_cols]

# Merger avec test_df
test_df = test_df.merge(test_lags, on=['date', 'store_nbr', 'family'], how='left')

# Fill NaN
test_df['transactions'] = test_df['transactions'].fillna(0)
for col in test_lag_cols:
    test_df[col] = test_df[col].fillna(0)

print(f"   ✅ Test set préparé: {test_df.shape}")

# ============================================
# 11. SAVE PROCESSED DATA
# ============================================
print("\n" + "="*60)
print("💾 SAUVEGARDE DES DONNÉES")
print("="*60)

print("\n📁 Sauvegarde...")
df.to_csv(PROCESSED_PATH / "train_processed.csv", index=False)
print(f"   ✅ train_processed.csv ({df.shape})")

test_df.to_csv(PROCESSED_PATH / "test_processed.csv", index=False)
print(f"   ✅ test_processed.csv ({test_df.shape})")

# Save a sample for quick testing
sample_df = df.sample(n=min(100000, len(df)), random_state=42)
sample_df.to_csv(PROCESSED_PATH / "train_sample.csv", index=False)
print(f"   ✅ train_sample.csv ({sample_df.shape}) - pour tests rapides")

# ============================================
# 12. FEATURE SUMMARY
# ============================================
print("\n" + "="*60)
print("📊 RÉSUMÉ DES FEATURES")
print("="*60)

all_features = df.columns.tolist()
print(f"\n✅ Total features: {len(all_features)}")

feature_groups = {
    'Original': ['id', 'date', 'store_nbr', 'family', 'sales', 'onpromotion'],
    'Store Info': ['city', 'state', 'type', 'cluster'],
    'External': ['dcoilwtico', 'transactions', 'is_holiday'],
    'Temporal': [c for c in all_features if c in ['year', 'month', 'day', 'dayofweek', 'dayofyear', 
                                                    'weekofyear', 'quarter', 'is_weekend', 
                                                    'is_month_start', 'is_month_end', 'is_payday']],
    'Lag': [c for c in all_features if 'lag' in c],
    'Rolling': [c for c in all_features if 'rolling' in c],
    'Promotion': [c for c in all_features if 'promo' in c or c == 'has_promotion'],
    'Encoded': [c for c in all_features if 'encoded' in c]
}

for group, features in feature_groups.items():
    print(f"\n{group}: {len(features)} features")
    if len(features) <= 15:
        for f in features:
            if f in all_features:
                print(f"   - {f}")

print("\n" + "="*60)
print("✅ PREPROCESSING TERMINÉ !")
print("="*60)
