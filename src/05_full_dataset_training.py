"""
Store Sales Forecasting - Full Dataset Training
Applique les meilleurs modèles sur le dataset complet
"""

import pandas as pd
import numpy as np
import time
from pathlib import Path
from datetime import datetime
import joblib
import warnings
warnings.filterwarnings('ignore')

# Paths
PROCESSED_PATH = Path("data/processed")
MODELS_PATH = Path("models")
PREDICTIONS_PATH = Path("data/predictions")
PREDICTIONS_PATH.mkdir(parents=True, exist_ok=True)

print("="*60)
print("🚀 FULL DATASET TRAINING")
print("="*60)
print(f"⏰ Début: {datetime.now().strftime('%H:%M:%S')}")

# ============================================
# 1. LOAD DATA
# ============================================
print("\n" + "="*60)
print("📦 CHARGEMENT DES DONNÉES")
print("="*60)

start_load = time.time()
print("\n⏳ Chargement du train dataset...")

train = pd.read_csv(PROCESSED_PATH / "train_processed.csv", parse_dates=['date'])
test_full = pd.read_csv(PROCESSED_PATH / "test_processed.csv", parse_dates=['date'])

elapsed_load = time.time() - start_load
print(f"✅ Données chargées en {elapsed_load:.2f}s")
print(f"   Train: {train.shape}")
print(f"   Test: {test_full.shape}")

# ============================================
# 2. PREPARE FEATURES
# ============================================
print("\n" + "="*60)
print("🔧 PRÉPARATION DES FEATURES")
print("="*60)

feature_cols = [
    'dayofweek', 'day', 'month', 'quarter', 'dayofyear', 'weekofyear',
    'is_weekend', 'is_month_start', 'is_month_end', 'is_payday',
    'sales_lag_1', 'sales_lag_7', 'sales_lag_14', 'sales_lag_30',
    'sales_rolling_mean_7', 'sales_rolling_std_7',
    'sales_rolling_mean_14', 'sales_rolling_std_14',
    'sales_rolling_mean_30', 'sales_rolling_std_30',
    'is_holiday', 'has_promotion', 'onpromotion',
    'cluster', 'transactions', 'type_encoded'
]

feature_cols = [col for col in feature_cols if col in train.columns]
print(f"\n📊 {len(feature_cols)} features sélectionnées")

X_train = train[feature_cols].fillna(0)
y_train = train['sales']
X_test = test_full[feature_cols].fillna(0)

print(f"✅ X_train: {X_train.shape}")
print(f"✅ X_test: {X_test.shape}")

# ============================================
# 3. TRAIN XGBOOST
# ============================================
print("\n" + "="*60)
print("1️⃣ XGBOOST - FULL DATASET")
print("="*60)

try:
    import xgboost as xgb
    
    print("\n⏳ Entraînement XGBoost sur 3M+ lignes...")
    print("   (Ceci peut prendre plusieurs minutes sur le full dataset)")
    
    start_xgb = time.time()
    
    model_xgb = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        tree_method='hist'  # Plus rapide pour gros datasets
    )
    
    print(f"   ⏰ Début training: {datetime.now().strftime('%H:%M:%S')}")
    model_xgb.fit(X_train, y_train, verbose=False)
    
    elapsed_xgb = time.time() - start_xgb
    print(f"   ⏰ Fin training: {datetime.now().strftime('%H:%M:%S')}")
    print(f"   ⏱️ Temps total: {elapsed_xgb/60:.2f} minutes ({elapsed_xgb:.1f}s)")
    
    print("\n⏳ Prédictions...")
    start_pred = time.time()
    predictions_xgb = model_xgb.predict(X_test)
    predictions_xgb = np.maximum(predictions_xgb, 0)
    elapsed_pred = time.time() - start_pred
    print(f"   ⏱️ Temps prédictions: {elapsed_pred:.2f}s")
    
    # Save
    joblib.dump(model_xgb, MODELS_PATH / 'xgboost_full_model.pkl')
    print("   💾 Modèle sauvegardé: xgboost_full_model.pkl")
    
    xgb_success = True
    
except Exception as e:
    print(f"❌ Erreur XGBoost: {e}")
    xgb_success = False

# ============================================
# 4. TRAIN LIGHTGBM
# ============================================
print("\n" + "="*60)
print("2️⃣ LIGHTGBM - FULL DATASET")
print("="*60)

try:
    import lightgbm as lgb
    
    print("\n⏳ Entraînement LightGBM sur 3M+ lignes...")
    
    start_lgb = time.time()
    
    model_lgb = lgb.LGBMRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    
    print(f"   ⏰ Début training: {datetime.now().strftime('%H:%M:%S')}")
    model_lgb.fit(X_train, y_train)
    
    elapsed_lgb = time.time() - start_lgb
    print(f"   ⏰ Fin training: {datetime.now().strftime('%H:%M:%S')}")
    print(f"   ⏱️ Temps total: {elapsed_lgb/60:.2f} minutes ({elapsed_lgb:.1f}s)")
    
    print("\n⏳ Prédictions...")
    start_pred = time.time()
    predictions_lgb = model_lgb.predict(X_test)
    predictions_lgb = np.maximum(predictions_lgb, 0)
    elapsed_pred = time.time() - start_pred
    print(f"   ⏱️ Temps prédictions: {elapsed_pred:.2f}s")
    
    # Save
    joblib.dump(model_lgb, MODELS_PATH / 'lightgbm_full_model.pkl')
    print("   💾 Modèle sauvegardé: lightgbm_full_model.pkl")
    
    lgb_success = True
    
except Exception as e:
    print(f"❌ Erreur LightGBM: {e}")
    lgb_success = False

# ============================================
# 5. ENSEMBLE & SAVE PREDICTIONS
# ============================================
print("\n" + "="*60)
print("🎯 ENSEMBLE & SAUVEGARDE")
print("="*60)

if xgb_success and lgb_success:
    print("\n⏳ Création des prédictions ensemble...")
    predictions_ensemble = (predictions_xgb + predictions_lgb) / 2
    predictions_ensemble = np.maximum(predictions_ensemble, 0)
    
    # Save predictions
    submission = test_full[['id']].copy()
    submission['sales'] = predictions_ensemble
    
    submission.to_csv(PREDICTIONS_PATH / 'submission_ensemble.csv', index=False)
    print("   ✅ submission_ensemble.csv sauvegardée")
    
    # Save individual predictions
    submission['sales'] = predictions_xgb
    submission.to_csv(PREDICTIONS_PATH / 'submission_xgboost.csv', index=False)
    print("   ✅ submission_xgboost.csv sauvegardée")
    
    submission['sales'] = predictions_lgb
    submission.to_csv(PREDICTIONS_PATH / 'submission_lightgbm.csv', index=False)
    print("   ✅ submission_lightgbm.csv sauvegardée")

# ============================================
# 6. SUMMARY
# ============================================
print("\n" + "="*60)
print("📊 RÉSUMÉ DU TRAINING")
print("="*60)

total_time = time.time() - start_load

print(f"""
📦 Dataset:
   - Train: {train.shape[0]:,} lignes
   - Test: {test_full.shape[0]:,} lignes
   - Features: {len(feature_cols)}

⏱️ Temps d'exécution:
   - Chargement: {elapsed_load:.2f}s
""")

if xgb_success:
    print(f"   - XGBoost: {elapsed_xgb/60:.2f} min ({elapsed_xgb:.1f}s)")
    
if lgb_success:
    print(f"   - LightGBM: {elapsed_lgb/60:.2f} min ({elapsed_lgb:.1f}s)")

print(f"   - TOTAL: {total_time/60:.2f} minutes")

print(f"""
💾 Fichiers sauvegardés:
   - models/xgboost_full_model.pkl
   - models/lightgbm_full_model.pkl
   - data/predictions/submission_xgboost.csv
   - data/predictions/submission_lightgbm.csv
   - data/predictions/submission_ensemble.csv
""")

print("\n" + "="*60)
print("✅ FULL DATASET TRAINING TERMINÉ !")
print(f"⏰ Fin: {datetime.now().strftime('%H:%M:%S')}")
print("="*60)
