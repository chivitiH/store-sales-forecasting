"""
Store Sales Forecasting - Baseline Models
Test de modèles simples pour établir une baseline
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# Paths
PROCESSED_PATH = Path("data/processed")
MODELS_PATH = Path("models")
REPORTS_PATH = Path("reports/figures")
MODELS_PATH.mkdir(parents=True, exist_ok=True)

print("="*60)
print("🎯 BASELINE MODELS - TIME SERIES FORECASTING")
print("="*60)

# ============================================
# 1. LOAD DATA
# ============================================
print("\n📦 Chargement des données...")

# Charger un sample pour tester rapidement
df = pd.read_csv(PROCESSED_PATH / "train_processed.csv", parse_dates=['date'])

print(f"✅ Data shape: {df.shape}")

# ============================================
# 2. SELECT SUBSET FOR TESTING
# ============================================
print("\n" + "="*60)
print("🎯 SÉLECTION D'UN SUBSET POUR TEST")
print("="*60)

# Prendre une famille populaire et un magasin
top_family = df.groupby('family')['sales'].sum().idxmax()
top_store = df.groupby('store_nbr')['sales'].sum().idxmax()

print(f"\n�� Famille sélectionnée: {top_family}")
print(f"🏪 Magasin sélectionné: {top_store}")

# Filtrer les données
subset = df[(df['family'] == top_family) & (df['store_nbr'] == top_store)].copy()
subset = subset.sort_values('date').reset_index(drop=True)

print(f"📊 Subset shape: {subset.shape}")
print(f"📅 Période: {subset['date'].min()} → {subset['date'].max()}")

# ============================================
# 3. TRAIN/TEST SPLIT
# ============================================
print("\n" + "="*60)
print("✂️ TRAIN/TEST SPLIT")
print("="*60)

# Derniers 30 jours = test
test_days = 30
train = subset[:-test_days].copy()
test = subset[-test_days:].copy()

print(f"\n✅ Train: {len(train)} jours ({train['date'].min()} → {train['date'].max()})")
print(f"✅ Test: {len(test)} jours ({test['date'].min()} → {test['date'].max()})")

# ============================================
# 4. EVALUATION METRICS
# ============================================

def evaluate_model(y_true, y_pred, model_name):
    """Calcule les métriques d'évaluation"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    print(f"\n📊 {model_name}:")
    print(f"   RMSE: {rmse:.2f}")
    print(f"   MAE:  {mae:.2f}")
    print(f"   MAPE: {mape:.2f}%")
    
    return {'model': model_name, 'rmse': rmse, 'mae': mae, 'mape': mape}

# ============================================
# 5. BASELINE 1: MEAN
# ============================================
print("\n" + "="*60)
print("1️⃣ BASELINE: MEAN")
print("="*60)

mean_prediction = train['sales'].mean()
pred_mean = np.full(len(test), mean_prediction)

results = []
results.append(evaluate_model(test['sales'].values, pred_mean, 'Mean Baseline'))

# ============================================
# 6. BASELINE 2: LAST VALUE
# ============================================
print("\n" + "="*60)
print("2️⃣ BASELINE: LAST VALUE")
print("="*60)

last_value = train['sales'].iloc[-1]
pred_last = np.full(len(test), last_value)

results.append(evaluate_model(test['sales'].values, pred_last, 'Last Value'))

# ============================================
# 7. BASELINE 3: SEASONAL NAIVE (7 days)
# ============================================
print("\n" + "="*60)
print("3️⃣ BASELINE: SEASONAL NAIVE (7 jours)")
print("="*60)

# Prendre les valeurs de la même période la semaine précédente
pred_seasonal = []
for i in range(len(test)):
    # Index correspondant il y a 7 jours
    if len(train) >= 7:
        pred_seasonal.append(train['sales'].iloc[-(7-i) if i < 7 else -7])
    else:
        pred_seasonal.append(train['sales'].mean())

pred_seasonal = np.array(pred_seasonal)
results.append(evaluate_model(test['sales'].values, pred_seasonal, 'Seasonal Naive (7d)'))

# ============================================
# 8. BASELINE 4: LINEAR REGRESSION
# ============================================
print("\n" + "="*60)
print("4️⃣ BASELINE: LINEAR REGRESSION")
print("="*60)

# Features pour la régression
feature_cols = ['dayofweek', 'day', 'month', 'is_weekend', 'is_holiday', 
                'has_promotion', 'sales_lag_7', 'sales_lag_14', 'sales_rolling_mean_7']

# Filtrer les features disponibles
feature_cols = [col for col in feature_cols if col in train.columns]

print(f"\n📊 Features utilisées: {feature_cols}")

# Préparer les données
X_train = train[feature_cols].fillna(0)
y_train = train['sales']
X_test = test[feature_cols].fillna(0)

# Entraîner
lr = LinearRegression()
lr.fit(X_train, y_train)

pred_lr = lr.predict(X_test)
pred_lr = np.maximum(pred_lr, 0)  # Pas de ventes négatives

results.append(evaluate_model(test['sales'].values, pred_lr, 'Linear Regression'))

# ============================================
# 9. BASELINE 5: ARIMA
# ============================================
print("\n" + "="*60)
print("5️⃣ BASELINE: ARIMA")
print("="*60)

try:
    from statsmodels.tsa.arima.model import ARIMA
    
    print("\n⏳ Entraînement ARIMA (peut prendre quelques secondes)...")
    
    # ARIMA simple (1,1,1)
    model_arima = ARIMA(train['sales'], order=(1, 1, 1))
    model_arima_fit = model_arima.fit()
    
    # Prédictions
    pred_arima = model_arima_fit.forecast(steps=len(test))
    pred_arima = np.maximum(pred_arima, 0)
    
    results.append(evaluate_model(test['sales'].values, pred_arima, 'ARIMA(1,1,1)'))
    
except Exception as e:
    print(f"❌ Erreur ARIMA: {e}")
    pred_arima = pred_mean  # Fallback

# ============================================
# 10. BASELINE 6: PROPHET
# ============================================
print("\n" + "="*60)
print("6️⃣ BASELINE: PROPHET (Facebook)")
print("="*60)

try:
    from prophet import Prophet
    
    print("\n⏳ Entraînement Prophet...")
    
    # Préparer les données pour Prophet (format spécifique)
    prophet_train = train[['date', 'sales']].copy()
    prophet_train.columns = ['ds', 'y']
    
    # Entraîner
    model_prophet = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True,
        seasonality_mode='multiplicative'
    )
    model_prophet.fit(prophet_train)
    
    # Prédictions
    future = model_prophet.make_future_dataframe(periods=len(test))
    forecast = model_prophet.predict(future)
    pred_prophet = forecast['yhat'].tail(len(test)).values
    pred_prophet = np.maximum(pred_prophet, 0)
    
    results.append(evaluate_model(test['sales'].values, pred_prophet, 'Prophet'))
    
except Exception as e:
    print(f"❌ Erreur Prophet: {e}")
    pred_prophet = pred_mean  # Fallback

# ============================================
# 11. RESULTS COMPARISON
# ============================================
print("\n" + "="*60)
print("📊 COMPARAISON DES MODÈLES")
print("="*60)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('rmse')

print("\n🏆 Classement par RMSE:")
print(results_df.to_string(index=False))

# Sauvegarder les résultats
results_df.to_csv(MODELS_PATH / 'baseline_results.csv', index=False)
print(f"\n💾 Résultats sauvegardés: {MODELS_PATH / 'baseline_results.csv'}")

# ============================================
# 12. VISUALIZATION
# ============================================
print("\n" + "="*60)
print("📈 GÉNÉRATION DES VISUALISATIONS")
print("="*60)

# Plot 1: Actual vs Predictions
print("\n📊 1. Comparaison des prédictions...")

fig, ax = plt.subplots(figsize=(15, 8))

# Actual values
ax.plot(test['date'], test['sales'], 'o-', label='Actual', linewidth=2, markersize=6, color='black')

# Predictions
ax.plot(test['date'], pred_mean, '--', label='Mean', alpha=0.7)
ax.plot(test['date'], pred_last, '--', label='Last Value', alpha=0.7)
ax.plot(test['date'], pred_seasonal, '--', label='Seasonal Naive', alpha=0.7)
ax.plot(test['date'], pred_lr, '--', label='Linear Regression', alpha=0.7)

try:
    ax.plot(test['date'], pred_arima, '--', label='ARIMA', alpha=0.7)
except:
    pass

try:
    ax.plot(test['date'], pred_prophet, '--', label='Prophet', alpha=0.7)
except:
    pass

ax.set_title(f'Baseline Models - Prédictions vs Actual\n{top_family} @ Store {top_store}', 
             fontsize=14, fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Sales')
ax.legend(loc='best')
ax.grid(alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(REPORTS_PATH / '09_baseline_predictions.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Sauvegardé: 09_baseline_predictions.png")

# Plot 2: Model Comparison
print("\n📊 2. Comparaison des métriques...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# RMSE
axes[0].barh(results_df['model'], results_df['rmse'], color='steelblue')
axes[0].set_xlabel('RMSE')
axes[0].set_title('Root Mean Squared Error', fontweight='bold')
axes[0].grid(alpha=0.3, axis='x')

# MAE
axes[1].barh(results_df['model'], results_df['mae'], color='coral')
axes[1].set_xlabel('MAE')
axes[1].set_title('Mean Absolute Error', fontweight='bold')
axes[1].grid(alpha=0.3, axis='x')

# MAPE
axes[2].barh(results_df['model'], results_df['mape'], color='green')
axes[2].set_xlabel('MAPE (%)')
axes[2].set_title('Mean Absolute Percentage Error', fontweight='bold')
axes[2].grid(alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(REPORTS_PATH / '10_baseline_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Sauvegardé: 10_baseline_comparison.png")

# ============================================
# 13. BEST MODEL
# ============================================
print("\n" + "="*60)
print("�� MEILLEUR MODÈLE BASELINE")
print("="*60)

best_model = results_df.iloc[0]
print(f"\n🥇 Modèle: {best_model['model']}")
print(f"   RMSE: {best_model['rmse']:.2f}")
print(f"   MAE:  {best_model['mae']:.2f}")
print(f"   MAPE: {best_model['mape']:.2f}%")

print("\n" + "="*60)
print("✅ BASELINE MODELS TERMINÉ !")
print("="*60)
print(f"\n📁 Résultats: {MODELS_PATH}")
print(f"📊 Figures: {REPORTS_PATH}")
