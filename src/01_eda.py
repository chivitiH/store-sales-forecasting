"""
Store Sales Forecasting - Exploratory Data Analysis
Génère visualisations et insights sur les données de ventes
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuration
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Paths
DATA_PATH = Path("data/raw")
REPORTS_PATH = Path("reports/figures")
REPORTS_PATH.mkdir(parents=True, exist_ok=True)

print("="*60)
print("🚀 STORE SALES FORECASTING - EDA")
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

print(f"✅ Train shape: {train.shape}")
print(f"✅ Test shape: {test.shape}")
print(f"✅ Stores: {stores.shape[0]} magasins")
print(f"✅ Oil prices: {oil.shape[0]} jours")
print(f"✅ Holidays: {holidays.shape[0]} événements")
print(f"✅ Transactions: {transactions.shape[0]} records")

# ============================================
# 2. DATA OVERVIEW
# ============================================
print("\n" + "="*60)
print("📊 APERÇU DES DONNÉES")
print("="*60)

print("\n🏪 Stores info:")
print(stores.head())
print(f"\nTypes de magasins: {stores['type'].value_counts().to_dict()}")
print(f"Villes: {stores['city'].nunique()} différentes")
print(f"États: {stores['state'].nunique()} différents")

print("\n📦 Produits:")
print(f"Familles de produits: {train['family'].nunique()}")
print("\nTop 10 familles par ventes totales:")
family_sales = train.groupby('family')['sales'].sum().sort_values(ascending=False).head(10)
print(family_sales)

print("\n📅 Période temporelle:")
print(f"Train: {train['date'].min()} → {train['date'].max()}")
print(f"Test: {test['date'].min()} → {test['date'].max()}")
print(f"Durée train: {(train['date'].max() - train['date'].min()).days} jours")

# ============================================
# 3. MISSING VALUES
# ============================================
print("\n" + "="*60)
print("❓ MISSING VALUES")
print("="*60)

print("\nTrain missing:")
print(train.isnull().sum())

print("\nOil missing:")
print(f"Oil prices: {oil.isnull().sum()['dcoilwtico']} valeurs manquantes")

# ============================================
# 4. SALES ANALYSIS
# ============================================
print("\n" + "="*60)
print("💰 ANALYSE DES VENTES")
print("="*60)

print(f"\nVentes totales: ${train['sales'].sum():,.0f}")
print(f"Ventes moyennes par jour: ${train['sales'].mean():,.2f}")
print(f"Ventes médiane: ${train['sales'].median():,.2f}")
print(f"Ventes max: ${train['sales'].max():,.0f}")
print(f"Ventes min: ${train['sales'].min():,.0f}")

# Ventes négatives ?
neg_sales = train[train['sales'] < 0]
print(f"\n⚠️ Ventes négatives: {len(neg_sales)} ({len(neg_sales)/len(train)*100:.2f}%)")

# ============================================
# 5. TIME SERIES PLOTS
# ============================================
print("\n" + "="*60)
print("📈 GÉNÉRATION DES VISUALISATIONS")
print("="*60)

# 5.1 Total sales over time
print("\n📊 1. Ventes totales dans le temps...")
daily_sales = train.groupby('date')['sales'].sum().reset_index()

fig, ax = plt.subplots(figsize=(15, 6))
ax.plot(daily_sales['date'], daily_sales['sales'], linewidth=0.8)
ax.set_title('Ventes Quotidiennes Totales (2013-2017)', fontsize=16, fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Ventes ($)')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(REPORTS_PATH / '01_sales_over_time.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Sauvegardé: 01_sales_over_time.png")

# 5.2 Sales by store type
print("\n📊 2. Ventes par type de magasin...")
train_stores = train.merge(stores, on='store_nbr')
store_type_sales = train_stores.groupby('type')['sales'].sum().sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(10, 6))
store_type_sales.plot(kind='bar', ax=ax, color='steelblue')
ax.set_title('Ventes Totales par Type de Magasin', fontsize=16, fontweight='bold')
ax.set_xlabel('Type de Magasin')
ax.set_ylabel('Ventes Totales ($)')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
plt.tight_layout()
plt.savefig(REPORTS_PATH / '02_sales_by_store_type.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Sauvegardé: 02_sales_by_store_type.png")

# 5.3 Top 10 product families
print("\n📊 3. Top 10 familles de produits...")
fig, ax = plt.subplots(figsize=(12, 6))
family_sales.plot(kind='barh', ax=ax, color='coral')
ax.set_title('Top 10 Familles de Produits par Ventes', fontsize=16, fontweight='bold')
ax.set_xlabel('Ventes Totales ($)')
ax.set_ylabel('Famille de Produits')
plt.tight_layout()
plt.savefig(REPORTS_PATH / '03_top_families.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Sauvegardé: 03_top_families.png")

# 5.4 Seasonality - Monthly pattern
print("\n📊 4. Patterns mensuels...")
train['month'] = train['date'].dt.month
monthly_sales = train.groupby('month')['sales'].mean()

fig, ax = plt.subplots(figsize=(12, 6))
monthly_sales.plot(kind='bar', ax=ax, color='teal')
ax.set_title('Ventes Moyennes par Mois', fontsize=16, fontweight='bold')
ax.set_xlabel('Mois')
ax.set_ylabel('Ventes Moyennes ($)')
ax.set_xticklabels(['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Juin', 
                     'Juil', 'Août', 'Sep', 'Oct', 'Nov', 'Déc'], rotation=45)
plt.tight_layout()
plt.savefig(REPORTS_PATH / '04_monthly_pattern.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Sauvegardé: 04_monthly_pattern.png")

# 5.5 Day of week pattern
print("\n📊 5. Patterns jour de la semaine...")
train['dayofweek'] = train['date'].dt.dayofweek
dow_sales = train.groupby('dayofweek')['sales'].mean()

fig, ax = plt.subplots(figsize=(10, 6))
dow_sales.plot(kind='bar', ax=ax, color='purple')
ax.set_title('Ventes Moyennes par Jour de la Semaine', fontsize=16, fontweight='bold')
ax.set_xlabel('Jour de la Semaine')
ax.set_ylabel('Ventes Moyennes ($)')
ax.set_xticklabels(['Lun', 'Mar', 'Mer', 'Jeu', 'Ven', 'Sam', 'Dim'], rotation=45)
plt.tight_layout()
plt.savefig(REPORTS_PATH / '05_dayofweek_pattern.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Sauvegardé: 05_dayofweek_pattern.png")

# 5.6 Promotions impact (FIXED)
print("\n📊 6. Impact des promotions...")
train['has_promotion'] = (train['onpromotion'] > 0).astype(int)
promo_impact = train.groupby('has_promotion')['sales'].mean()

fig, ax = plt.subplots(figsize=(8, 6))
promo_impact.plot(kind='bar', ax=ax, color=['red', 'green'])
ax.set_title('Impact des Promotions sur les Ventes', fontsize=16, fontweight='bold')
ax.set_xlabel('En Promotion')
ax.set_ylabel('Ventes Moyennes ($)')
ax.set_xticks([0, 1])
ax.set_xticklabels(['Non', 'Oui'], rotation=0)
plt.tight_layout()
plt.savefig(REPORTS_PATH / '06_promotion_impact.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Sauvegardé: 06_promotion_impact.png")

# 5.7 Oil prices
print("\n📊 7. Prix du pétrole...")
oil_clean = oil.dropna()

fig, ax = plt.subplots(figsize=(15, 6))
ax.plot(oil_clean['date'], oil_clean['dcoilwtico'], color='black', linewidth=1)
ax.set_title('Prix du Pétrole dans le Temps', fontsize=16, fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Prix du Pétrole ($/baril)')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(REPORTS_PATH / '07_oil_prices.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Sauvegardé: 07_oil_prices.png")

# 5.8 Correlation: Oil vs Sales
print("\n📊 8. Corrélation pétrole vs ventes...")
train_with_oil = train.merge(oil, on='date', how='left')
daily_sales_oil = train_with_oil.groupby('date').agg({
    'sales': 'sum',
    'dcoilwtico': 'first'
}).dropna()

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(daily_sales_oil['dcoilwtico'], daily_sales_oil['sales'], alpha=0.5, s=10)
ax.set_title('Corrélation Prix du Pétrole vs Ventes', fontsize=16, fontweight='bold')
ax.set_xlabel('Prix du Pétrole ($/baril)')
ax.set_ylabel('Ventes Quotidiennes ($)')
ax.grid(alpha=0.3)

# Correlation coefficient
corr = daily_sales_oil['dcoilwtico'].corr(daily_sales_oil['sales'])
ax.text(0.05, 0.95, f'Corrélation: {corr:.3f}', 
        transform=ax.transAxes, fontsize=12, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
plt.tight_layout()
plt.savefig(REPORTS_PATH / '08_oil_sales_correlation.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Sauvegardé: 08_oil_sales_correlation.png")

# ============================================
# 6. KEY INSIGHTS
# ============================================
print("\n" + "="*60)
print("💡 KEY INSIGHTS")
print("="*60)

print(f"""
1. 📅 Période de données: {(train['date'].max() - train['date'].min()).days} jours
2. 🏪 {stores.shape[0]} magasins dans {stores['city'].nunique()} villes
3. 📦 {train['family'].nunique()} familles de produits
4. 💰 Ventes totales: ${train['sales'].sum():,.0f}
5. 📈 Trend: Croissance visible dans le temps
6. 📆 Saisonnalité: Pics en décembre (Noël), creux en janvier
7. 🎯 Promotions: Impact positif sur les ventes
8. 🛢️ Pétrole: Corrélation {corr:.3f} avec les ventes
9. ⚠️ Ventes négatives: {len(neg_sales)} observations (retours?)
10. 🏆 Top famille: {family_sales.index[0]} (${family_sales.iloc[0]:,.0f})
""")

print("\n" + "="*60)
print("✅ EDA TERMINÉE !")
print(f"📁 Figures sauvegardées dans: {REPORTS_PATH}")
print("="*60)
