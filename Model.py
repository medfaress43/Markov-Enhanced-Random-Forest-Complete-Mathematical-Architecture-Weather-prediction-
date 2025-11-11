# ===============================================================
# 1️⃣ Imports
# ===============================================================
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ===============================================================
# 2️⃣ Markov Chain Feature Extractor
# ===============================================================
class MarkovChainFeatures:
    """
    Markov Chain for capturing temporal transition patterns.
    Discretizes continuous values into states and learns transition probabilities.
    """
    def __init__(self, n_states=5):
        self.n_states = n_states
        self.state_bins = None
        self.transition_matrix = None
        self.state_probs = None

    def fit(self, series):
        """Learn state boundaries and transition probabilities"""
        series = np.array(series)
        
        # Create state bins using quantiles for better distribution
        self.state_bins = np.percentile(series, np.linspace(0, 100, self.n_states + 1))
        self.state_bins[0] = series.min() - 0.001  # Ensure all values fit
        self.state_bins[-1] = series.max() + 0.001
        
        # Digitize into states
        states = np.digitize(series, self.state_bins) - 1
        states = np.clip(states, 0, self.n_states - 1)
        
        # Build transition matrix
        self.transition_matrix = np.zeros((self.n_states, self.n_states))
        for s1, s2 in zip(states[:-1], states[1:]):
            self.transition_matrix[s1, s2] += 1
        
        # Add smoothing and normalize
        self.transition_matrix += 0.1
        self.transition_matrix /= self.transition_matrix.sum(axis=1, keepdims=True)
        
        # Calculate stationary state probabilities
        self.state_probs = np.bincount(states, minlength=self.n_states) / len(states)
        
        return self

    def transform(self, series):
        """Extract Markov features for each value"""
        series = np.array(series)
        states = np.digitize(series, self.state_bins) - 1
        states = np.clip(states, 0, self.n_states - 1)
        
        features = []
        for s in states:
            # Transition probabilities from current state
            trans_probs = self.transition_matrix[s]
            
            # Additional features: entropy and expected next state
            entropy = -np.sum(trans_probs * np.log(trans_probs + 1e-10))
            expected_next = np.dot(trans_probs, np.arange(self.n_states))
            
            # Combine: transition probs + entropy + expected next state
            feat = np.concatenate([trans_probs, [entropy, expected_next]])
            features.append(feat)
        
        return np.array(features)

# ===============================================================
# 3️⃣ Hybrid Markov + Random Forest Model
# ===============================================================
class MarkovEnhancedRandomForest:
    """
    Random Forest enriched with Markov Chain features for better temporal patterns.
    """
    def __init__(self, n_states=5, n_trees=150, max_depth=6):
        self.markov = MarkovChainFeatures(n_states=n_states)
        self.rf = RandomForestRegressor(
            n_estimators=n_trees,
            max_depth=max_depth,
            min_samples_split=15,
            min_samples_leaf=5,
            max_features='sqrt',
            max_samples=0.7,
            random_state=42,
            n_jobs=-1
        )
        self.feature_names = None

    def create_features(self, df):
        """Create lag and rolling features"""
        df_feat = df.copy()
        
        # Simple lag features
        df_feat['lag1'] = df_feat['temperature'].shift(1)
        df_feat['lag2'] = df_feat['temperature'].shift(2)
        
        # Rolling statistics
        df_feat['rolling_mean5'] = df_feat['temperature'].shift(1).rolling(5, min_periods=1).mean()
        df_feat['rolling_std5'] = df_feat['temperature'].shift(1).rolling(5, min_periods=1).std()
        
        # Trend
        df_feat['trend'] = np.arange(len(df_feat)) / len(df_feat)
        
        # Fill any NaN
        df_feat = df_feat.bfill().ffill()
        
        return df_feat

    def fit(self, X, y):
        """Fit Markov Chain and Random Forest"""
        # Learn Markov transitions from target variable
        self.markov.fit(y)
        
        # Extract Markov features
        markov_feats = self.markov.transform(y)
        
        # Combine with regular features
        X_combined = np.hstack([X, markov_feats])
        
        # Store feature names for importance
        n_base = X.shape[1]
        n_markov = markov_feats.shape[1]
        self.feature_names = (
            [f'base_{i}' for i in range(n_base)] +
            [f'markov_trans_{i}' for i in range(self.markov.n_states)] +
            ['markov_entropy', 'markov_expected_next']
        )
        
        # Fit Random Forest
        self.rf.fit(X_combined, y)
        
        return self

    def predict(self, X, last_values):
        """Predict using both base features and Markov enrichment"""
        # Extract Markov features from recent values
        markov_feats = self.markov.transform(last_values)
        
        # Combine features
        X_combined = np.hstack([X, markov_feats])
        
        return self.rf.predict(X_combined)

    def forecast_one_day(self, df):
        """1-day ahead forecast"""
        X_feat = self.create_features(df).drop(columns=['temperature']).values
        last_temp = df['temperature'].values
        
        pred = self.predict(X_feat[-1:], last_temp[-1:])[0]
        return pred

# ===============================================================
# 4️⃣ Generate Synthetic Weather Data
# ===============================================================
np.random.seed(42)
n_samples = 200

# Realistic temperature pattern: seasonal + noise
t = np.linspace(0, 20, n_samples)
temperature = (
    np.sin(t) * 10 +                    # Seasonal pattern
    0.5 * np.sin(t * 3) * 5 +          # Higher frequency variation
    20 +                                 # Base temperature
    np.random.randn(n_samples) * 2      # Random noise
)

df = pd.DataFrame({'temperature': temperature})

# Train/Val/Test split
train_size = int(0.7 * n_samples)
val_size = int(0.15 * n_samples)

train_df = df.iloc[:train_size]
val_df = df.iloc[train_size:train_size+val_size]
test_df = df.iloc[train_size+val_size:]

print("="*70)
print("🌤️  MARKOV-ENHANCED RANDOM FOREST WEATHER PREDICTION")
print("="*70)
print(f"\nDataset Split:")
print(f"  • Training:   {len(train_df)} samples")
print(f"  • Validation: {len(val_df)} samples")
print(f"  • Test:       {len(test_df)} samples")

# ===============================================================
# 5️⃣ Train Model
# ===============================================================
print("\n🔄 Training Markov-Enhanced Random Forest...\n")

model = MarkovEnhancedRandomForest(n_states=5, n_trees=150, max_depth=6)

# Prepare features
X_train = model.create_features(train_df).drop(columns=['temperature']).values
y_train = train_df['temperature'].values

# Fit model
model.fit(X_train, y_train)

print("✅ Model trained successfully!")
print(f"   • Markov States: {model.markov.n_states}")
print(f"   • Base Features: {X_train.shape[1]}")
print(f"   • Markov Features: {model.markov.n_states + 2}")
print(f"   • Total Features: {X_train.shape[1] + model.markov.n_states + 2}")

# ===============================================================
# 6️⃣ Make Predictions
# ===============================================================
print("\n🔮 Generating predictions...\n")

# Validation predictions (1-day ahead rolling)
val_preds = []
for i in range(len(val_df)):
    # Use all data up to current point
    current_df = pd.concat([train_df, val_df.iloc[:i+1]])
    pred = model.forecast_one_day(current_df)
    val_preds.append(pred)
val_preds = np.array(val_preds)

# Test predictions (1-day ahead rolling)
test_preds = []
for i in range(len(test_df)):
    # Use all data up to current point
    current_df = pd.concat([train_df, val_df, test_df.iloc[:i+1]])
    pred = model.forecast_one_day(current_df)
    test_preds.append(pred)
test_preds = np.array(test_preds)

# ===============================================================
# 7️⃣ Evaluate Performance
# ===============================================================
def evaluate(y_true, y_pred, name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    print(f"{name} Metrics:")
    print(f"  • R² Score:  {r2:.4f} ({r2*100:.2f}%)")
    print(f"  • RMSE:      {rmse:.4f}°C")
    print(f"  • MAE:       {mae:.4f}°C")
    print(f"  • MAPE:      {mape:.2f}%")
    
    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R²': r2, 'MAPE': mape}

y_val = val_df['temperature'].values
y_test = test_df['temperature'].values

val_metrics = evaluate(y_val, val_preds, "Validation")
print()
test_metrics = evaluate(y_test, test_preds, "Test")

# ===============================================================
# 8️⃣ Comprehensive Visualizations
# ===============================================================
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)

# 1. Main Time Series
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(range(len(df)), df['temperature'], 'k-', linewidth=2.5, alpha=0.7, label='Actual')
ax1.plot(range(train_size, train_size+val_size), val_preds, 'o-', 
         linewidth=2, markersize=5, color='#2E86AB', label='Val Predictions')
ax1.plot(range(train_size+val_size, len(df)), test_preds, 's-', 
         linewidth=2, markersize=5, color='#A23B72', label='Test Predictions')
ax1.axvline(train_size, color='gray', linestyle='--', alpha=0.6, linewidth=2, label='Train|Val')
ax1.axvline(train_size+val_size, color='red', linestyle='--', alpha=0.6, linewidth=2, label='Val|Test')
ax1.fill_between(range(train_size), df['temperature'][:train_size].min()-3, 
                  df['temperature'][:train_size].max()+3, alpha=0.1, color='green')
ax1.set_xlabel('Time Steps', fontsize=12, fontweight='bold')
ax1.set_ylabel('Temperature (°C)', fontsize=12, fontweight='bold')
ax1.set_title('Markov-Enhanced Random Forest: 1-Day Ahead Forecast', fontsize=14, fontweight='bold')
ax1.legend(loc='best', fontsize=10)
ax1.grid(True, alpha=0.3)

# 2. Validation Scatter
ax2 = fig.add_subplot(gs[1, 0])
errors_val = np.abs(y_val - val_preds)
scatter = ax2.scatter(y_val, val_preds, c=errors_val, cmap='YlOrRd', 
                     s=80, alpha=0.7, edgecolors='k', linewidth=0.5)
ax2.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2.5)
ax2.set_xlabel('Actual (°C)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Predicted (°C)', fontsize=11, fontweight='bold')
ax2.set_title('Validation', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax2, label='Error (°C)')
text = f"R² = {val_metrics['R²']:.4f}\nRMSE = {val_metrics['RMSE']:.3f}°C\nMAE = {val_metrics['MAE']:.3f}°C"
ax2.text(0.05, 0.95, text, transform=ax2.transAxes, fontsize=10, 
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

# 3. Test Scatter
ax3 = fig.add_subplot(gs[1, 1])
errors_test = np.abs(y_test - test_preds)
scatter = ax3.scatter(y_test, test_preds, c=errors_test, cmap='YlOrRd', 
                     s=80, alpha=0.7, edgecolors='k', linewidth=0.5)
ax3.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2.5)
ax3.set_xlabel('Actual (°C)', fontsize=11, fontweight='bold')
ax3.set_ylabel('Predicted (°C)', fontsize=11, fontweight='bold')
ax3.set_title('Test', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax3, label='Error (°C)')
text = f"R² = {test_metrics['R²']:.4f}\nRMSE = {test_metrics['RMSE']:.3f}°C\nMAE = {test_metrics['MAE']:.3f}°C"
ax3.text(0.05, 0.95, text, transform=ax3.transAxes, fontsize=10, 
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='#FFB3C6', alpha=0.8))

# 4. Markov Transition Matrix Heatmap
ax4 = fig.add_subplot(gs[1, 2])
im = ax4.imshow(model.markov.transition_matrix, cmap='YlOrRd', aspect='auto')
ax4.set_xlabel('Next State', fontsize=11, fontweight='bold')
ax4.set_ylabel('Current State', fontsize=11, fontweight='bold')
ax4.set_title('Markov Transition Matrix', fontsize=12, fontweight='bold')
for i in range(model.markov.n_states):
    for j in range(model.markov.n_states):
        text = ax4.text(j, i, f'{model.markov.transition_matrix[i, j]:.2f}',
                       ha="center", va="center", color="black", fontsize=9)
plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)

# 5. Residual Analysis
ax5 = fig.add_subplot(gs[2, :2])
val_residuals = y_val - val_preds
test_residuals = y_test - test_preds
ax5.scatter(range(len(val_residuals)), val_residuals, s=60, alpha=0.7, 
           color='#2E86AB', edgecolors='k', linewidth=0.5, label='Validation')
ax5.scatter(range(len(val_residuals), len(val_residuals)+len(test_residuals)), 
           test_residuals, s=60, alpha=0.7, color='#A23B72', 
           edgecolors='k', linewidth=0.5, label='Test')
ax5.axhline(0, color='r', linestyle='--', linewidth=2.5)
std_all = np.std(np.concatenate([val_residuals, test_residuals]))
ax5.axhspan(-2*std_all, 2*std_all, alpha=0.15, color='gray', label='±2σ')
ax5.set_xlabel('Sample Index', fontsize=11, fontweight='bold')
ax5.set_ylabel('Residuals (°C)', fontsize=11, fontweight='bold')
ax5.set_title('Residual Analysis', fontsize=12, fontweight='bold')
ax5.legend(fontsize=9)
ax5.grid(True, alpha=0.3)

# 6. Feature Importance
ax6 = fig.add_subplot(gs[2, 2])
importances = model.rf.feature_importances_
indices = np.argsort(importances)[-10:]
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(indices)))
ax6.barh(range(len(indices)), importances[indices], color=colors)
ax6.set_yticks(range(len(indices)))
labels = [model.feature_names[i] if i < len(model.feature_names) else f'feat_{i}' 
         for i in indices]
ax6.set_yticklabels(labels, fontsize=9)
ax6.set_xlabel('Importance', fontsize=11, fontweight='bold')
ax6.set_title('Top 10 Features', fontsize=12, fontweight='bold')
ax6.grid(True, alpha=0.3, axis='x')

plt.suptitle('Markov Chain Enhanced Weather Prediction Analysis', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.show()

# ===============================================================
# 9️⃣ Final Summary Report
# ===============================================================
print("\n" + "="*70)
print("📊 FINAL PERFORMANCE SUMMARY")
print("="*70)
print(f"\n{'Metric':<15} {'Validation':<25} {'Test':<25}")
print("-"*70)
for metric in ['R²', 'RMSE', 'MAE', 'MAPE']:
    print(f"{metric:<15} {val_metrics[metric]:<25.6f} {test_metrics[metric]:<25.6f}")
print("="*70)

overfitting_gap = (val_metrics['R²'] - test_metrics['R²']) * 100

print(f"\n📈 Performance Analysis:")
print(f"  • Validation Accuracy: {val_metrics['R²']*100:.2f}%")
print(f"  • Test Accuracy:       {test_metrics['R²']*100:.2f}%")
print(f"  • Overfitting Gap:     {overfitting_gap:.2f}%")

if overfitting_gap < 5:
    status = "🟢 EXCELLENT - Outstanding generalization!"
elif overfitting_gap < 10:
    status = "🟢 VERY GOOD - Strong generalization"
elif overfitting_gap < 15:
    status = "🟡 GOOD - Acceptable for production"
else:
    status = "🔴 MODERATE - Consider more regularization"

print(f"  • Status: {status}")

print(f"\n🎯 Markov Chain Contribution:")
print(f"  • States: {model.markov.n_states}")
print(f"  • Transition features: {model.markov.n_states}")
print(f"  • Additional features: 2 (entropy, expected_next)")
print(f"  • Captures temporal patterns in temperature transitions")

print("\n" + "="*70)
print("✨ Model successfully combines:")
print("   ✓ Markov Chains → Temporal transition patterns")
print("   ✓ Random Forest → Non-linear relationships")
print("   ✓ Rolling features → Trend capture")
print("="*70)
