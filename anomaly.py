# Anomaly Detection in User Behavior Data
# This notebook demonstrates anomaly detection using Isolation Forest and Autoencoder approaches

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("Libraries imported successfully!")

# ## 1. Generate Dummy User Behavior Data

def generate_user_behavior_data(n_samples=10000, anomaly_fraction=0.05):
    """
    Generate synthetic user behavior data with normal and anomalous patterns
    """
    n_anomalies = int(n_samples * anomaly_fraction)
    n_normal = n_samples - n_anomalies
    
    # Normal user behavior patterns
    normal_data = {
        'session_duration': np.random.normal(25, 8, n_normal),  # minutes
        'pages_visited': np.random.poisson(12, n_normal),
        'clicks_per_session': np.random.normal(45, 15, n_normal),
        'time_between_clicks': np.random.exponential(2, n_normal),  # seconds
        'bounce_rate': np.random.beta(2, 8, n_normal),  # 0-1
        'scroll_depth': np.random.beta(5, 2, n_normal),  # 0-1
        'device_score': np.random.normal(0.7, 0.2, n_normal),  # trust score
        'hour_of_day': np.random.choice(range(24), n_normal, 
                                       p=[0.02, 0.01, 0.01, 0.01, 0.01, 0.02, 0.03, 0.05,
                                          0.08, 0.12, 0.15, 0.12, 0.08, 0.06, 0.05, 0.04,
                                          0.03, 0.03, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02])
    }
    
    # Anomalous user behavior patterns (bots, fraudsters, etc.)
    anomaly_data = {
        'session_duration': np.concatenate([
            np.random.normal(2, 1, n_anomalies//3),      # Very short sessions
            np.random.normal(120, 20, n_anomalies//3),   # Very long sessions
            np.random.normal(25, 8, n_anomalies//3)      # Some normal mixed in
        ]),
        'pages_visited': np.concatenate([
            np.random.poisson(50, n_anomalies//2),       # Excessive page visits
            np.random.poisson(1, n_anomalies//2)         # Very few page visits
        ]),
        'clicks_per_session': np.concatenate([
            np.random.normal(200, 50, n_anomalies//2),   # Excessive clicking
            np.random.normal(5, 2, n_anomalies//2)       # Very few clicks
        ]),
        'time_between_clicks': np.concatenate([
            np.random.exponential(0.1, n_anomalies//2),  # Very fast clicking
            np.random.exponential(10, n_anomalies//2)    # Very slow clicking
        ]),
        'bounce_rate': np.concatenate([
            np.random.beta(8, 2, n_anomalies//2),        # High bounce rate
            np.random.beta(1, 10, n_anomalies//2)        # Unusually low bounce rate
        ]),
        'scroll_depth': np.concatenate([
            np.random.beta(1, 5, n_anomalies//2),        # Very shallow scrolling
            np.ones(n_anomalies//2)                      # Perfect scrolling (bot-like)
        ]),
        'device_score': np.concatenate([
            np.random.normal(0.1, 0.05, n_anomalies//2), # Low trust devices
            np.random.normal(1.0, 0.01, n_anomalies//2)  # Suspiciously perfect scores
        ]),
        'hour_of_day': np.random.choice(range(24), n_anomalies, 
                                       p=[0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.02, 0.02,
                                          0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02,
                                          0.02, 0.02, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08])
    }
    
    # Combine normal and anomalous data
    data = {}
    labels = []
    
    for feature in normal_data.keys():
        data[feature] = np.concatenate([normal_data[feature], anomaly_data[feature]])
    
    labels = np.concatenate([np.zeros(n_normal), np.ones(n_anomalies)])
    
    # Create DataFrame
    df = pd.DataFrame(data)
    df['is_anomaly'] = labels
    
    # Add some feature engineering
    df['clicks_per_minute'] = df['clicks_per_session'] / df['session_duration']
    df['pages_per_minute'] = df['pages_visited'] / df['session_duration']
    
    # Shuffle the data
    df = df.sample(frac=1).reset_index(drop=True)
    
    return df

# Generate the dataset
print("Generating user behavior data...")
df = generate_user_behavior_data(n_samples=10000, anomaly_fraction=0.05)
print(f"Dataset generated: {len(df)} samples, {df['is_anomaly'].sum()} anomalies ({df['is_anomaly'].mean():.1%})")
print("\nDataset Info:")
print(df.info())
print("\nFirst few rows:")
print(df.head())

# ## 2. Exploratory Data Analysis

# Basic statistics
print("\nDataset Statistics:")
print(df.describe())

# Visualize distributions
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.ravel()

numerical_features = ['session_duration', 'pages_visited', 'clicks_per_session', 
                     'time_between_clicks', 'bounce_rate', 'scroll_depth', 
                     'device_score', 'clicks_per_minute', 'pages_per_minute']

for i, feature in enumerate(numerical_features):
    df[df['is_anomaly'] == 0][feature].hist(alpha=0.7, bins=30, label='Normal', ax=axes[i])
    df[df['is_anomaly'] == 1][feature].hist(alpha=0.7, bins=30, label='Anomaly', ax=axes[i])
    axes[i].set_title(feature)
    axes[i].legend()

plt.tight_layout()
plt.show()

# Correlation matrix
plt.figure(figsize=(10, 8))
correlation_matrix = df[numerical_features].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Feature Correlation Matrix')
plt.show()

# ## 3. Data Preprocessing

# Separate features and labels
X = df[numerical_features].copy()
y = df['is_anomaly'].copy()

# Handle any infinite or NaN values
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median())

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training set: {X_train_scaled.shape[0]} samples")
print(f"Test set: {X_test_scaled.shape[0]} samples")
print(f"Training anomalies: {y_train.sum()} ({y_train.mean():.1%})")
print(f"Test anomalies: {y_test.sum()} ({y_test.mean():.1%})")

# ## 4. Anomaly Detection with Isolation Forest

print("\n" + "="*50)
print("ISOLATION FOREST APPROACH")
print("="*50)

# Initialize and train Isolation Forest
iso_forest = IsolationForest(
    contamination=0.05,  # Expected proportion of anomalies
    random_state=42,
    n_estimators=100
)

# Fit on training data (unsupervised)
iso_forest.fit(X_train_scaled)

# Make predictions
y_pred_train_iso = iso_forest.predict(X_train_scaled)
y_pred_test_iso = iso_forest.predict(X_test_scaled)

# Convert predictions to binary (1 for anomaly, 0 for normal)
# Isolation Forest returns -1 for anomalies, 1 for normal
y_pred_train_iso_binary = (y_pred_train_iso == -1).astype(int)
y_pred_test_iso_binary = (y_pred_test_iso == -1).astype(int)

# Get anomaly scores
anomaly_scores_train = iso_forest.decision_function(X_train_scaled)
anomaly_scores_test = iso_forest.decision_function(X_test_scaled)

print("Isolation Forest Results:")
print("\nTraining Set Performance:")
print(classification_report(y_train, y_pred_train_iso_binary))
print("\nTest Set Performance:")
print(classification_report(y_test, y_pred_test_iso_binary))

# Confusion Matrix
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
cm_train = confusion_matrix(y_train, y_pred_train_iso_binary)
sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues')
plt.title('Training Set - Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

plt.subplot(1, 2, 2)
cm_test = confusion_matrix(y_test, y_pred_test_iso_binary)
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues')
plt.title('Test Set - Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

plt.tight_layout()
plt.show()

# Plot anomaly scores distribution
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.hist(anomaly_scores_test[y_test == 0], bins=30, alpha=0.7, label='Normal', density=True)
plt.hist(anomaly_scores_test[y_test == 1], bins=30, alpha=0.7, label='Anomaly', density=True)
plt.xlabel('Anomaly Score')
plt.ylabel('Density')
plt.title('Isolation Forest - Anomaly Score Distribution')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(range(len(anomaly_scores_test)), sorted(anomaly_scores_test), c='blue', alpha=0.6)
plt.axhline(y=iso_forest.threshold_, color='red', linestyle='--', label=f'Threshold: {iso_forest.threshold_:.3f}')
plt.xlabel('Sample Index (sorted)')
plt.ylabel('Anomaly Score')
plt.title('Isolation Forest - Sorted Anomaly Scores')
plt.legend()

plt.tight_layout()
plt.show()

# ## 5. Anomaly Detection with Autoencoder

print("\n" + "="*50)
print("AUTOENCODER APPROACH")
print("="*50)

# Build Autoencoder model
def build_autoencoder(input_dim, encoding_dim=5):
    """
    Build a simple autoencoder for anomaly detection
    """
    # Input layer
    input_layer = Input(shape=(input_dim,))
    
    # Encoder
    encoded = Dense(encoding_dim * 2, activation='relu')(input_layer)
    encoded = Dense(encoding_dim, activation='relu')(encoded)
    
    # Decoder
    decoded = Dense(encoding_dim * 2, activation='relu')(encoded)
    decoded = Dense(input_dim, activation='linear')(decoded)
    
    # Autoencoder model
    autoencoder = Model(input_layer, decoded)
    
    return autoencoder

# Build and compile the autoencoder
input_dim = X_train_scaled.shape[1]
autoencoder = build_autoencoder(input_dim, encoding_dim=4)

autoencoder.compile(optimizer=Adam(learning_rate=0.001), 
                   loss='mse', 
                   metrics=['mae'])

print("Autoencoder Architecture:")
autoencoder.summary()

# Train the autoencoder on normal data only
X_train_normal = X_train_scaled[y_train == 0]
print(f"\nTraining autoencoder on {len(X_train_normal)} normal samples...")

history = autoencoder.fit(
    X_train_normal, X_train_normal,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=1,
    shuffle=True
)

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Autoencoder Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Autoencoder Training MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()

plt.tight_layout()
plt.show()

# Make predictions and calculate reconstruction errors
X_train_pred = autoencoder.predict(X_train_scaled, verbose=0)
X_test_pred = autoencoder.predict(X_test_scaled, verbose=0)

# Calculate reconstruction errors (MSE)
train_mse = np.mean(np.square(X_train_scaled - X_train_pred), axis=1)
test_mse = np.mean(np.square(X_test_scaled - X_test_pred), axis=1)

print(f"Training reconstruction error - Mean: {train_mse.mean():.4f}, Std: {train_mse.std():.4f}")
print(f"Test reconstruction error - Mean: {test_mse.mean():.4f}, Std: {test_mse.std():.4f}")

# Determine threshold for anomaly detection
# Use 95th percentile of normal training data reconstruction error
threshold = np.percentile(train_mse[y_train == 0], 95)
print(f"Anomaly threshold (95th percentile of normal training data): {threshold:.4f}")

# Make binary predictions
y_pred_train_ae = (train_mse > threshold).astype(int)
y_pred_test_ae = (test_mse > threshold).astype(int)

print("\nAutoencoder Results:")
print("\nTraining Set Performance:")
print(classification_report(y_train, y_pred_train_ae))
print("\nTest Set Performance:")
print(classification_report(y_test, y_pred_test_ae))

# Confusion Matrix for Autoencoder
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
cm_train_ae = confusion_matrix(y_train, y_pred_train_ae)
sns.heatmap(cm_train_ae, annot=True, fmt='d', cmap='Greens')
plt.title('Autoencoder Training - Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

plt.subplot(1, 2, 2)
cm_test_ae = confusion_matrix(y_test, y_pred_test_ae)
sns.heatmap(cm_test_ae, annot=True, fmt='d', cmap='Greens')
plt.title('Autoencoder Test - Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

plt.tight_layout()
plt.show()

# Plot reconstruction error distributions
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.hist(test_mse[y_test == 0], bins=30, alpha=0.7, label='Normal', density=True)
plt.hist(test_mse[y_test == 1], bins=30, alpha=0.7, label='Anomaly', density=True)
plt.axvline(x=threshold, color='red', linestyle='--', label=f'Threshold: {threshold:.4f}')
plt.xlabel('Reconstruction Error (MSE)')
plt.ylabel('Density')
plt.title('Autoencoder - Reconstruction Error Distribution')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(range(len(test_mse)), sorted(test_mse), c='green', alpha=0.6)
plt.axhline(y=threshold, color='red', linestyle='--', label=f'Threshold: {threshold:.4f}')
plt.xlabel('Sample Index (sorted)')
plt.ylabel('Reconstruction Error')
plt.title('Autoencoder - Sorted Reconstruction Errors')
plt.legend()

plt.tight_layout()
plt.show()

# ## 6. Model Comparison

print("\n" + "="*50)
print("MODEL COMPARISON")
print("="*50)

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

def evaluate_model(y_true, y_pred, scores, model_name):
    """Evaluate anomaly detection model"""
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, scores)
    
    return {
        'Model': model_name,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'AUC-ROC': auc
    }

# Evaluate both models
iso_results = evaluate_model(y_test, y_pred_test_iso_binary, -anomaly_scores_test, 'Isolation Forest')
ae_results = evaluate_model(y_test, y_pred_test_ae, test_mse, 'Autoencoder')

# Create comparison DataFrame
comparison_df = pd.DataFrame([iso_results, ae_results])
print("\nModel Performance Comparison:")
print(comparison_df.round(4))

# Plot ROC curves
from sklearn.metrics import roc_curve

plt.figure(figsize=(10, 6))

# Isolation Forest ROC
fpr_iso, tpr_iso, _ = roc_curve(y_test, -anomaly_scores_test)
plt.plot(fpr_iso, tpr_iso, label=f'Isolation Forest (AUC = {iso_results["AUC-ROC"]:.3f})', linewidth=2)

# Autoencoder ROC
fpr_ae, tpr_ae, _ = roc_curve(y_test, test_mse)
plt.plot(fpr_ae, tpr_ae, label=f'Autoencoder (AUC = {ae_results["AUC-ROC"]:.3f})', linewidth=2)

# Random classifier
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves - Anomaly Detection Models')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# ## 7. Feature Importance Analysis (Isolation Forest)

# Get feature importance from Isolation Forest
feature_importance = pd.DataFrame({
    'feature': numerical_features,
    'importance': np.abs(iso_forest.score_samples(X_test_scaled).std())  # Approximation
})

# Alternative: Use permutation-based feature importance
from sklearn.inspection import permutation_importance

print("\nCalculating feature importance...")
perm_importance = permutation_importance(
    iso_forest, X_test_scaled, y_test, 
    n_repeats=10, random_state=42, scoring='f1'
)

feature_importance_df = pd.DataFrame({
    'feature': numerical_features,
    'importance_mean': perm_importance.importances_mean,
    'importance_std': perm_importance.importances_std
}).sort_values('importance_mean', ascending=False)

print("\nFeature Importance (Isolation Forest):")
print(feature_importance_df)

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['feature'], feature_importance_df['importance_mean'])
plt.xlabel('Importance')
plt.title('Feature Importance for Anomaly Detection')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# ## 8. Example Predictions

print("\n" + "="*50)
print("EXAMPLE PREDICTIONS")
print("="*50)

# Show some example predictions
n_examples = 10
example_indices = np.random.choice(len(X_test), n_examples, replace=False)

example_df = pd.DataFrame({
    'Index': example_indices,
    'True_Label': y_test.iloc[example_indices].values,
    'ISO_Prediction': y_pred_test_iso_binary[example_indices],
    'ISO_Score': anomaly_scores_test[example_indices],
    'AE_Prediction': y_pred_test_ae[example_indices],
    'AE_Score': test_mse[example_indices]
})

print("Example Predictions:")
print(example_df.round(4))

# Show detailed analysis for one anomaly
if np.any(y_test == 1):
    anomaly_idx = np.where(y_test == 1)[0][0]
    print(f"\nDetailed Analysis for Anomaly (Index {anomaly_idx}):")
    print("Original features:")
    for i, feature in enumerate(numerical_features):
        print(f"  {feature}: {X_test.iloc[anomaly_idx, i]:.3f}")
    
    print(f"\nIsolation Forest Score: {anomaly_scores_test[anomaly_idx]:.4f}")
    print(f"Autoencoder Reconstruction Error: {test_mse[anomaly_idx]:.4f}")

print("\n" + "="*50)
print("ANALYSIS COMPLETE")
print("="*50)
print("\nKey Findings:")
print("1. Generated synthetic user behavior data with realistic patterns")
print("2. Implemented and compared Isolation Forest vs Autoencoder approaches")
print("3. Both models can detect anomalies, with different strengths:")
print("   - Isolation Forest: Good for global outliers, interpretable")
print("   - Autoencoder: Good for complex patterns, captures feature interactions")
print("4. Feature importance analysis helps understand what drives anomaly detection")
print("5. Threshold tuning is crucial for both approaches")

# Save results
results_summary = {
    'isolation_forest_f1': iso_results['F1-Score'],
    'autoencoder_f1': ae_results['F1-Score'],
    'isolation_forest_auc': iso_results['AUC-ROC'],
    'autoencoder_auc': ae_results['AUC-ROC'],
    'dataset_size': len(df),
    'anomaly_rate': df['is_anomaly'].mean()
}

print(f"\nResults Summary: {results_summary}")