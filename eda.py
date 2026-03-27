import pandas as pd
import numpy as np

# Load data
train_df = pd.read_csv('forward_prediction/train.csv')
labels_df = pd.read_csv('forward_prediction/train_labels.csv')

# Combine for correlation check
combined_df = pd.concat([train_df, labels_df], axis=1)

print("--- Train Data Info ---")
print(train_df.info())
print("\n--- Labels Data Info ---")
print(labels_df.info())

print("\n--- Descriptive Statistics ---")
print(combined_df.describe().T)

print("\n--- Correlations with Labels ---")
corr_matrix = combined_df.corr()
for label in labels_df.columns:
    print(f"\nCorrelations for {label}:")
    print(corr_matrix[label].sort_values(ascending=False))
