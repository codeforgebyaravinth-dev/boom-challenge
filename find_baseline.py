import pandas as pd
import numpy as np

# Load data
train_df = pd.read_csv('forward_prediction/train.csv')
labels_df = pd.read_csv('forward_prediction/train_labels.csv')

# Combine
combined = pd.concat([train_df, labels_df], axis=1)

# Filter by constraints
valid_mask = (combined['P80'] >= 96) & (combined['P80'] <= 101) & (combined['R95'] <= 175)
valid_points = combined[valid_mask]

print(f"Number of valid points in training data: {len(valid_points)}")

if len(valid_points) > 0:
    # Sort by energy
    best_point = valid_points.sort_values('energy').iloc[0]
    print("\n--- Best Point in Training Data ---")
    print(best_point)
else:
    print("\nNo points in training data satisfy the constraints.")
    # Check for near matches
    near_mask = (combined['P80'] >= 90) & (combined['P80'] <= 110) & (combined['R95'] <= 200)
    near_points = combined[near_mask]
    print(f"Number of near-match points: {len(near_points)}")
    if len(near_points) > 0:
        print("\n--- Best Near-Match Point ---")
        print(near_points.sort_values('energy').iloc[0])
