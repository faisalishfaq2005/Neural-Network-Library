import numpy as np
import pandas as pd

# Set the random seed for reproducibility
np.random.seed(42)

# Number of samples and features
num_samples = 200
num_features = 3

# Generate random feature values between 0 and 10
features = np.random.uniform(0, 10, size=(num_samples, num_features))

# Generate binary labels based on a condition (e.g., sum of features > 15 for label 1, else 0)
labels = np.array([1 if np.sum(features[i]) > 15 else 0 for i in range(num_samples)])

# Create a DataFrame
df = pd.DataFrame(features, columns=[f"Feature_{i+1}" for i in range(num_features)])
df['Label'] = labels

# Save the DataFrame to a CSV file
df.to_csv('binary_classification_data.csv', index=False)