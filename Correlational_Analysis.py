import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("merged_log.csv")

# Ensure Timestamp is not treated as a numeric column
if "Timestamp" in df.columns:
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")  # Convert to datetime
    df = df.set_index("Timestamp")  # Set Timestamp as index (optional)

# Convert percentage columns (like "CPU Usage") to numeric if needed
if df["CPU_Usage"].dtype == "object":
    df["CPU_Usage"] = df["CPU_Usage"].str.replace("%", "").astype(float)

# Convert all possible numeric columns, ignoring non-numeric data
df = df.apply(pd.to_numeric, errors='coerce')  # Converts non-numeric to NaN

# Drop non-numeric columns (like Timestamp, if still present)
df = df.select_dtypes(include=["number"])

# Compute correlation matrix for all numerical columns
correlation_matrix = df.corr()

# Display correlation matrix
print(correlation_matrix)

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()
