import pandas as pd
import numpy as np
import os

print("Current Folder:", os.getcwd())
print("Files Here:", os.listdir())
# ============================
# LOAD DATASET
# ============================
df = pd.read_csv(r"C:\Users\Ram Anthony\OneDrive\Desktop\PAMPLIEGA\Lab6_NumPyandPandas_Pampliega\Most Popular Programming Languages.csv")

# Convert Month column to datetime
df["Month"] = pd.to_datetime(df["Month"])

# ============================
# ASSIGN LANGUAGE
# ============================
student_id = 250703
last_three = student_id % 1000

languages = sorted([
    "C# Worldwide(%)",
    "Flutter Worldwide(%)",
    "Java Worldwide(%)",
    "JavaScript Worldwide(%)",
    "Matlab Worldwide(%)",
    "PhP Worldwide(%)",
    "Python Worldwide(%)",
    "React Worldwide(%)",
    "Swift Worldwide(%)",
    "TypeScript Worldwide(%)"
])

language_index = last_three % len(languages)
assigned_language = languages[language_index]

# Second language for comparison
second_language = languages[(language_index + 1) % len(languages)]

print("Assigned Language:", assigned_language)
print("Second Language:", second_language)

# ============================
# TASK 1: GROWTH ANALYSIS
# ============================
lang_df = df[["Month", assigned_language]].copy()
lang_df.rename(columns={assigned_language: "Popularity"}, inplace=True)

lang_df["Growth_Rate"] = lang_df["Popularity"].pct_change() * 100
lang_df["Moving_Avg"] = lang_df["Popularity"].rolling(window=6).mean()
lang_df["Moving_STD"] = lang_df["Popularity"].rolling(window=6).std()

# Phase Classification
conditions = [
    lang_df["Growth_Rate"] > 5,
    lang_df["Growth_Rate"] < -5
]

choices = ["Growth", "Decline"]

lang_df["Phase"] = np.select(conditions, choices, default="Stable")

# Statistics
stats_summary = lang_df["Popularity"].describe()
mean_val = lang_df["Popularity"].mean()
median_val = lang_df["Popularity"].median()
std_val = lang_df["Popularity"].std()

initial_val = lang_df["Popularity"].iloc[0]
final_val = lang_df["Popularity"].iloc[-1]
overall_growth = ((final_val - initial_val) / initial_val) * 100

phase_counts = lang_df["Phase"].value_counts()

print("\n=== TASK 1: GROWTH ANALYSIS ===")
print(lang_df.head(12))
print("\nStatistical Summary:")
print(stats_summary)
print("\nPhase Counts:")
print(phase_counts)
print(f"\nInitial Popularity: {initial_val:.2f}")
print(f"Final Popularity: {final_val:.2f}")
print(f"Overall Growth: {overall_growth:.2f}%")

# ============================
# TASK 2: LIFECYCLE CLASSIFICATION
# ============================
growth_mean = np.nanmean(lang_df["Growth_Rate"])
growth_std = np.nanstd(lang_df["Growth_Rate"])

conditions2 = [
    (lang_df["Growth_Rate"] > 0) & (lang_df["Growth_Rate"] < growth_mean),
    (lang_df["Growth_Rate"] > growth_mean),
    (abs(lang_df["Growth_Rate"]) <= 1),
    (lang_df["Growth_Rate"] < 0) & (lang_df["Growth_Rate"] < (-1 * growth_std))
]

choices2 = ["Introduction", "Growth", "Maturity", "Decline"]

lang_df["Lifecycle_Phase"] = np.select(conditions2, choices2, default="Maturity")

lifecycle_counts = lang_df["Lifecycle_Phase"].value_counts()
dominant_stage = lifecycle_counts.idxmax()

print("\n=== TASK 2: LIFECYCLE CLASSIFICATION ===")
print(lang_df[["Month", "Popularity", "Growth_Rate", "Moving_Avg", "Moving_STD", "Lifecycle_Phase"]].head(12))
print("\nLifecycle Counts:")
print(lifecycle_counts)
print(f"\nDominant Stage: {dominant_stage}")

# ============================
# TASK 3: COMPARATIVE ANALYSIS
# ============================
comp_df = df[["Month", assigned_language, second_language]].copy()

A = comp_df[assigned_language]
B = comp_df[second_language]

# Stats
mean_A, mean_B = A.mean(), B.mean()
median_A, median_B = A.median(), B.median()
std_A, std_B = A.std(), B.std()

# Normalize
A_norm = (A - A.mean()) / A.std()
B_norm = (B - B.mean()) / B.std()

# Correlation
correlation = np.corrcoef(A, B)[0, 1]

# Dominance Ratio
dominance_count = (A > B).sum()
dominance_ratio = (dominance_count / len(comp_df)) * 100

# Relative Performance Index
rpi = (mean_A / mean_B) * 100

# Cross-over points
crossovers = comp_df[comp_df[assigned_language] > comp_df[second_language]]

summary_df = pd.DataFrame({
    "Mean_A": [mean_A],
    "Mean_B": [mean_B],
    "Std_A": [std_A],
    "Std_B": [std_B],
    "Correlation": [correlation],
    "Dominance_Ratio": [dominance_ratio],
    "RPI": [rpi]
})

print("\n=== TASK 3: COMPARATIVE ANALYSIS ===")
print(summary_df)
print("\nCross-over Points:")
print(crossovers.head(10))