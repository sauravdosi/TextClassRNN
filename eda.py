import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

train_df = pd.read_json('./Data_Embedding/training.json')
val_df = pd.read_json('./Data_Embedding/validation.json')
test_df = pd.read_json('./Data_Embedding/test.json')

print(train_df)
print(train_df["stars"].unique())

print(val_df)
print(val_df["stars"].unique())

print(test_df)
print(test_df["stars"].unique())

# 1. Pie chart for counts of reviews in each dataset
review_counts = [len(train_df), len(val_df), len(test_df)]
labels = ['Train', 'Validation', 'Test']
plt.figure(figsize=(7, 7))
plt.pie(review_counts, labels=labels, autopct='%1.1f%%', startangle=140, colors=['skyblue', 'lightgreen', 'salmon'])
plt.title('Distribution of Review Counts Across Datasets')
plt.show()

# 2. Bar chart for ratings distribution across datasets
# Get counts of each rating (1-5) in each dataset
train_counts = train_df['stars'].value_counts().reindex(range(1, 6), fill_value=0)
val_counts = val_df['stars'].value_counts().reindex(range(1, 6), fill_value=0)
test_counts = test_df['stars'].value_counts().reindex(range(1, 6), fill_value=0)
print(train_counts)

# Create a DataFrame with counts for each rating across datasets
rating_counts = pd.DataFrame({
    'Train': train_counts,
    'Validation': val_counts,
    'Test': test_counts
}, index=range(1, 6))

# Plot the bar chart
x = np.arange(len(rating_counts.index))  # the label locations for ratings 1 to 5
width = 0.25  # width of the bars

fig, ax = plt.subplots(figsize=(10, 6))
bar1 = ax.bar(x - width, rating_counts['Train'], width, label='Train', color='skyblue')
bar2 = ax.bar(x, rating_counts['Validation'], width, label='Validation', color='lightgreen')
bar3 = ax.bar(x + width, rating_counts['Test'], width, label='Test', color='salmon')

# Add labels, title, and custom x-axis tick labels
ax.set_xlabel('Ratings')
ax.set_ylabel('Count of Ratings')
ax.set_title('Rating Distribution Across Train, Validation, and Test Sets')
ax.set_xticks(x)
ax.set_xticklabels(rating_counts.index)
ax.legend()

plt.show()

# Tokenize and calculate sequence lengths for each dataset
def calculate_sequence_lengths(df):
    df['sequence_length'] = df['text'].apply(lambda x: len(x.split()))
    return df

# Apply to train, validation, and test datasets
train_df = calculate_sequence_lengths(train_df)
val_df = calculate_sequence_lengths(val_df)
test_df = calculate_sequence_lengths(test_df)

# 1. Average sequence length for each dataset
train_avg_length = train_df['sequence_length'].mean()
val_avg_length = val_df['sequence_length'].mean()
test_avg_length = test_df['sequence_length'].mean()

print(f"Average sequence length (Train): {train_avg_length}")
print(f"Average sequence length (Validation): {val_avg_length}")
print(f"Average sequence length (Test): {test_avg_length}")

# 2. Average sequence length distribution across ratings for each dataset
# Group by 'stars' and calculate the mean sequence length
train_avg_length_by_rating = train_df.groupby('stars')['sequence_length'].mean().reindex(range(1, 6), fill_value=0)
val_avg_length_by_rating = val_df.groupby('stars')['sequence_length'].mean().reindex(range(1, 6), fill_value=0)
test_avg_length_by_rating = test_df.groupby('stars')['sequence_length'].mean().reindex(range(1, 6), fill_value=0)

# Plotting the bar chart
x = np.arange(1, 6)  # the label locations for ratings 1 to 5
width = 0.25  # width of the bars

fig, ax = plt.subplots(figsize=(10, 6))
bar1 = ax.bar(x - width, train_avg_length_by_rating, width, label='Train', color='skyblue')
bar2 = ax.bar(x, val_avg_length_by_rating, width, label='Validation', color='lightgreen')
bar3 = ax.bar(x + width, test_avg_length_by_rating, width, label='Test', color='salmon')

# Add labels, title, and custom x-axis tick labels
ax.set_xlabel('Ratings')
ax.set_ylabel('Average Sequence Length')
ax.set_title('Average Sequence Length Distribution Across Ratings')
ax.set_xticks(x)
ax.set_xticklabels(range(1, 6))
ax.legend()

plt.show()

