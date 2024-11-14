import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('./best_ffnn_results.csv')

# final_epoch_df = df[df['epoch'] == 20]
# grouped_df = final_epoch_df.groupby('momentum')['val_accuracy'].mean().reset_index()
# print(grouped_df)
# max_accuracy_record = df.loc[df['val_accuracy'].idxmax()]
# print(max_accuracy_record)

# best_ffnn_df = df[(df["batch_size"] == 512) & (df["learning_rate"] == 0.05) & (df["momentum"] == 0.9) & (df["hidden_dim"] == 20)]



# print(best_ffnn_df)

# Plotting the parameters as line graphs
plt.plot(df['epoch'], df['loss'], label='Loss', color='red')
plt.plot(df['epoch'], df['train_accuracy'], label='Train Accuracy', color='blue')
plt.plot(df['epoch'], df['val_accuracy'], label='Validation Accuracy', color='green')

# Adding labels and title
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.title('Loss and Accuracy vs Iterations for Best FFNN Run')

# Display legend
plt.legend()

# Show the plot
plt.show()