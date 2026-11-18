import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/exp1.csv")
print(df.head())

# A graph to show accuracy differences between all of the models
accuracy = df.groupby(["model", "setting"])["accuracy"].mean()

print(accuracy)

for name in df["model"].unique():
    model_data = accuracy[name]
    plt.plot([5, 10, 15, 30], model_data, label = name)
plt.xlabel("Number of Features")
plt.ylabel("Accuracy")
plt.title("Experiment 1: Accuracy vs. Number of Features")
plt.xticks([5, 10, 15, 30])
plt.legend(loc = "upper left", bbox_to_anchor = (1,1))
plt.tight_layout()
plt.savefig("results/exp1_accuracy_graph.png")
plt.show()

# A graph to show recall differences between all of the models
recall = df.groupby(["model", "setting"])["recall"].mean()

for name in df["model"].unique():
    model_data = recall[name]
    plt.plot([5, 10, 15, 30], model_data, label=name)

plt.xlabel("Number of Features")
plt.ylabel("Average Recall")
plt.title("Experiment 1: Recall vs. Number of Features")
plt.xticks([5, 10, 15, 30])
plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.savefig("results/exp1_recall.png")
plt.show()