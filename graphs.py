import matplotlib.pyplot as plt
import pandas as pd

accuracy = pd.read_csv("./output/more_labels/500/accuracy500.csv")
recall = pd.read_csv("./output/more_labels/500/recall500.csv")
precision = pd.read_csv("./output/more_labels/500/precision500.csv")


plt.plot(accuracy['threshold'], accuracy['sum'], label='Accuracy', color='green')
plt.plot(recall['threshold'], recall['sum'], label='Recall', color='blue')
plt.plot(precision['threshold'], precision['sum'], label='Precision', color='red')

plt.legend()

plt.show()