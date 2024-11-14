import matplotlib.pyplot as plt
import pandas as pd

label = 'address'


accuracy5 = pd.read_csv("./output/more_labels/500/acc.csv")
accuracy = pd.read_csv("./output/"+ label + "/accuracy500.csv")

recall5 = pd.read_csv("./output/more_labels/500/recall500.csv")
recall = pd.read_csv("./output/"+ label + "/recall500.csv")
precision5 = pd.read_csv("./output/more_labels/500/precision500.csv")
precision = pd.read_csv("./output/"+ label + "/precision500.csv")

df5 = pd.read_csv("output/more_labels/500/accuracy500.csv")
df5ref = pd.read_csv("output/more_labels/refine/accuracy500.csv")

dfcon = pd.concat([df5, df5ref], ignore_index=True)
dfcon =
#dfcon = dfcon.sort_values("threshold")
#dfcon = dfcon.drop_duplicates('threshold')
print(dfcon)

exit(0)
plt.plot(accuracy5['threshold'], accuracy5[label], label=label, color='green')
plt.plot(accuracy['threshold'], accuracy[label], label=label, color='green')
#plt.plot(recall5['threshold'], recall5[label], label=label, color='brown')
#plt.plot(recall['threshold'], recall[label], label=label, color='orange')
#plt.plot(precision5['threshold'], precision5[label], label=label, color='black')
#plt.plot(precision['threshold'], precision[label], label=label, color='grey')

#plt.plot(recall['threshold'], recall['sum'], label='Recall', color='blue')
#plt.plot(precision['threshold'], precision['sum'], label='Precision', color='red')

plt.legend()
#plt.title('Accuracy')

plt.show()