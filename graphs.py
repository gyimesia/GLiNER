import matplotlib.pyplot as plt
import pandas as pd

accuracy5 = pd.read_csv("./output/more_labels/500/accuracy500.csv")
accuracy_phone = pd.read_csv("./output/phone/accuracy500.csv")

recall5 = pd.read_csv("./output/more_labels/500/recall500.csv")
recall_phone = pd.read_csv("./output/phone/recall500.csv")
precision5 = pd.read_csv("./output/more_labels/500/precision500.csv")
precision_phone = pd.read_csv("./output/phone/precision500.csv")



label = 'phone'

#plt.plot(accuracy5['threshold'], accuracy5[label], label=label, color='green')
#plt.plot(accuracy_phone['threshold'], accuracy_phone[label], label=label, color='green')
#plt.plot(recall5['threshold'], recall5[label], label=label, color='brown')
#plt.plot(recall_phone['threshold'], recall_phone[label], label=label, color='orange')
plt.plot(precision5['threshold'], precision5[label], label=label, color='black')
plt.plot(precision_phone['threshold'], precision_phone[label], label=label, color='grey')

#plt.plot(recall['threshold'], recall['sum'], label='Recall', color='blue')
#plt.plot(precision['threshold'], precision['sum'], label='Precision', color='red')

plt.legend()
#plt.title('Accuracy')

plt.show()