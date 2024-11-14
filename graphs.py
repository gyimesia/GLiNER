import matplotlib.pyplot as plt
import pandas as pd
import os


def merge_sort_reset(df1, df2, sort_col):
    merged_df = pd.concat([df1, df2], ignore_index=True)
    merged_df = merged_df.loc[:, ~merged_df.columns.str.contains('^Unnamed')]
    merged_df = merged_df.drop_duplicates('threshold')
    sorted_df = merged_df.sort_values(by=sort_col)
    sorted_df.reset_index(drop=True, inplace=True)
    return sorted_df

directory = 'output/phone/500/'
directory2 = 'output/phone/refine/'
for filename in os.listdir(directory):
    f1 = os.path.join(directory, filename)
    f2 = os.path.join(directory2, filename)
    print(f1, f2)
    df1 = pd.read_csv(f1)
    df2 = pd.read_csv(f2)
    df_merged = merge_sort_reset(df1, df2, 'threshold')
    df_merged.to_csv('output/phone/' + filename)
    #print(df_merged)

'''
label = 'address'




df5 = pd.read_csv("output/more_labels/500/accuracy500.csv")
df5ref = pd.read_csv("output/more_labels/refine/accuracy500.csv")

dfcon = merge_sort_reset(df5, df5ref, 'threshold')
print(dfcon)

#lt.plot(accuracy5['threshold'], accuracy5[label], label=label, color='green')
#lt.plot(accuracy['threshold'], accuracy[label], label=label, color='green')
#plt.plot(recall5['threshold'], recall5[label], label=label, color='brown')
#plt.plot(recall['threshold'], recall[label], label=label, color='orange')
#plt.plot(precision5['threshold'], precision5[label], label=label, color='black')
#plt.plot(precision['threshold'], precision[label], label=label, color='grey')

#plt.plot(recall['threshold'], recall['sum'], label='Recall', color='blue')
#plt.plot(precision['threshold'], precision['sum'], label='Precision', color='red')

#plt.legend()
#plt.title('Accuracy')

plt.show()


'''