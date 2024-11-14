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

'''
directory = 'output/more_labels/500/'
directory2 = 'output/more_labels/refine/'
for filename in os.listdir(directory):
    f1 = os.path.join(directory, filename)
    f2 = os.path.join(directory2, filename)
    print(f1, f2)
    df1 = pd.read_csv(f1)
    df2 = pd.read_csv(f2)
    df_merged = merge_sort_reset(df1, df2, 'threshold')
    df_merged.to_csv('output/more_labels/' + filename, index = False)
    #print(df_merged)

'''

label = 'phone'

df5 = pd.read_csv("output/more_labels/accuracy500.csv")
df = pd.read_csv("output/" + label + "/accuracy500.csv")

'''
for i in df5:
    if i not in ['threshold','sum']:
        plt.plot(df5['threshold'], df5[i], label=i)
'''

plt.plot(df5['threshold'], df5[label], label="phone + 4 other", color='green', )
plt.plot(df['threshold'], df[label], label="only phone", color='blue')
#plt.plot(recall5['threshold'], recall5[label], label=label, color='brown')
#plt.plot(recall['threshold'], recall[label], label=label, color='orange')
#plt.plot(precision5['threshold'], precision5[label], label=label, color='black')
#plt.plot(precision['threshold'], precision[label], label=label, color='grey')

#plt.plot(recall['threshold'], recall['sum'], label='Recall', color='blue')
#plt.plot(precision['threshold'], precision['sum'], label='Precision', color='red')

plt.xlabel("threshold")
#plt.ylabel()
plt.legend()
plt.title('Accuracy(phone)')

plt.show()


