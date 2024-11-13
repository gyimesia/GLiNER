from gliner import GLiNER
import pandas as pd
import math
import ast
import matplotlib.pyplot as plt

numberofevals = 10


def set_labels(labels, ser):
    for label_key in list(labels.keys()):
        labels[label_key] = not (type(ser[label_key]) is float and math.isnan(ser[label_key]))
    return labels


def token_label_pairs(tokens_labels):
    pairs = []
    for i, x in enumerate(tokens_labels):
        token_list = ast.literal_eval(tokens_labels[i][0])
        label_list = ast.literal_eval(tokens_labels[i][1])
        pairs.append([token_list, label_list])
    return pairs


def add_df_row(dframe, cthreshold, rownum):
    for i in dframe:
        dframe.loc[rownum, i] = 0
    dframe.loc[rownum, 'threshold'] = cthreshold
    return dframe


# Loading the dataset into a dataframe and dropping unused columns
df_orig = pd.read_csv('./input/pii_dataset.csv', header=0)
df = df_orig.drop(columns=['document', 'prompt', 'prompt_id', 'len', 'trailing_whitespace'])

label_map = {'hobby': True,
             'phone': True,
             'email': True,
             'address': True,
             'url': True}

# Used [ENT] types
all_labels = list(label_map.keys())
tokens_labels = token_label_pairs([list(i) for i in zip(df['tokens'], df['labels'])])

# Loading the module
model_small = GLiNER.from_pretrained("vicgalle/gliner-small-pii", load_tokenizer="True")

columns = list(label_map.keys())
columns.extend(['sum', 'threshold'])
true_positives = pd.DataFrame(columns=columns)
false_positives = pd.DataFrame(columns=columns)
false_negatives = pd.DataFrame(columns=columns)
true_negatives = pd.DataFrame(columns=columns)
model_accuracy = pd.DataFrame(columns=columns)
model_recall = pd.DataFrame(columns=columns)
model_precision = pd.DataFrame(columns=columns)

for eval_round in range(numberofevals):
    certainty_threshold = round(eval_round * 0.05 + 0.54, 2)
    print(eval_round, certainty_threshold)
    true_positives = add_df_row(true_positives, certainty_threshold, eval_round)
    false_positives = add_df_row(false_positives, certainty_threshold, eval_round)
    false_negatives = add_df_row(false_negatives, certainty_threshold, eval_round)

    for i in range(200):
        text = df.loc[i]['text']
        label_map = set_labels(label_map, df.loc[i])
        entities = model_small.predict_entities(text, all_labels, threshold=certainty_threshold)

        for entity in entities:
            if df.loc[i, entity['label']] == entity['text']:
                true_positives.loc[eval_round, entity['label']] += 1
                true_positives.loc[eval_round, 'sum'] += 1
                label_map[entity['label']] = False
            else:
                false_positives.loc[eval_round, entity['label']] += 1
                false_positives.loc[eval_round, 'sum'] += 1

        for key in list(label_map.keys()):
            if label_map[key]:
                false_negatives.loc[eval_round, key] += 1
                false_negatives.loc[eval_round, 'sum'] += 1


# Accuracy: (TP+TN)/token_num *
# Precision: TP/(TP+FP)
# Recall: TP/(TP+FN) *

# Calculating true negatives, accuracy, precision and recall

for i, row in true_positives.iterrows():
    token_num = len(tokens_labels[i][0])
    for col in list(row.keys()):
        if col != 'threshold':
            true_negatives.loc[i, col] = token_num - true_positives.loc[i, col] - false_positives.loc[i, col] - \
                                         false_negatives.loc[i, col]
            model_accuracy.loc[i, col] = (true_positives.loc[i, col] + true_negatives.loc[i, col]) / token_num
            model_recall.loc[i, col] = true_positives.loc[i, col] / (true_positives.loc[i, col] + false_negatives.loc[i, col])
            model_precision.loc[i, col] = (true_positives.loc[i, col] + false_positives.loc[i, col]) and\
                                          (true_positives.loc[i, col] / (true_positives.loc[i, col] + false_positives.loc[i, col]))
        else:
            true_negatives.loc[i, col] = true_positives.loc[i, col]
            model_accuracy.loc[i, col] = true_positives.loc[i, col]
            model_recall.loc[i, col] = true_positives.loc[i, col]
            model_precision.loc[i, col] = true_positives.loc[i, col]



plt.plot(model_accuracy['threshold'], model_accuracy['email'], label='Accuracy', color='green')
plt.plot(model_recall['threshold'], model_recall['email'], label='Recall', color='blue')
plt.plot(model_precision['threshold'], model_precision['email'], label='Precision', color='red')

plt.legend()

plt.show()
