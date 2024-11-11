from gliner import GLiNER
import pandas as pd
import math
import time


def set_labels(labels, ser):
    for label_key in list(labels.keys()):
        labels[label_key] = not (type(ser[label_key]) is float and math.isnan(ser[label_key]))
    return labels


# Loading the dataset into a dataframe and dropping unused columns
df_orig = pd.read_csv('./input/pii_dataset.csv', header=0)
df = df_orig.drop(columns=['document', 'prompt', 'prompt_id', 'len', 'trailing_whitespace', 'hobby'])
df = df.rename(columns={"name": "person", "phone": "phone number"})


# extractLabels.py gathers the list of used BIO labels in the dataset
# BIO labels and [ENT] type labels mapped manually

res = []

label_map = {  # 'job': True,
              'phone number': True,
              'email': True,
              'address': True,
              # 'username': True,
              'url': True}

# Used [ENT] types
all_labels = list(label_map.keys())

# Loading the module
model_small = GLiNER.from_pretrained("vicgalle/gliner-small-pii", load_tokenizer="True")

for eval_round in range(0, 5, 1):
    certainty_threshold = eval_round * 0.1 + 0.58
    print(eval_round, certainty_threshold)
    res.append({'TH': certainty_threshold, 'TP': 0, 'FP': 0, 'FN': 0})
    for i in range(100):
        text = df.loc[i]['text']
        label_map = set_labels(label_map, df.loc[i])
        entities = model_small.predict_entities(text, all_labels, threshold=certainty_threshold)
        for entity in entities:
            if df.loc[i, entity['label']] == entity['text']:
                # print(entity['text'], 'TP')
                res[eval_round]['TP'] += 1
                label_map[entity['label']] = False
            else:
                # print(entity['text'], 'FP', df.loc[i, entity['label']])
                res[eval_round]['FP'] += 1
        for key in list(label_map.keys()):
            if label_map[key]:
                # print('FN', key)
                res[eval_round]['FN'] += 1

    print(res[eval_round])
    time.sleep(5)

print(res)
