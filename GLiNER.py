from gliner import GLiNER
import pandas as pd
import math


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

res = {''TP': 0, 'FP': 0, 'FN': 0}

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

for i in range(5):
    text = df.loc[i]['text']
    label_map = set_labels(label_map, df.loc[i])
    entities = model_small.predict_entities(text, all_labels, threshold=0.7)
    for entity in entities:
        if df.loc[i, entity['label']] == entity['text']:
            print(entity['text'], 'TP')
            res['TP'] += 1
            label_map[entity['label']] = False
        else:
            print(entity['text'], 'FP', df.loc[i, entity['label']])
            res['FP'] += 1
    for key in list(label_map.keys()):
        if label_map[key]:
            print('FN', key)
            res['FN'] += 1

print('FN:', FN, 'TP:', TP, 'FP:', FP)
print(res)

print()
