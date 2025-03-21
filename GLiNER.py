from gliner import GLiNER
import pandas as pd
import math
import ast
import mlflow
import os

thresholds = [0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
numberofinputs = 10
model_name = "vicgalle/gliner-small-pii"

mlflow.set_tracking_uri("http://10.20.160.89:5000")
mlflow.set_experiment("GLiNER")

def set_labels(labels, ser):
    for label_key in list(labels.keys()):
        labels[label_key] = not (type(ser[label_key]) is float and math.isnan(ser[label_key]))
    return labels


def token_label_pairs(token_label_list):
    pairs = []
    for i, x in enumerate(token_label_list):
        token_list = ast.literal_eval(token_label_list[i][0])
        label_list = ast.literal_eval(token_label_list[i][1])
        pairs.append([token_list, label_list])
    return pairs

def add_df_row(dframe, cthreshold):
    dframe.loc[len(dframe)] = [0] * (len(dframe.columns) - 1) + [cthreshold]
    return dframe


def calculate_metrics(tp, fp, fn):
    tn = token_num - tp - fp - fn
    return {
        'true_negatives': tn,
        'accuracy': ((tp + tn) / token_num) * 100,
        'recall': (tp / (tp + fn)) * 100 if (tp + fn) else 0,
        'precision': (tp / (tp + fp)) * 100 if (tp + fp) else 0
    }

# Loading the dataset into a dataframe and dropping unused columns
df_orig = pd.read_csv('./input/pii_dataset.csv', header=0)
df = df_orig.drop(columns=['document', 'prompt', 'prompt_id', 'len', 'trailing_whitespace'])

label_map = {'phone': True,
             'email': True,
             #'address': True,
             #'url': True,
             #'hobby': True
             }

# Used [ENT] types
all_labels = list(label_map.keys())
tokens_labels = token_label_pairs([list(i) for i in zip(df['tokens'], df['labels'])])

# Loading the module
model = GLiNER.from_pretrained(model_name, load_tokenizer=True)

columns = list(label_map.keys())
columns.extend(['sum', 'threshold'])
true_positives = pd.DataFrame(columns=columns)
false_positives = pd.DataFrame(columns=columns)
false_negatives = pd.DataFrame(columns=columns)
true_negatives = pd.DataFrame(columns=columns)
accuracy = pd.DataFrame(columns=columns)
recall = pd.DataFrame(columns=columns)
precision = pd.DataFrame(columns=columns)

for certainty_threshold in thresholds:
    true_positives = add_df_row(true_positives, certainty_threshold)
    false_positives = add_df_row(false_positives, certainty_threshold)
    false_negatives = add_df_row(false_negatives, certainty_threshold)
    for i in range(numberofinputs):
        text = df.loc[i]['text']
        label_map = set_labels(label_map, df.loc[i])
        entities = model.predict_entities(text, all_labels, threshold=certainty_threshold)
        for entity in entities:
            if df.loc[i, entity['label']] == entity['text']:
                true_positives.at[true_positives.index[-1], entity['label']] += 1
                true_positives.at[true_positives.index[-1], 'sum'] += 1
                label_map[entity['label']] = False
            else:
                false_positives.at[false_positives.index[-1], entity['label']] += 1
                false_positives.at[false_positives.index[-1], 'sum'] += 1

        for key in list(label_map.keys()):
            if label_map[key]:
                false_negatives.at[false_negatives.index[-1], key] += 1
                false_negatives.at[false_negatives.index[-1], 'sum'] += 1


# Calculate the total number of tokens
token_num = sum(len(tokens[0]) for tokens in tokens_labels)

with mlflow.start_run(run_name=f"GLiNER-{model_name}-{numberofinputs}"):
    # Log parameters
    mlflow.log_param("model_name", model_name)
    mlflow.log_param("number_of_inputs", numberofinputs)
    mlflow.log_param("labels", all_labels)
    mlflow.log_param("thresholds", thresholds)
    mlflow.log_param("token_num", token_num)

    # Calculate true negatives, accuracy, recall, and precision
    for i, row in true_positives.iterrows():
        for col in row.keys():
            if col != 'threshold':
                tp = true_positives.loc[i, col]
                fp = false_positives.loc[i, col]
                fn = false_negatives.loc[i, col]

                metrics = calculate_metrics(tp, fp, fn)

                true_negatives.loc[i, col] = metrics['true_negatives']
                accuracy.loc[i, col] = metrics['accuracy']
                recall.loc[i, col] = metrics['recall']
                precision.loc[i, col] = metrics['precision']
            else:
                # Copy threshold values
                threshold = row['threshold']  # Get correct threshold value
                true_negatives.loc[i, col] = threshold
                accuracy.loc[i, col] = threshold
                recall.loc[i, col] = threshold
                precision.loc[i, col] = threshold

            step = int(float(row['threshold']) * 100)
            mlflow.log_metrics({
                f"{col}/accuracy": accuracy.loc[i, col],
                f"{col}/precision": precision.loc[i, col],
                f"{col}/recall": recall.loc[i, col],
                f"{col}/true_positives": true_positives.loc[i, col],
                f"{col}/true_negatives": true_negatives.loc[i, col],
                f"{col}/false_positives": false_positives.loc[i, col],
                f"{col}/false_negatives": false_negatives.loc[i, col]
            }, step=step)
        

    path = f'output/email/{numberofinputs}/'

    metrics_files = {
        "true_positives": true_positives,
        "true_negatives": true_negatives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall
    }

    if not os.path.exists(path):
        os.makedirs(path)

    for name, df in metrics_files.items():
        file_path = f"./{path}{name}.csv"
        abs_path = os.path.abspath(file_path)
        df.to_csv(abs_path, index=False)
        mlflow.log_artifact(abs_path)


