from gliner import GLiNER
import pandas as pd
import math
import ast
import os
import time
from mlflow_utils import ModelLogger

def set_labels(labels, ser):
    for label_key in list(labels.keys()):
        labels[label_key] = not (type(ser[label_key]) is float and math.isnan(ser[label_key]))
    return labels

def token_label_pairs(token_label_list):
    pairs = []
    for i, x in enumerate(token_label_list):
        token_list = ast.literal_eval(token_label_list[i][0])
        labels = ast.literal_eval(token_label_list[i][1])
        pairs.append([token_list, labels])
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

def prepare_dataset(data, numbertoinputs=100):
    # Loading the dataset into a dataframe and dropping unused columns
    df_orig = pd.read_csv(data, header=0)
    df = df_orig.drop(columns=['document', 'prompt', 'prompt_id', 'len', 'trailing_whitespace'])
    return df[:numbertoinputs].reset_index(drop=True)

def labellist_to_labeldict(labels):
    label_dict = {}
    for i in labels:
        if i not in label_dict:
            label_dict[i] = True
    return label_dict

def prepare_result_dataframes(columns):
        # Create empty dataframe with columns plus 'sum' and 'threshold'
        columns = columns.copy()  # Create a copy to avoid modifying the original
        columns.extend(['sum', 'threshold'])
        return pd.DataFrame(columns=columns)

def test(dataset, thresholds, label_list):
    # Prepare the result dataframes
    true_positives = prepare_result_dataframes(label_list)
    false_positives = prepare_result_dataframes(label_list)
    false_negatives = prepare_result_dataframes(label_list)
    true_negatives = prepare_result_dataframes(label_list)

    # Initialize the model
    model = GLiNER.from_pretrained(model_name, load_tokenizer=True)
    # Create a label map
    label_map = labellist_to_labeldict(label_list)
    for certainty_threshold in thresholds:
        true_positives = add_df_row(true_positives, certainty_threshold)
        false_positives = add_df_row(false_positives, certainty_threshold)
        false_negatives = add_df_row(false_negatives, certainty_threshold)
        for index, row in dataset.iterrows():
            text = row['text']
            label_map = set_labels(label_map, row)
            entities = model.predict_entities(text, label_list, threshold=certainty_threshold)
            for entity in entities:
                if row[entity['label']] == entity['text']:
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
    return true_positives, false_positives, false_negatives, true_negatives


# Set environment variables before any MLflow imports
os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"
os.environ["MLFLOW_SYSTEM_METRICS_SAMPLES_BEFORE_LOG"] = "1"
os.environ["MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL"] = "1"
os.environ["MLFLOW_SYSTEM_METRICS_DEFAULT_TAGS"] = "cpu,memory,disk.io"
os.environ["MLFLOW_SYSTEM_METRICS_SYNCHRONOUS_LOGGING"] = "true"

if __name__ == "__main__":
    experiment_name = "GLiNER"
    tracking_uri = "http://10.20.160.89:5000"
    # Initialize the ModelLogger
    logger = ModelLogger(experiment_name, tracking_uri=tracking_uri)

    thresholds = [0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
    numberofinputs = 5
    model_name = "vicgalle/gliner-small-pii"
    #model_name = "gretelai/gretel-gliner-bi-small-v1.0"
    labels = ['phone',
              #'email',
              #'address',
              #'url',
              #'hobby'
              ]

    # Load the dataset
    dataset_file = './input/pii_dataset.csv'
    df = prepare_dataset(dataset_file,numberofinputs)

    # Run the test function
    (true_positives,
     false_positives,
     false_negatives,
     true_negatives) = test(df, thresholds, labels)

    # Calculate the total number of tokens
    tokens_labels = token_label_pairs([list(i) for i in zip(df['tokens'], df['labels'])])
    token_num = sum(len(tokens[0]) for tokens in tokens_labels)

    accuracy = prepare_result_dataframes(labels)
    precision = prepare_result_dataframes(labels)
    recall = prepare_result_dataframes(labels)

    # Start MLflow run
    run_name = f"GLiNER-{model_name}-{numberofinputs}"
    logger.start_run(run_name=run_name)

    try:
        # Log parameters
        logger.log_params({
            "model_name": model_name,
            "number_of_inputs": numberofinputs,
            "labels": labels,
            "thresholds": thresholds,
            "token_num": token_num
        })

        # Calculate and log metrics
        for i, row in true_positives.iterrows():
            for col in row.keys():
                if col == 'threshold':
                    threshold = row['threshold']
                    true_negatives.loc[i, 'threshold'] = threshold
                    accuracy.loc[i, 'threshold'] = threshold
                    recall.loc[i, 'threshold'] = threshold
                    precision.loc[i, 'threshold'] = threshold
                    continue

                if col == 'sum' and len(labels) == 1:
                    continue

                tp = true_positives.loc[i, col]
                fp = false_positives.loc[i, col]
                fn = false_negatives.loc[i, col]

                metrics = calculate_metrics(tp, fp, fn)

                true_negatives.loc[i, col] = metrics['true_negatives']
                accuracy.loc[i, col] = metrics['accuracy']
                recall.loc[i, col] = metrics['recall']
                precision.loc[i, col] = metrics['precision']

                step = int(float(row['threshold']) * 100)
                logger.log_metrics({
                    f"{col}/accuracy": accuracy.loc[i, col],
                    f"{col}/precision": precision.loc[i, col],
                    f"{col}/recall": recall.loc[i, col],
                    f"{col}/true_positives": true_positives.loc[i, col],
                    f"{col}/true_negatives": true_negatives.loc[i, col],
                    f"{col}/false_positives": false_positives.loc[i, col],
                    f"{col}/false_negatives": false_negatives.loc[i, col]
                }, step=step)

        # Save and log artifacts
        path = f'output/{numberofinputs}/'
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
            logger.log_artifact(abs_path)

        time.sleep(15)
    finally:
        logger.end_run()