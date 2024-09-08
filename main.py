# This is a sample Python script.
import logging

from transformers import DistilBertTokenizerFast, DistilBertConfig, TFDistilBertModel, AutoModel, AutoTokenizer, \
    TFBertModel, BertTokenizer
from preprocessing import load_data, split_train_test_data
from models import ModelBuilder
import glob
import csv

from pyt_finetune import train_and_evaluate_model
import yaml


with open("config.yaml", "r") as ymlfile:
    cfg = yaml.safe_load(ymlfile)
    model_name = cfg['Model']['model_name']
    framework = cfg['Model']['framework']
    data_dir = cfg['Data']['data_dir']
    datasets = cfg['Data']['datasets']

def scifive_model(input_data_file, seed=18):
    print("distil bert model for seed - ", seed)
    DISTILBERT_DROPOUT = 0.2
    DISTILBERT_ATT_DROPOUT = 0.2
    X, y = load_data(input_data_file)

    x_train, x_test, y_train, y_test = split_train_test_data(X, y, test_size=0.5, seed=seed)

    tokenizer = AutoTokenizer.from_pretrained('razent/SciFive-base-Pubmed_PMC')
    model = AutoModel.from_pretrained('razent/SciFive-base-Pubmed_PMC')

    # # Configure DistilBERT's initialization
    # config = DistilBertConfig(dropout=DISTILBERT_DROPOUT,
    #                           attention_dropout=DISTILBERT_ATT_DROPOUT,
    #                           output_hidden_states=True)

    # distilBERT = TFDistilBertModel.from_pretrained('distilbert-base-uncased', config=config)
    # Unfreeze distilBERT layers and make available for training
    for layer in model.layers:
        layer.trainable = True
    transformer_model = ModelBuilder(model=model, tokenizer=tokenizer)
    # Encode X_train
    x_train_ids, x_train_attention = transformer_model.batch_encode(x_train.tolist())

    # # Encode X_valid
    # x_valid_ids, x_valid_attention = batch_encode(tokenizer, X_valid.tolist())

    # Encode X_test
    x_test_ids, x_test_attention = transformer_model.batch_encode(x_test.tolist())

    transformer_model.train_model(x_train_ids, x_train_attention, y_train, x_test_ids, x_test_attention, y_test)

    wss_95 = transformer_model.evaluate_model(x_test_ids, x_test_attention, y_test)
    return wss_95


def distil_bert_model(input_data_file, seed=18):
    print("distil bert model for seed - ", seed)
    DISTILBERT_DROPOUT = 0.2
    DISTILBERT_ATT_DROPOUT = 0.2
    # input_data_file = '/home/max/Work/Coventry/project/CitationScreeningReplicability/data/processed/ADHD.tsv'


    X, y = load_data(input_data_file)

    x_train, x_test, y_train, y_test = split_train_test_data(X, y, test_size=0.5, seed=seed)

    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    # Configure DistilBERT's initialization
    config = DistilBertConfig(dropout=DISTILBERT_DROPOUT,
                              attention_dropout=DISTILBERT_ATT_DROPOUT,
                              output_hidden_states=True)

    distilBERT = TFDistilBertModel.from_pretrained('distilbert-base-uncased', config=config)
    # Unfreeze distilBERT layers and make available for training
    for layer in distilBERT.layers:
        layer.trainable = True
    transformer_model = ModelBuilder(model=distilBERT, tokenizer=tokenizer)
    # Encode X_train
    x_train_ids, x_train_attention = transformer_model.batch_encode(x_train.tolist())

    # # Encode X_valid
    # x_valid_ids, x_valid_attention = batch_encode(tokenizer, X_valid.tolist())

    # Encode X_test
    x_test_ids, x_test_attention = transformer_model.batch_encode(x_test.tolist())

    transformer_model.train_model(x_train_ids, x_train_attention, y_train, x_test_ids, x_test_attention, y_test)

    wss_95 = transformer_model.evaluate_model(x_test_ids, x_test_attention, y_test)
    return wss_95


def pytorch_models(model_name='distilbert-base-uncased', input_file_name='data/ADHD.tsv', seed=42):
    wss_95 = train_and_evaluate_model(model_name=model_name, input_file_name=input_file_name, seed=seed)
    return wss_95


def train_eval_model(model_fn=distil_bert_model, model_name='distilbert-base-uncased'):
    files_dir = data_dir
    files = glob.glob(files_dir+'data/*.tsv')
    if datasets:
        files_to_train = datasets
    else:
        files_to_train = None
    seeds = [60, 55, 98, 27, 36, 44, 72, 67, 3, 42]
    wss_95 = {}
    with open('results.csv', 'w') as f:
        writer = csv.writer(f)
        for file in files:
            filename = file.split('/')[-1]
            logging.info("\n\n\n")
            logging.info("-" * 90)
            if files_to_train:
                if filename not in files_to_train:
                    continue
            wss_95[filename] = []
            for seed in seeds:
                logging.info("-" * 70)
                logging.info(f'running for Dataset - > {filename}  with Seed - > {seed}')
                if model_fn==pytorch_models:
                    wss = model_fn(model_name, file, seed=seed)
                else:
                     wss = model_fn(file, seed=seed)
                wss_95[filename].append(wss)
                logging.info("-" * 70)
            logging.info("-" * 90)
            writer.writerow([filename] + wss_95[filename])
    print(wss_95)


if __name__ == '__main__':
    logging.info(f"Deep learning framework {framework}")
    logging.info(f"Deep learning model  {model_name}")
    if framework == 'pt':
        train_eval_model(model_fn=pytorch_models, model_name=model_name)
    else:
        train_eval_model(model_fn=distil_bert_model)
