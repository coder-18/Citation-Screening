from googletrans import Translator
from pathlib import Path
import glob
import csv
import os
import re
import pandas as pd
from preprocessing import load_data, split_train_test_data
import logging
import sys
from tqdm import tqdm

# Create super basic logger
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

translator = Translator()


def translate_text(texts, source_lang='en', dest_lang='de'):
    # translate from english
    translations_dest = []
    translations_src = []
    logging.info(f"translating to language {dest_lang}")
    for text in tqdm(texts):
        try:
            translated_text = translator.translate(text, dest=dest_lang, src=source_lang)
            translations_dest.append(translated_text.text)
        except:
            translations_dest.append('')

    # translate back to  english
    logging.info(f"translating back to language {source_lang}")
    for text in tqdm(translations_dest):
        if text == '':
            translations_src.append('')
            continue
        try:
            augmented_text = translator.translate(text, dest=source_lang, src=dest_lang)
            translations_src.append(augmented_text.text)
        except:
            translations_src.append('')

    return translations_src, translations_dest


def translate_dataset(source_lang='en', dest_lang='de'):
    files = glob.glob('data_dir/data/*.tsv')
    augmented_dir = 'augmented_' + source_lang + dest_lang
    dir_path = Path(augmented_dir)
    if not dir_path.exists():
        os.mkdir(augmented_dir)
    for file in files:
        filename = file.split('/')[-1]
        logging.info(f'Augmenting on dataset - {filename}')
        df = pd.read_csv(file, delimiter="\t")
        df["Title"] = df["Title"].fillna("")
        df["Abstract"] = df["Abstract"].fillna("")
        df["Abstract"] = df["Title"] +' '+  df["Abstract"]
        df = df[df["Label"]==1]
        X = list(df["Abstract"])
        # X = [preprocess_text(elem) for elem in X]
        X = [re.sub(r"[\W]+", " ", elem) for elem in X]
        X = [re.sub(r"[\n\r\t ]+", " ", elem) for elem in X]
        texts = [elem.lower() for elem in X]
        augmented_filename = 'augmented_' + filename
        logging.info(f'Augmented filename - {augmented_filename}')
        complete_filename = augmented_dir + '/' + augmented_filename
        file_path = Path(complete_filename)
        if file_path.exists():
            continue
        with open(complete_filename, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['Original_text', 'translated_text', 'augmented_text'])
            translated_texts, augmented_texts = translate_text(texts, source_lang, dest_lang)
            for org, trans, aug in zip(texts, translated_texts, augmented_texts):
                writer.writerow([org, trans, aug])


translate_dataset()
# print(translator.translate(' this is to be translated', dest='fr', src='en'))
