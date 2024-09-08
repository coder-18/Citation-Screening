import yaml
import pandas as pd


def get_augmentation_text(train_text, aug_csv):
    df = pd.read_csv(aug_csv)
    org_keys = df['Original_text']
    translated_values = df['translated_text']
    translations = []
    augdict = dict(zip(org_keys, translated_values))
    for text in train_text:
        if text in augdict.keys():
            value = augdict[text]
            if value != '':
                translations.append(value)

    return translations
