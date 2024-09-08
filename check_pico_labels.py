import pandas as pd
import glob
import numpy as np


files = glob.glob('/data_dir/data/*.tsv')

for file in files:
    try:
        print(file)
        df = pd.read_csv(file, delimiter="\t")
        df["Title"] = df["Title"].fillna("")
        df["Abstract"] = df["Abstract"].fillna("")
        df["Abstract"] = df["Title"] + ' ' + df["Abstract"]
        texts = list(df["Abstract"])
        y = list(df["Label"])

        filename = file.split('/')[-1]
        pico_dir = '/data_dir/pico_data_dir'
        pico_filename = 'pico_' + filename
        complete_filename = pico_dir + '/' + pico_filename
        df_pico = pd.read_csv(complete_filename)
        df_pico= df_pico.fillna("no_picos")
        pico_sents = list(df_pico['pico_text'])
        included_citations = 0
        excluded_citations = 0
        included_citations_with_no_picos = 0
        excluded_citations_with_no_picos = 0

        for label, picos in zip(y, pico_sents):
            if label == 1:
                included_citations += 1
                if picos == 'no_picos':
                    included_citations_with_no_picos += 1

            if label == 0:
                excluded_citations += 1
                if picos == 'no_picos':
                    excluded_citations_with_no_picos += 1


        included_citations_with_no_picos_ratio = included_citations_with_no_picos / included_citations
        excluded_citations_with_no_picos_ratio = excluded_citations_with_no_picos / excluded_citations
        print(f"for {filename} citations included with no picos = {included_citations_with_no_picos}")
        print(f"for {filename} citations excluded with no picos = {excluded_citations_with_no_picos}")
        print(f"for {filename} citations included with no picos ratio = {included_citations_with_no_picos_ratio}")
        print(f"for {filename} citations excluded with no picos ratio = {excluded_citations_with_no_picos_ratio}")
    except Exception as e:
        print(e)
