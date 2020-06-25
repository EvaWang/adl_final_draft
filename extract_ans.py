import csv
from tqdm import tqdm
import argparse
import logging
import os
from pathlib import Path
import unicodedata
import re
import pandas as pd

def main(args):
        
    print(f"data path:{args.data_path}")

    # loading datasets from excel files
    file_list = os.listdir(args.data_path)
    df_train = []
    df_list = []
    for filepath in tqdm(file_list):
        filename = os.path.basename(filepath).split('.')[0]
        df = pd.read_excel(os.path.join(args.data_path, filepath), header=0)
        df['ID'] = df['Index'].map(lambda x: filename+'-'+str(x))
        df_list.append(df)
        
    df_train = pd.concat(df_list, axis=0, ignore_index=True)  
    del df_list
    df_train_n = normalize_data(df_train)

    converted_data = []
    for i, row in tqdm(df_train_n.iterrows(), total=df_train_n.shape[0]):
        # tag_str = row["Tag"] if isinstance(row["Tag"], str) else ""
        # tags = tag_str.split(';') if tag_str != "" else []
        # value_str = row["Value"] if isinstance(row["Value"], str) else ""
        # values = value_str.split(';') if value_str != "" else []
        # print(row["Tag"])
        # print(row["Value"])

        tags = row["Tag"]
        values = row["Value"]

        if len(tags) != len(values):
            if len(tags) != 1 and len(values) != 1:
                print(row["Tag"])
                print(row["Value"])
                raise ValueError("# of tags and # of values should match, or one of them should be 1. Got len(tags) = {} and len(values) = {} at row {}".format(len(tags), len(values), i))
            if len(tags) == 1:
                tags = [tags[0]] * len(values)
            else:
                values = [values[0]] * len(tags)
        
        # tags = [normalize_tag(tag) for tag in tags]

        prediction = " ".join(["{tag}:{value}".format(tag=tag.replace(" ", ""), value=value.replace(" ", "")) for tag, value in zip(tags, values)])
        only_tag = " ".join(["{tag}".format(tag=tag.replace(" ", "")) for tag in tags])
        if prediction == "":
            prediction = "NONE"
            only_tag = "NONE"
        converted_data.append({"ID": row["ID"], "Prediction": prediction, "Only_Tag": only_tag})

    with open(args.output_file, "w") as csvfile:
        writer = csv.DictWriter(csvfile, ["ID", "Prediction", "Only_Tag"])
        writer.writeheader()
        writer.writerows(converted_data)

# clean and normalization
def normalize_data(df_train):
    df_train['Tag'] = df_train['Tag'].map(lambda x: unicodedata.normalize("NFKC", re.sub('＊|\*|\s+', '', x)).split(';') if pd.notnull(x) else [])
    df_train['Value'] = df_train['Value'].map(lambda x: unicodedata.normalize("NFKC", re.sub('＊|\*|\s+', '', x)).split(';') if pd.notnull(x) else [])

    return df_train



def _parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('data_path', type=Path, help='')
    parser.add_argument('output_file', type=Path, help='')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = _parse_args()
    loglevel = os.environ.get('LOGLEVEL', 'INFO').upper()
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s',
                        level=loglevel, datefmt='%Y-%m-%d %H:%M:%S')
    main(args)