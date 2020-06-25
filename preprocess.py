import argparse
import logging
import os
import json
import pickle
from pathlib import Path
from transformers import BertTokenizer
from BertDataset import BertDataset
from tqdm import tqdm
import pandas as pd
import numpy as np
import unicodedata
import re
from rakutenma import RakutenMA

import sys

rma = RakutenMA() # (default: phi = 2048, c = 0.003906)
tokenizer = BertTokenizer.from_pretrained('cl-tohoku/bert-base-japanese', do_lower_case=False)

def main(args):

    with open(args.config_path / 'config.json') as f:
        config = json.load(f)
    
    rma.load(args.config_path / "model_ja.min.json")
    rma.hash_func = rma.create_hash_func(15)
        
    print(f"config:{config}")

    # loading datasets from excel files
    file_list = os.listdir(config["data_folder"])
    df_train = []
    df_list = []
    for filepath in tqdm(file_list):
        filename = os.path.basename(filepath).split('.')[0]
        df = pd.read_excel(os.path.join(config["data_folder"], filepath), header=0)
        df['ID'] = df['Index'].map(lambda x: filename+'-'+str(x))
        df_list.append(df)
        
    df_train = pd.concat(df_list, axis=0, ignore_index=True)  
    del df_list

    with open(config["parent_text"]) as f:
        parent_text_list = json.load(f)

    logging.info('Creating dataset pickle...')
    create_bert_dataset(
        process_samples(df_train, config, config["is_testset"]),
        # process_samples_with_parent_text(df_train, parent_text_list, config, config["is_testset"]),
        config["output_filename"],
        config["max_text_len"]
    )

# clean and normalization
def normalize_data(df_train, config):
    df_train['Tag'] = df_train['Tag'].map(lambda x: unicodedata.normalize("NFKC", re.sub('＊|\*|\s+', '', x)).split(';') if pd.notnull(x) else [])
    df_train['Tag_n'] = df_train['Tag'].map(lambda x: [1 if (type(x) != float and i in x) else 0 for i in config["tag_map"]])

    df_train['Text'] = df_train['Text'].map(lambda x: unicodedata.normalize("NFKC", re.sub('＊|\*|\s+', '', x)))
    df_train['Value'] = df_train['Value'].map(lambda x: unicodedata.normalize("NFKC", re.sub('＊|\*|\s+', '', x)) if pd.notnull(x) else '')

    return df_train

def map_pos(tokenized_text, config_pos_map):
    text_pos = rma.tokenize("".join(tokenized_text))
    pos_indices = []
    current_len = 0
    for pos_token in text_pos:
        current_len = current_len+len(pos_token[0])
        pos_indices.append(current_len)

    # 尋找對應詞性
    tokenized_text_pos = [0] # 第一個token是CLS 擺0
    count_len = 0 # 累積長度
    pos_idx = 0 # 目前pos index
    for token_char in tokenized_text:
        if count_len > pos_indices[pos_idx]:
            pos_idx = pos_idx+1
            
        try:
            pos_tag = config_pos_map[text_pos[pos_idx][1]]
        except:
            pos_tag = 0

        tokenized_text_pos.append(pos_tag)
        if token_char == tokenizer.unk_token:# 遇到[UNK]先加一 看有沒有bug (有)
            count_len = count_len+1
        elif '##' in token_char: # 遇到wordpieces_prefix: [##___]
            count_len = count_len+ len(token_char) -2
        else: 
            count_len = count_len+ len(token_char)

    return tokenized_text_pos
    # 詞性對應完成

# 1-D
def map_value(tokenized_text, vals):
    # ['(', '4', ')', '納', '入', '期', '間', '平', '成', '30', '年', '3', '月', '1', '日', 'から', '平', '成', '31', '年', '2', '月', '28', '日', 'ま', '##て']
    # ['平成30年', '平成30年3月1日', '平成31年2月28日']
    content_str = "".join(tokenized_text).replace("##","")
    content_tag = np.zeros(len(tokenized_text))
    for val in vals:
        tokenized_val = "".join(tokenizer.tokenize(val)).replace("##","")
        start = content_str.find(tokenized_val)
        if start<0: 
            print(content_str)
            print(tokenized_val)
            continue
        end = start+ len(tokenized_val)
        content_tag[start:end] = 1

    return content_tag.tolist()

# 2-D src_len*20
def map_valuebyTag(tokenized_text, tags, vals):
    start_idx = np.full(20, -1)
    end_idx = np.full(20, -1)
    for v_i, val in enumerate(vals):
        start = -1
        end = -1
        if not val or len(tags)<=v_i:  continue

        tokenized_val = "".join(tokenizer.tokenize(val)).replace('##','')
        for t_i, token in enumerate(tokenized_text):
            if token.replace('##','') in tokenized_val:
                for end_i in range(t_i, len(tokenized_text),1):
                    extract = "".join(tokenized_text[t_i:end_i+1]).replace('##','')
                    if tokenized_val == extract:
                        start = t_i
                        end = end_i
                        break
            if end>=0: break

        try:
            start_idx[tags[v_i]] = start
            end_idx[tags[v_i]] = end
            # extracted_ans = "".join(tokenized_text[start:end+1]).replace('##','')
            # print(extracted_ans)
            # assert  extracted_ans == tokenized_val
        except:
            print("content_tag[tags[v_i]] = [start, end]")
            print(f"vals:{vals}")
            print(f"v_i:{v_i}")
            print(f"tags:{tags}")
            print(f"check:{len(tags)<=v_i}")
            
        # try:
        #     assert start<100
        #     assert start>=0
        #     assert end<100
        #     assert end>=0
        # except:
        #     print("start/end out of range")
        #     print(f"val:{val}")
        #     print(f"tokenized_val:{tokenized_val}")
        #     print(len(tokenized_text))
        #     print([start, end])
        #     print(tokenized_text)
        #     print(tags)
        #     print(vals)
        
    return start_idx, end_idx

def process_samples(samples, config, is_test=False):
    samples = normalize_data(samples, config)

    stack = []
    for i, sample in tqdm(samples.iterrows(), total=samples.shape[0]):
        tag_n = []
        value_list = sample["Value"]
        tokenized_text = tokenizer.tokenize(sample["Text"])
        if is_test == False and type(sample["Value"]) != float:
            tag_idx = [(config["tag_map"][t]-1) for t in sample['Tag']]
            value_list = sample["Value"].split(";")
            if len(value_list)!= len(tag_idx):
                value_list = value_list* len(tag_idx)
            start_idx, end_idx = map_valuebyTag(tokenized_text, tag_idx,  value_list)
            tag_n = sample['Tag_n']
            if sum(tag_n) == 0:
                tag_n[0] = 1
            else:
                tag_n[0] = 0

        tokenized_text_pos = map_pos(tokenized_text, config["pos_map"])
        tokenized_text_encode = tokenizer.encode_plus(sample["Text"], pad_to_max_length=True, return_attention_mask=True, return_token_type_ids=True, max_length=config["max_text_len"])
        item = {
            'id': sample["ID"],
            'content': sample["Text"],
            'input_ids': tokenized_text_encode["input_ids"],
            'token_type_ids': tokenized_text_encode["token_type_ids"],
            'attention_mask': tokenized_text_encode["attention_mask"],
            'pos_tag': tokenized_text_pos,
            'tag_n': tag_n,
            'value': value_list,
            'tag': sample['Tag'],
            'start_idx': start_idx.tolist(), 
            'end_idx':end_idx.tolist()
        }

        stack.append(item)
            # # debug
            # stop_count = stop_count -1
            # if stop_count<0:
            #     sys.exit()
            # else:
            #     pass
            #     print(f'content:     {content}')
            #     print(f'segment_idx: {item["segment_idx"]}')
            #     print(f'input_ids:   {item["input_ids"]}')
            #     print(f'pos_tag:     {item["pos_tag"]}')
            #     print(f'tag_n:       {item["tag_n"]}')
            #     print(f'value:       {item["value"]}')

    return stack

def process_samples_with_parent_text(samples, parent_text_list, config, is_test=False):
    samples = normalize_data(samples, config)
    target_max_len = config["target_max_len"]
    max_text_len = config["max_text_len"]

    stack = []
    for i, sample in tqdm(samples.iterrows(), total=samples.shape[0]):
        line_idx = sample["ID"]
        tag_n = sample['Tag_n']

        content_index = parent_text_list["id2content"][line_idx]
        parent_text = parent_text_list["content_list"][f"content_{content_index['index']}"]

        tokenized_text = tokenizer.tokenize(sample["Text"])
        if is_test == False:
            if type(sample["Value"]) != float: 
                value_list = sample["Value"].split(";")
                tag_idx = [(config["tag_map"][t]-1) for t in sample['Tag']]
                start_idx, end_idx = map_valuebyTag(tokenized_text[:target_max_len], tag_idx,  value_list)

        context_token_ids = tokenizer.encode_plus(parent_text["content_text"], pad_to_max_length=True, return_attention_mask=True, return_token_type_ids=True, max_length=(max_text_len-target_max_len+1))
        target_token_ids = tokenizer.encode_plus(sample["Text"], pad_to_max_length=True, return_attention_mask=True, return_token_type_ids=True, max_length=target_max_len)

        combine_inputs = context_token_ids['input_ids']+target_token_ids['input_ids'][1:]
        combine_type_ids = context_token_ids['token_type_ids']+target_token_ids['attention_mask'][1:]
        combine_attn_masks = context_token_ids['attention_mask']+target_token_ids['attention_mask'][1:]

        tokenized_text_pos = map_pos(tokenized_text, config["pos_map"])

        item = {
            'id': line_idx,
            'input_ids': combine_inputs,
            'token_type_ids': combine_type_ids,
            'attention_mask': combine_attn_masks,
            'pos_tag': tokenized_text_pos, # 只作目標的
            'tag_n': tag_n,
            'tag': sample['Tag'],
            'start_idx': start_idx, 
            'end_idx':end_idx
        }

        stack.append(item)
   
    return stack


def create_bert_dataset(samples, save_path, max_text_len):
    dataset = BertDataset(samples, max_text_len)
    with open(save_path, 'wb') as f:
        pickle.dump(dataset, f)


def _parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('config_path', type=Path, help='')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = _parse_args()
    loglevel = os.environ.get('LOGLEVEL', 'INFO').upper()
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s',
                        level=loglevel, datefmt='%Y-%m-%d %H:%M:%S')
    main(args)