
import os
import pickle
from BertDataset import BertDataset as dataset
from tqdm import tqdm


def read_pkl(set_name):
    data_path = os.path.join(os.getcwd(), f"./dataset/{set_name}.pkl")

    with open(data_path, 'rb') as f:
        bertDataset = pickle.load(f)

    return bertDataset


def count_tag_ratio(dataset):
    for i in range(1, 21):
        poz_val = 0
        count_line = 0
        print(i)
        for data in tqdm(dataset):
            poz_val = poz_val + data["tag_n"][i]
            count_line = count_line+1
        print(f"count_line:{count_line}")
        print(f"poz_val:{poz_val}")
        print(f"weight:{(count_line-poz_val)/poz_val}")
    pass

def check_val(dataset):
    limit = 10
    for data in tqdm(dataset):
        print(data["input_ids"])
        print(data["token_type_ids"])
        print(data["attention_mask"])
        print(data["pos_tag"])
        print(data["tag_n"])
        print(data["tag"])
        print(data["value"])
        print(f'data["start_idx"]:{data["start_idx"]}')
        print(f'data["end_idx"]:  {data["end_idx"]}')
        print()
        limit = limit -1
        if limit <0: break
    pass


if __name__ == '__main__':

    dataset = read_pkl("train_max100")
    check_val(dataset)
