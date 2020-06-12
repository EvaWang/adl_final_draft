import torch
from torch.utils.data import Dataset

def pad_to_len(seqs, to_len, padding=0):
    paddeds = []
    for seq in seqs:
        paddeds.append(
            seq[:to_len] + [padding] * max(0, to_len - len(seq))
        )

    return paddeds

class BertDataset(Dataset):
    def __init__(self, data, max_text_len=512):
        # 開頭是[CLS], QA分隔是[SEP], [PAD]補滿
        self.data = data
        self.max_text_len = max_text_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        item = {
            'id': self.data[index]['id'],
            'input_ids': self.data[index]["input_ids"],
            'token_type_ids': self.data[index]["token_type_ids"],
            'attention_mask': self.data[index]["attention_mask"],
            'pos_tag': self.data[index]["pos_tag"],
        }

        if 'tag_n' in self.data[index]:
            item['tag_n'] = self.data[index]['tag_n']
            item['value'] = self.data[index]['value']

        return item

    def collate_fn(self, samples):
        batch = {}
        key_1 = ['id']
        key2tensor = ['pos_tag', 'input_ids',"token_type_ids", 'attention_mask']

        if 'tag_n' in samples[0]:
            batch["tag_n"] = torch.tensor([sample["tag_n"] for sample in samples])

        for key in key_1:
            batch[key] = [sample[key] for sample in samples]

        for key in key2tensor:
            batch[key] = [sample[key] for sample in samples]
            batch[key] = pad_to_len(batch[key], 512) 
            batch[key] = torch.tensor(batch[key])
        
        return batch