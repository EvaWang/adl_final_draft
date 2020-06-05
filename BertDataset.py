import torch
from torch.utils.data import Dataset

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
        }

        if 'answerable' in self.data[index]:
            item['answerable'] = self.data[index]['answerable']
            item['answer_token'] = self.data[index]['answer_token']
            item['answer_text'] = self.data[index]['answer_text']
            item['answer_start'] = self.data[index]['answer_start']
            item['answer_end'] = self.data[index]['answer_end']

        return item

    def collate_fn(self, samples):
        batch = {}
        key_1 = ['id']
        key2tensor = ['input_ids',"token_type_ids", 'attention_mask']

        if 'answerable' in samples[0]:
            key_1.append("answer_text")
            key_1.append("answer_token")
            key2tensor.append("answerable")
            key2tensor.append("answer_start")
            key2tensor.append("answer_end")

        for key in key_1:
            batch[key] = [sample[key] for sample in samples]

        for key in key2tensor:
            batch[key] = [sample[key] for sample in samples]
            batch[key] = torch.tensor(batch[key])
        
        return batch