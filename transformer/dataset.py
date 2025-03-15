from pandas import DataFrame
from torch import Tensor, int64
from torch.utils.data import Dataset

from transformer.tokenizer import get_tokenizer


class TranslationDataset(Dataset):
    def __init__(self,
                ds: DataFrame,
                source_lang: str,
                target_lang: str,
                source_tokenizer_path: str,
                target_tokenizer_path: str,
                max_length: int,
            ):
        self.data = ds
        self.max_length = max_length
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.source_tokenizer= get_tokenizer(source_tokenizer_path)
        self.target_tokenizer= get_tokenizer(target_tokenizer_path)

        self.sos_token = self.source_tokenizer.encode("<sos>").ids
        self.eos_token = self.source_tokenizer.encode("<eos>").ids
        self.pad_token = self.source_tokenizer.encode("<pad>").ids

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        source_tokens = self.source_tokenizer.encode(row[self.source_lang]).ids[:self.max_length - 2]
        target_tokens = self.target_tokenizer.encode(row[self.target_lang]).ids[:self.max_length - 2]

        source_tokens = self.sos_token + source_tokens + self.eos_token
        target_tokens = self.sos_token + target_tokens
        label_tokens = target_tokens + self.eos_token

        source_tokens += self.pad_token * (self.max_length - len(source_tokens))
        target_tokens += self.pad_token * (self.max_length - len(target_tokens))
        label_tokens += self.pad_token * (self.max_length - len(label_tokens))

        return (
            Tensor(source_tokens).to(int64),
            Tensor(target_tokens).to(int64),
            Tensor(label_tokens).to(int64),
        )