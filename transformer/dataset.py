import pandas as pd
import torch
from pandas import DataFrame
from torch import Tensor, int64
from torch.utils.data import Dataset

from transformer.tokenizer import load_tokenizer


class DynamicBatchTranslationDataset:
    def __init__(
            self,
            ds: pd.DataFrame,
            source_lang: str,
            target_lang: str,
            tokenizer_path: str,
            batch_size: int,
            max_length: int,
        ):
        
        self.data = ds
        self.batch_size = batch_size
        self.max_length = max_length
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.tokenizer= load_tokenizer(tokenizer_path)

        self.sos_token = self.tokenizer.encode("<sos>").ids
        self.eos_token = self.tokenizer.encode("<eos>").ids
        self.pad_token = self.tokenizer.encode("<pad>").ids

    def __len__(self):
        return (len(self.data) // self.batch_size) + 1

    def __iter__(self):
        for idx in range(0, len(self.data), self.batch_size):
            batch = self.data.iloc[idx: idx + self.batch_size]
            source_tokens = batch[self.source_lang].apply(lambda x: self.tokenizer.encode(x).ids)
            target_tokens = batch[self.target_lang].apply(lambda x: self.tokenizer.encode(x).ids)

            source_max_length = source_tokens.apply(len).max()
            target_max_length = target_tokens.apply(len).max()
            max_length = target_max_length if target_max_length > source_max_length else source_max_length
            max_length = self.max_length if max_length > self.max_length else max_length

            source_tokens = source_tokens.apply(lambda x: self.sos_token + x[:max_length - 2] + self.eos_token)
            target_tokens = target_tokens.apply(lambda x: self.sos_token + x[:max_length - 1])
            label_tokens = target_tokens.apply(lambda x: x[1: max_length] + self.eos_token)

            source_tokens = source_tokens.apply(lambda x: torch.Tensor(x + self.pad_token * (max_length - len(x))).to(torch.int64))
            target_tokens = target_tokens.apply(lambda x: torch.Tensor(x + self.pad_token * (max_length - len(x))).to(torch.int64))
            label_tokens = label_tokens.apply(lambda x: torch.Tensor(x + self.pad_token * (max_length - len(x))).to(torch.int64))

            yield (
                torch.stack(source_tokens.to_list()),
                torch.stack(target_tokens.to_list()),
                torch.stack(label_tokens.to_list())
            )


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
        self.source_tokenizer= load_tokenizer(source_tokenizer_path)
        self.target_tokenizer= load_tokenizer(target_tokenizer_path)

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
