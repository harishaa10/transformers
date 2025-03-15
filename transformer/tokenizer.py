import pandas as pd

from tokenizers import normalizers, pre_tokenizers, Tokenizer
from tokenizers.pre_tokenizers import Whitespace, Digits, Punctuation
from tokenizers.normalizers import NFD, Lowercase
from tokenizers.trainers import WordLevelTrainer
from tokenizers.models import WordLevel

def get_tokenizer(tokenizer_path: str= None, ds: pd.DataFrame= None, lang: str = None, vocab_size: int = 4096):
    if tokenizer_path:
        tokenizer= Tokenizer.from_file(tokenizer_path)
    else:
        assert ds is not None, "If tokenizer path is not given, a dataset must be provided"
        assert lang in ds.columns, f"{lang} should be a column in the dataset"
        tokenizer= Tokenizer(WordLevel(unk_token="<unk>"))
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            Whitespace(),
            Digits(individual_digits=True),
            Punctuation()
        ])
        tokenizer.normalizer= normalizers.Sequence([NFD(), Lowercase()])
        trainer = WordLevelTrainer(
            vocab_size= vocab_size,
            min_frequency= 2,
            show_progress= True,
            special_tokens= ["<pad>", "<sos>", "<eos>", "<unk>"]
        )
        tokenizer.train_from_iterator(ds[lang].unique(), trainer= trainer)
        file_name = f"./tokenizer/{lang}_vocab_{vocab_size}.json"
        tokenizer.save(file_name)
        print(f"Tokenizer saved at: {file_name}")

    return tokenizer
