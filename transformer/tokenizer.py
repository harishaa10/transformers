import pandas as pd

from tokenizers import normalizers, pre_tokenizers, Tokenizer
from tokenizers.pre_tokenizers import Whitespace, Digits, Punctuation
from tokenizers.normalizers import NFD, Lowercase
from tokenizers.trainers import WordLevelTrainer, BpeTrainer
from tokenizers.models import WordLevel, BPE

def get_tokenizer(
        ds: pd.DataFrame,
        lang: list,
        tokenizer_path: str,
        vocab_size: int = 32000
    ):

    for l in lang:
        assert l in ds.columns, f"{l} should be a column in the dataset"
    texts = pd.concat([ds[l] for l in lang]).astype(str).unique().tolist()
    
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.normalizer = normalizers.Sequence([
        NFD(),
        Lowercase()
    ])
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<sos>", "<eos>", "<unk>"]
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)
    tokenizer.save(tokenizer_path)
    print(f"Shared tokenizer saved at: {tokenizer_path}")
    
    return tokenizer


def get_custom_tokenizer(ds: pd.DataFrame, lang: str, tokenizer_path: str, vocab_size: int = 4096):
    
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
        min_frequency= 5,
        show_progress= True,
        special_tokens= ["<pad>", "<sos>", "<eos>", "<unk>"]
    )
    tokenizer.train_from_iterator(ds[lang].unique(), trainer= trainer)
    tokenizer.save(tokenizer_path)
    print(f"Tokenizer saved at: {tokenizer_path}")

    return tokenizer

def load_tokenizer(tokenizer_path: str):
    return Tokenizer.from_file(tokenizer_path)
