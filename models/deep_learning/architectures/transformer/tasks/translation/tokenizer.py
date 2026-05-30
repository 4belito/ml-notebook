from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer

from .dataset import TranslationHFDataset


def get_all_sentences(ds: TranslationHFDataset, lang: str):
    for item in ds:
        yield item["translation"][lang]  # type: ignore


# Most code taken from: https://huggingface.co/docs/tokenizers/quicktour
def build_tokenizer(ds: TranslationHFDataset, lang: str):
    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = WordLevelTrainer(
        special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"],
    )
    tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
    return tokenizer


def get_or_build_tokenizer(tokenizer_path: Path, ds: TranslationHFDataset, lang: str):
    if not Path.exists(tokenizer_path):
        tokenizer = build_tokenizer(ds, lang)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer
