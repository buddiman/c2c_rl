from tokenizers import Tokenizer, pre_tokenizers, decoders, processors, models, trainers
from tokenizers.implementations import CharBPETokenizer


def create_new_tokenizer(datasets, vocab_size, name):
    pass


def load_tokenizer_from_file(file):
    return Tokenizer.from_file(file)


def load_tokenizer_from_pretrained_bpe():
    return CharBPETokenizer()


def train_tokenizer(tokenizer: CharBPETokenizer, datasets, name):
    tokenizer.train(datasets)


def save_tokenizer(tokenizer: CharBPETokenizer, name):
    tokenizer.save(name)

    
def tokenize(tokenizer: CharBPETokenizer, text):
    tokenizer.encode(text)

