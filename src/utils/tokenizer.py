from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace


class SimpleTokenizer:
    def __init__(self):
        self.tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

    def train_and_save(self, datasets, name):
        trainer = BpeTrainer(special_tokens=["[MASK]", "[UNK]", "[CLS]", "[SEP]", "[PAD]"])
        self.tokenizer.pre_tokenizer = Whitespace()

        self.tokenizer.train(datasets, trainer)
        self.tokenizer.save("data/tokenizers/" + name)

    def load(self, name):
        self.tokenizer = Tokenizer.from_file("data/tokenizers/" + name)

    def encode(self, text):
        return self.tokenizer.encode(text)

    def decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def get_vocabulary(self):
        return self.tokenizer.get_vocab()

    def get_vocabulary_size(self):
        return self.tokenizer.get_vocab_size()

