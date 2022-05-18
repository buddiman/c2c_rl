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

