import torch
import data_preprocess

class sentence_tensor:
    def __init__(self, vocab: list[str]):
        self.vocab = vocab

    def sentence_to_tensor(self, sentence: list[str]) -> torch.Tensor:
        return torch.tensor([self.vocab.index(x) for x in sentence])
    
    def tensor_to_sentence(self, tensor: torch.Tensor) -> list[str]:
        return [self.vocab[x] for x in tensor.tolist()]
    
def module_test():
    english: list[list[str]] = [["hello", "world"], ["good", "morning"], ["good", "night"]]
    sent = ["hello", "world"]
    english_vocab: list[str] = data_preprocess.create_vocab(english)

    st = sentence_tensor(english_vocab)
    ten = st.sentence_to_tensor(sent)
    print(ten)