from collections import Counter
import torch
import first_file_loading

'''
    over here is a utility class, which can convert sentence to tensor and tensor to sentence, by though it will auto generate vocabulary
    if device is not specified, it will use cuda if cuda is available
'''

class sentence_tensor:

    vocab: list[str] = None
    sentences: list[list[str]] = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __init__(self, sentences: list[list[str]]):
        self.sentences = sentences
        self.vocab = self.create_vocab(sentences)

    def sentence_to_tensor(self, sentence: list[str]) -> torch.Tensor:
        '''
        sentence_to_tensor convert sentence to tensor, if the word is not in vocabulary, it will be replaced by <unk>

        Args:
            sentence (list[str]): input tokenized sentence

        Returns:
            torch.Tensor: tensor of sentence
        '''
        return torch.tensor([self.vocab.index(x) if x in self.vocab else self.vocab.index('<unk>') for x in sentence], dtype=torch.long, device=self.device)
    
    def tensor_to_sentence(self, tensor: torch.Tensor) -> list[str]:
        '''
        tensor_to_sentence convert tensor to sentence

        Args:
            tensor (torch.Tensor): input tensor

        Returns:
            list[str]: sentence
        '''
        return [self.vocab[x] if x < len(self.vocab) else '<unk>' for x in tensor]

    def sentences_to_tensor(self, sentences: list[list[str]]) -> list[torch.Tensor]:
        '''
        sentences_to_tensor convert sentences to tensor

        Args:
            sentences (list[list[str]]): input tokenized sentences. When sentences is None, it will use self.sentences

        Returns:
            torch.Tensor: tensor of sentences
        '''
        if sentences is None:
            sentences = self.sentences
        return [self.sentence_to_tensor(x) for x in sentences]
    
    def tensor_to_sentences(self, tensor: torch.Tensor) -> list[list[str]]:
        '''
        tensor_to_sentences convert tensor to sentences

        Args:
            tensor (torch.Tensor): input tensor

        Returns:
            list[list[str]]: sentences
        '''
        return [self.tensor_to_sentence(x) for x in tensor]

    def create_vocab(self, sentences: list[list[str]], max_elements: int = None) -> list[str]:
        '''
        create_vocab use sentences to create vocabulary include default vocabulary without frequency

        Args:
            sentences (list[list[str]]): input tokenized sentences
            max_elements (int, optional): token appear in top max_element times. Defaults to None.

        Returns:
            list[str]: vocabulary
        '''
        default_vocab = ['<pad>', '<unk>', '<bos>', '<eos>'] # pad: padding, unk: unknown, bos: begin of sentence, eos: end of sentence

        counter = Counter()
        for sentence in sentences:
            counter.update(sentence)
        if max_elements is not None:
            max_elements -= 4 # the reason for this is that we have 4 default vocab
        counter = counter.most_common(max_elements)

        words, _ = zip(*counter)
        return default_vocab + list(words)
    
    
def module_test() -> tuple[torch.Tensor]:
    english, chinese = first_file_loading.module_test()
    english = sentence_tensor(english).sentences_to_tensor(None)
    chinese = sentence_tensor(chinese).sentences_to_tensor(None)

    print(len(english), len(chinese))
    print('\n')
    print(english[0], chinese[0])
    print("Stage 2 passed")

    return english, chinese


