import json
import jieba
from torchtext.data import get_tokenizer
from collections import Counter

class dataset_generator():

    file_path: str = None
    batch_size: int = None

    def __init__(self, file_path: str, batch_size: int = -1): # when batch_size = -1, it means that the batch size is the size of the dataset
        self.file_path = file_path
        self.batch_size = batch_size
    
    def __iter__(self):
        return self

    def __next__(self):
        '''
        __next__ when batch_size = -1, it means that the batch size is the size of the dataset
            return iterator, which contains english and chinese sentences in tokenized form

        Yields:
            generator: contain english and chinese sentences
        '''
        chinese_sentences: list = []
        english_sentences: list = []
        for english, chinese in self.load_json(self.file_path):
            english, chinese = self.tokenize(english, chinese)
            chinese_sentences.append(chinese)
            english_sentences.append(english)
            if self.batch_size != -1 and len(chinese_sentences) == self.batch_size:
                yield english_sentences, chinese_sentences
                chinese_sentences.clear()
                english_sentences.clear()
        if len(chinese_sentences) != 0:
            yield english_sentences, chinese_sentences

    def return_all_sentences(self) -> tuple[list[list[str]], list[list[str]]]:
        '''
        return_all_sentences ignore batch_size, return all sentences in tokenized form

        Returns:
            tuple[list[list[str]], list[list[str]]]: all english and chinese sentences contain in json file
        '''
        chinese_sentences: list = []
        english_sentences: list = []
        for english, chinese in self.load_json(self.file_path):
            english, chinese = self.tokenize(english, chinese)
            chinese_sentences.append(chinese)
            english_sentences.append(english)
        return english_sentences, chinese_sentences

    def load_json(self, file_path: str) -> tuple[str, str]:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = json.loads(line)
                english, chinese = line['english'], line['chinese']
                if not english.isascii():
                    english, chinese = chinese, english
                yield english, chinese
        
    def tokenize(self, english: str, chinese: str) -> tuple[list[str]]:
        tokenizer = get_tokenizer('basic_english')
        english: list[str] = tokenizer(english)
        chinese: list[str] = list(jieba.cut(chinese))
        chinese = [x for x in chinese if x not in {' ', '\t'}]
        return english, chinese
    
def create_vocab(sentences: list[list[str]], max_elements: int = None) -> list[str]:
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

def module_test():
    file_path: str = "Core/translation/data/functional_test.json"
    dg = dataset_generator(file_path)
    english, chinese = dg.return_all_sentences()
    english_vocab = create_vocab(english)
    chinese_vocab = create_vocab(chinese)
    print(english_vocab)
    print(chinese_vocab)

module_test()




