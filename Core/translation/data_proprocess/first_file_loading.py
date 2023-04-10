import json
import jieba
from torchtext.data import get_tokenizer
from collections import Counter

'''
    dataset_generator is a class, by input json file (in here, we specifically talk about translation2019zh dataset), 
    it will return a generator, which contains english and chinese sentences in tokenized form
'''

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
    


def module_test(json_file: str = "Core/translation/data/functional_test.json") -> tuple[list[list[str]]]:
    '''
    module_test is a function, which is used to test the module, return value is a tuple, which contains all english and chinese sentences (in order) in tokenized form
    '''
    english, chinese = dataset_generator(json_file).return_all_sentences()
    print(english[0], '\n', chinese[0])
    print("Stage 1 passed")
    return english, chinese




