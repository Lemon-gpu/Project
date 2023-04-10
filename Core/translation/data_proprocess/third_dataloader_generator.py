from torch.utils.data import Dataset, DataLoader
import torch
import second_vocabulary_generator
from torch.nn.utils.rnn import pad_sequence

class translation_dataset(Dataset):

    '''
    when use this dataset, we assume top four tokens are <pad>, <unk>, <bos>, <eos>, which means padding, unknown, begin of sentence, end of sentence
    '''

    default_vocab: dict = {'<pad>': 0, '<unk>': 1, '<bos>': 2, '<eos>': 3}

    english_tensor: list[torch.Tensor] = None
    chinese_tensor: list[torch.Tensor] = None

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __init__(self, english_tensor: list[torch.Tensor], chinese_tensor: list[torch.Tensor]):
        assert len(english_tensor) == len(chinese_tensor)
        super().__init__()
        self.english_tensor = english_tensor
        self.chinese_tensor = chinese_tensor

    def __len__(self):
        return len(self.english_tensor)
    
    def __getitem__(self, index: int):
        '''
        __getitem__ would be called when we use dataloader to get data

        Args:
            index (int): as it is

        Returns:
            tuple[torch.Tensor]: english and chinese sentence
        '''
        # torch.cat is a function to concat two tensors, which has no difference with torch.concat
        english: torch.Tensor = torch.cat(
            (torch.tensor(self.default_vocab['<bos>'], device=self.device).reshape(-1, 1), 
             self.english_tensor[index].reshape(-1, 1), 
             torch.tensor(self.default_vocab['<eos>'], device=self.device).reshape(-1, 1))
            , dim=0)
        chinese: torch.Tensor = torch.cat(
            (torch.tensor(self.default_vocab['<bos>'], device=self.device).reshape(-1, 1),
                self.chinese_tensor[index].reshape(-1, 1),
                torch.tensor(self.default_vocab['<eos>'], device=self.device).reshape(-1, 1))
            , dim=0)
        return english, chinese
    
class dataloader_generator:

    def gen_dataloader(self, english_tensor: torch.tensor, chinese_tensor: torch.tensor, batch_size: int = 256, shuffle: bool = True) -> DataLoader:
        '''
        get_dataloader is a function to get dataloader

        Args:
            english_tensor (torch.tensor): input english sentence
            chinese_tensor (torch.tensor): input chinese sentence
            batch_size (int, optional): size of the single batch. Defaults to 256.
            shuffle (bool, optional): shuffle or not. Defaults to True.

        Returns:
            DataLoader: dataloader
        '''
        
        def collate_fun(input: list) -> tuple[torch.Tensor]:
            english, chinese = zip(*input)
            english = pad_sequence(english, batch_first=True, padding_value=0) # padding_value = 0 means padding with <pad>
            chinese = pad_sequence(chinese, batch_first=True, padding_value=0)
            return english, chinese
        
        dataset: translation_dataset = translation_dataset(english_tensor, chinese_tensor)
        dataloader: DataLoader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fun)
        return dataloader

    
def module_test():
    english, chinese = second_vocabulary_generator.module_test()
    dataloader = dataloader_generator().gen_dataloader(english, chinese)
    for english, chinese in dataloader:
        print(english, chinese)
        break
    print("Stage 3 passed")
