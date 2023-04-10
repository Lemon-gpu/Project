import torch
import torch.nn as nn

'''
    In transformers, d_model is usually the same size as the embedding size (e.g., 512) 
'''

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_seq_len: int):

        '''
        这段代码是 PositionalEncoding 类的 __init__ 方法中的一部分，它用于计算位置编码矩阵 pe。
        首先，它创建了两个张量 i_seq 和 j_seq。i_seq 是一个长度为 max_seq_len 的张量，其元素从 1 到 max_seq_len 等间隔排列。
        j_seq 是一个长度为 d_model // 2 的张量，其元素从 2 到 d_model 以 2 为间隔排列。
        然后，使用 torch.meshgrid 函数创建两个矩阵 pos 和 two_i。这两个矩阵的形状都是 (max_seq_len, d_model // 2)。
        其中，pos 矩阵的每一行都是 i_seq 的副本，而 two_i 矩阵的每一列都是 j_seq 的副本。
        接下来，使用正弦和余弦函数计算两个矩阵 pe_2i 和 pe_2i_1。这两个矩阵的元素是通过对应位置的元素进行计算得到的。结合位置编码公式，这可以并行处理。
        最后，使用 torch.stack 函数将这两个矩阵沿第三维堆叠起来，并使用 reshape 函数将其重塑为形状为 (1, max_seq_len, d_model) 的张量。这就是位置编码矩阵 pe。
        '''

        super().__init__()
        # Assume d_model is an even number for convenience
        assert d_model % 2 == 0

        i_seq = torch.linspace(1, max_seq_len, max_seq_len) # 1, 2, 3, ..., max_seq_len
        j_seq = torch.linspace(2, d_model, d_model // 2) # 2, 4, 6, ..., d_model. d_model // 2 means integer division, e.g., 5 // 2 = 2. 
        # since we assume d_model is an even number, d_model // 2 is the same as d_model / 2
        pos, two_i = torch.meshgrid(i_seq, j_seq, indexing="ij") # both matrix in shape (max_seq_len, d_model // 2)
        pe_2i = torch.sin(pos / 10000**(two_i / d_model))
        pe_2i_1 = torch.cos(pos / 10000**(two_i / d_model))
        pe = torch.stack((pe_2i, pe_2i_1), 2).reshape(1, max_seq_len, d_model)

        self.register_buffer('pe', pe, False) # False means pe is not a parameter of the model, so it will not be updated during training. 
        # when move model to GPU, pe will be moved to GPU

    def forward(self, x: torch.Tensor):
        n, seq_len, d_model = x.shape
        pe: torch.Tensor = self.pe
        assert seq_len <= pe.shape[1] # pe.shape[1] is max_seq_len
        assert d_model == pe.shape[2] # pe.shape[2] is d_model
        x *= d_model**0.5 # scale by sqrt(d_model)
        return x + pe[:, 0:seq_len, :] # pe[:, 0:seq_len, :] is the positional encoding matrix for the current batch, plus the positional encoding matrix for the current batch