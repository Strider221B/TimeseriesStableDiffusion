import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tokenizers import Tokenizer

from constants import Constants as const

class TextDataset(Dataset):

    def __init__(self, 
                 dataset, 
                 tokenizer: Tokenizer, 
                 language: str, 
                 sequence_length: int) -> None:
        super().__init__()
        self._dataset = dataset
        self._tokenizer_source = tokenizer
        self._language_source = language
        self._sequence_length = sequence_length

        self._token_sos = torch.Tensor([tokenizer.token_to_id([const.TOKEN_START_OF_SENTENCE])], dtype=torch.int64)
        self._token_eos = torch.Tensor([tokenizer.token_to_id([const.TOKEN_END_OF_SENTENCE])], dtype=torch.int64)
        self._token_pad = torch.Tensor([tokenizer.token_to_id([const.TOKEN_PADDING])], dtype=torch.int64)

    def __len__(self):
        return len(self._dataset)
    
    def __getitem__(self, index) -> dict:
        source_text = self._dataset[index]
        source_tokens = self._tokenizer_source.encode(source_text).ids
        # source will need to have the EOS and SOS appended later so subtracting 2
        no_of_padding_tokens_source = self._get_number_of_padding_tokens_required_for(source_tokens, 2, source_text)
        encoder_input = torch.cat([
            self._token_sos,
            torch.tensor(source_tokens, dtype=torch.int64),
            self._token_eos,
            torch.tensor([self._token_pad] * no_of_padding_tokens_source, dtype=torch.int64)
        ])

        return {
            const.INPUT_ENCODER: encoder_input,
            const.MASK_ENCODER: self._get_padding_mask(encoder_input),
            const.TEXT_SOURCE: source_text
        }
    
    def _get_padding_mask(self, tensor: torch.Tensor):
        return (tensor != self._token_pad).unsqueeze(0).unsqueeze(0).int() # (1, 1, sequence_len) => for sequence and batch dimension later

    def _get_number_of_padding_tokens_required_for(self, 
                                                   tokens: list,
                                                   tokens_to_be_added_later: int,
                                                   original_str: str = '') -> int:
        no_of_tokens = self._sequence_length - len(tokens) - tokens_to_be_added_later
        if no_of_tokens < 0:
            raise ValueError(f'Sentence is too long. Original string - {original_str}')
        return no_of_tokens
    