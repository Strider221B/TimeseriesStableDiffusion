import os
import json
import glob
from typing import Tuple

import pandas as pd
from tokenizers import Tokenizer

from pipeline.config import Config
from pipeline.constants import Constants as const

class Utils:
    
    _DF_FILE_SUFFIX = '_df'
    _FILE_NAME_SEP = '_'
    _INDEX_ASSET = 0
    _INDEX_SPEED = 1
    _INDEX_PROBLEM = 2
    _SPEED_UNIT = 'rpm'

    @classmethod
    def format_data_at(cls, config: dict):
        path = config[const.DATASET_PATH]
        original_files = [f for f in os.listdir(path) if (os.path.isfile(os.path.join(path, f)) and f.endswith(const.EXTN_DATA_FILE))]
        for original_file in original_files:
            cls._extract_metadata_and_part_df(path, original_file, config)
        
    @classmethod
    def _extract_metadata_and_part_df(cls, path: str, file_name: str, config: dict):
        original_df = pd.read_parquet(os.path.join(path, file_name))
        for i, col in enumerate(original_df):
            cls._export_metadata_and_col_data(original_df, col, file_name, i,config)
    
    @classmethod
    def _export_metadata_and_col_data(cls, 
                                      data_frame: pd.DataFrame, 
                                      column_name: str,
                                      original_file_name: str,
                                      column_index: int,
                                      config: dict):
        file_details = original_file_name.removesuffix(const.EXTN_DATA_FILE).removesuffix(cls._DF_FILE_SUFFIX)
        file_details = file_details.split(cls._FILE_NAME_SEP) 
        speed_in_rpm = file_details[cls._INDEX_SPEED]
        speed_val = float(speed_in_rpm.removesuffix(cls._SPEED_UNIT))
        asset_name = file_details[cls._INDEX_ASSET]
        problem = file_details[cls._INDEX_PROBLEM]
        file_name = (f'{asset_name}{cls._FILE_NAME_SEP}'
                     f'{speed_in_rpm}{cls._FILE_NAME_SEP}'
                     f'{problem}{cls._FILE_NAME_SEP}'
                     f'{column_index}')
        data_frame.loc[:, [column_name]].to_parquet(os.path.join(config[const.DATA_EXPORT_PATH], 
                                                                 (f'{file_name}'
                                                                  f'{const.EXTN_DATA_FILE}')))
        metadata = f'Asset {asset_name} running at {speed_val} {cls._SPEED_UNIT} is having {problem} problem.\n'
        with open(os.path.join(f'{config[const.METADATA_EXPORT_PATH]}{file_name}{const.EXTN_METADATA_FILE}'), 'w') as f:
            f.write(metadata)


# def load_hf_model(model_path: str, device: str) -> Tuple[str]:
#     # Load the tokenizer
#     tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
#     assert tokenizer.padding_side == "right"

#     # Find all the *.safetensors files
#     safetensors_files = glob.glob(os.path.join(model_path, "*.safetensors"))

#     # ... and load them one by one in the tensors dictionary
#     tensors = {}
#     for safetensors_file in safetensors_files:
#         with safe_open(safetensors_file, framework="pt", device="cpu") as f:
#             for key in f.keys():
#                 tensors[key] = f.get_tensor(key)

#     # Load the model's config
#     with open(os.path.join(model_path, "config.json"), "r") as f:
#         model_config_file = json.load(f)
#         config = PaliGemmaConfig(**model_config_file)

#     # Create the model using the configuration
#     model = PaliGemmaForConditionalGeneration(config).to(device)

#     # Load the state dict of the model
#     model.load_state_dict(tensors, strict=False)

#     # Tie weights
#     model.tie_weights()

#     return (model, tokenizer)

if __name__ == '__main__':
    Utils.format_data_at(Config.get_config())