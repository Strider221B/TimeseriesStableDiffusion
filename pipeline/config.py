from pathlib import Path

from pipeline.constants import Constants as const

class Config:

    def get_config():
        return {
            const.BATCH_SIZE: 8,
            const.DATA_EXPORT_PATH: './dataset/final_data/',
            const.METADATA_EXPORT_PATH: './dataset/final_metadata/',
            const.DATASET_PATH: './dataset/',
            const.NO_OF_EPOCHS: 20,
            const.LEARNING_RATE: 10**-4,
            const.SEQUENCE_LENGTH: 350,
            const.MODEL_DIMENSION: 512,
            const.LANGUAGE_SOURCE: 'en',
            const.MODEL_FOLDER: 'weights',
            const.MODEL_BASENAME: 'tmodel_',
            const.PRELOAD: None,
            const.TOKENIZER_FILE: 'tokenizer_{0}.json',
            const.EXPERIMENT_NAME: 'runs/tmodel',
            const.WAVEFORM_SIZE: 1000,
            const.WAVEFORM_TOKENS: 100
        }
    
    def get_weights_file_path(config, epoch: str):
        model_folder = config[const.MODEL_FOLDER]
        model_basename = config[const.MODEL_BASENAME]
        model_filename = f'{model_basename}{epoch}.pt'
        return str(Path('.') / model_folder / model_filename)