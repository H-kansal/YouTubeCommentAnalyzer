import os
from src.constant import training


class DataIngestionConfig:
    def __init__(self):
        self.db_name=training.DB_NAME
        self.collection_name=training.DB_COLLECTION_NAME
        self.data_dir=training.DATA_DIR
        self.ingestion_dir=os.path.join(self.data_dir,training.INGESTION_DIR)
        self.raw_data_path=os.path.join(self.ingestion_dir,training.RAW_DATA_FILE)
        self.train_file_path=os.path.join(self.ingestion_dir,training.TRAIN_FILE)
        self.test_file_path=os.path.join(self.ingestion_dir,training.TEST_FILE)
        self.test_split_ratio=training.TEST_SPLIT_RATIO

class DataTransformationConfig:
    def __init__(self):
        self.data_dir=training.DATA_DIR
        self.transformation_dir=os.path.join(self.data_dir,training.TRANSORMATION_DIR)
        self.transformed_train_file=os.path.join(self.transformation_dir,training.TRANSFORMATION_TRAIN_FILE)
        self.transformed_test_file=os.path.join(self.transformation_dir,training.TRANSFORMATION_TEST_FILE)
        self.processor_file_path=os.path.join(self.transformation_dir,training.PROCESSOR_FILE_PATH)
        self.tokenizer_file_path=os.path.join(self.transformation_dir,training.TOKENIZER_FILE_PATH)

class ModelTrainingConfig:
    def __init__(self):
        self.data_dir=training.DATA_DIR
        self.model_training_dir=os.path.join(self.data_dir,training.MODEL_DIR)
        self.ml_model_file_path=os.path.join(self.model_training_dir,training.ML_MODEL_FILE_PATH)
        self.dl_model_file_path=os.path.join(self.model_training_dir,training.DL_MODEL_FILE_PATH)
        self.model_info_path=os.path.join(self.model_training_dir,training.MODEL_INFO)