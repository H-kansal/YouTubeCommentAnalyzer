from dataclasses import dataclass
import numpy as np
from typing import Any

@dataclass
class DataIngestionArtifact:
    raw_file_path:str
    train_file_path:str
    test_file_path:str

@dataclass
class DataTransformationArtifact:
    X_train: Any   # sparse matrix (TF-IDF output)
    X_test: Any
    y_train: np.ndarray
    y_test: np.ndarray
    processor_file_path: str
    tokenizer_file_path: str
    vocabluraySize:int


@dataclass
class ModelTrainingArtifact:
    ml_model_file_path:str
    dl_model_file_path:str
    processor_file_path:str
    tokenizer_file_path:str
    isDLModel:bool
