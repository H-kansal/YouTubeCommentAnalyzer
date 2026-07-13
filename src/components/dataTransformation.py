from src.entity.config import DataTransformationConfig
from src.entity.artifact import DataTransformationArtifact
from src.exception import youTubeAnalysisException
from src.logger import logging
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import sys
import pandas as pd
import os
import pickle
import traceback

class DataTransformation:
    def __init__(self,dataIngestionArtifact):
        self.dataTransformationConfig = DataTransformationConfig()
        self.ingestionArtifact=dataIngestionArtifact
    
    def get_preprocessor(self):
        #tf-idf for comments
        try:
            text_pipeline=Pipeline([
                ("tf-idf",TfidfVectorizer(max_features=100000))
            ])
            processor=ColumnTransformer([
                ("text",text_pipeline,"CommentText")
            ])
            
            return processor
        except Exception as e:
            raise youTubeAnalysisException(e,sys)
    
    def tokenization(self,trainData):
        tokenizer=Tokenizer()
        tokenizer.fit_on_texts(trainData["CommentText"])
        return tokenizer

    def transformation(self):
        try:
            processor=self.get_preprocessor()
            train_data=pd.read_csv(self.ingestionArtifact.train_file_path)
            test_data=pd.read_csv(self.ingestionArtifact.test_file_path)
            train_data["CommentText"] = train_data["CommentText"].fillna("").astype(str)
            test_data["CommentText"] = test_data["CommentText"].fillna("").astype(str)

            tokenizer=self.tokenization(train_data)

            X_train=train_data.drop("Sentiment",axis=1)
            X_test=test_data.drop("Sentiment",axis=1)
            y_train=train_data["Sentiment"]
            y_test=test_data["Sentiment"]
            print("Hello:",X_train.shape,X_test.shape)

            # X_train=tokenizer.texts_to_sequences(X_train["CommentText"])
            # X_test=tokenizer.texts_to_sequences(X_test["CommentText"])
            # for mutliclass classification convert y into vector of 3
            
            # maxLen=0
            # for data in X_train:
            #     if len(data)>maxLen:
            #         maxLen=len(data)
            # for data in X_test:
            #     if len(data)>maxLen:
            #         maxLen=len(data)

            # X_train=pad_sequences(X_train,maxlen=maxLen,padding="pre")
            # X_test=pad_sequences(X_test,maxlen=maxLen,padding="pre")

            dir_name=os.path.dirname(self.dataTransformationConfig.processor_file_path)
            os.makedirs(dir_name,exist_ok=True)


            with open(self.dataTransformationConfig.processor_file_path,"wb") as f:
                pickle.dump(processor,f)
            
            with open(self.dataTransformationConfig.tokenizer_file_path,"wb") as f:
                pickle.dump(tokenizer,f)

            return DataTransformationArtifact(X_train=X_train,X_test=X_test,y_train=y_train,y_test=y_test,processor_file_path=self.dataTransformationConfig.processor_file_path,tokenizer_file_path=self.dataTransformationConfig.tokenizer_file_path,vocabluraySize=len(tokenizer.word_index)+1)
        
        except Exception as e:
            print("ERROR OCCURRED:")
            print(traceback.format_exc())
            raise youTubeAnalysisException(e,sys)

    def Intialize_transformation(self):
        try:
            transformation_artifact=self.transformation()
            return transformation_artifact
        except Exception as e:
            youTubeAnalysisException(e,sys)

