from src.entity.config import DataIngestionConfig
from src.logger import logging
from src.exception import youTubeAnalysisException
from dotenv import load_dotenv
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from src.entity.artifact import DataIngestionArtifact
import re
import sys
import os
import pandas as pd
import pymongo

load_dotenv()


lemmatizer=WordNetLemmatizer()
stopwords=set(stopwords.words("english"))

class DataIngestion:
    def __init__(self):
        self.data_ingestion_config=DataIngestionConfig()
        self.mongo_uri=os.getenv("MONGO_URI")

    def clean_text(self,text):
        text=str(text).lower()
        text=re.sub(r'[^\x00-\x7F]+','',text)  # emoji regex
        text=re.sub(r'[^a-zA-Z\s]','',text)    # remove special characters
        text=" ".join(text.split())            # remove extra spaces

        # these below two things are not needed in deep learning models
        # text=" ".join([w for w in text.split() if w not in stopwords])
        # text=" ".join([lemmatizer.lemmatize(w) for w in text.split()])
        return text


    def load_data(self):
        try:
            mongoClient=pymongo.MongoClient(self.mongo_uri)
            db_instance=mongoClient[self.data_ingestion_config.db_name]
            db_collection=db_instance[self.data_ingestion_config.collection_name]
            df=pd.DataFrame(list(db_collection.find({},{"_id":1,"CommentText":1,"Sentiment":1})))
            df.drop(["_id"],axis=1,inplace=True)
            # CommentText cleaning
            df["CommentText"]=df["CommentText"].apply(self.clean_text)
            # correcting datatype
            # df["Likes"]=df["Likes"].astype(int)
            # df["Replies"]=df["Replies"].astype(int)
            # mapping sentiment column
            df["Sentiment"]=df["Sentiment"].map({"Positive":2,"Neutral":1,"Negative":0})
            train_data,test_data=train_test_split(df,test_size=0.2,random_state=42)  

            dir_name=os.path.dirname(self.data_ingestion_config.raw_data_path)
            os.makedirs(dir_name,exist_ok=True)
            df.to_csv(self.data_ingestion_config.raw_data_path,index=False)
            train_data.to_csv(self.data_ingestion_config.train_file_path,index=False)
            test_data.to_csv(self.data_ingestion_config.test_file_path,index=False)
            return DataIngestionArtifact(raw_file_path=self.data_ingestion_config.raw_data_path,
                                        train_file_path=self.data_ingestion_config.train_file_path,
                                        test_file_path=self.data_ingestion_config.test_file_path)
        except Exception as e:
            raise youTubeAnalysisException(e,sys)
    
    def Intialize_DataIngestion(self):
        try:
            dataIngestionArtifact=self.load_data()
            return dataIngestionArtifact
        except Exception as e:
            raise youTubeAnalysisException(e,sys)