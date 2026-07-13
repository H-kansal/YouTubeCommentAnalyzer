from dotenv import load_dotenv
import pymongo
from src.exception import youTubeAnalysisException
from src.logger import logging
import os
import pandas as pd
import json
import sys

load_dotenv()

class PushData:
    def __init__(self):
        try:
            self.MONGOURI=os.getenv("MONGO_URI")
            self.MONGODB=os.getenv("MONGO_DB")

            if not self.MONGOURI:
                raise Exception("MONGO_URI is not defined")
            if not self.MONGODB:
                raise Exception("MONGO_DB is not defined")
        except Exception as e:
            raise youTubeAnalysisException(e,sys)


    def csv_to_json(self,file_path:str):
        try:
            csv_data=pd.read_csv(file_path)
            csv_data.reset_index(drop=True,inplace=True)
            csv_data.drop_duplicates(subset=["CommentID"],inplace=True)
            json_data=list(json.loads(csv_data.T.to_json()).values())
            print("this is the length",len(json_data))
            # with open("youtube_comments_cleaned.json","w") as f:
            #     json.dump(json_data,f,indent=4)
            return json_data[:1000000]
        except Exception as e:
            raise youTubeAnalysisException(e,sys)

    def push_data(self,json_data):
        try:
            mongoClient=pymongo.MongoClient(self.MONGOURI)
            db_instance=mongoClient[self.MONGODB]
            db_collection=db_instance["youtube_data"]
            batch_size=50000
            for i in range(0,len(json_data),batch_size):
                db_collection.insert_many(json_data[i:i+batch_size])
                print("inserted",i,"to",i+batch_size,"records")
            print("this is collection",db_collection)
            return len(json_data)
        except Exception as e:
            raise youTubeAnalysisException(e,sys)

if __name__ == "__main__":
    push_data=PushData()
    json_data=push_data.csv_to_json("youtube_comments_cleaned.csv")
    json_len=push_data.push_data(json_data)
    print(json_len)