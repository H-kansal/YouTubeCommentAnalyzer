from src.exception import youTubeAnalysisException
from src.logger import logging
from src.utils.youtubeCommentScapping import get_comments
import pickle
import os
import pandas as pd
import traceback
import sys,json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from src.utils.downloadModel import download_model
import numpy as np

model_info_file_path=os.path.join("artifact","modeltraining","model_info.json")


class PredictionPipeline:
    def __int__(self):
        pass

    def predition(self,comments):
        try:
            data=comments.drop("PublishedAt",axis=1)
            download_model()
            
            with open(model_info_file_path,"r") as f:
                model_info=json.load(f)
            
            # model input format is different for dl and ml
            if model_info["model_type"]=="dl":
                
                model=load_model(model_info["model_file_path"])
                with open(model_info["preProcessor_file_path"],"rb") as f:
                    transformer=pickle.load(f)
                
                transformed_data=transformer.texts_to_sequences(data["CommentText"])
                for i in range(len(transformed_data)):
                    if len(transformed_data[i])>model_info["maxLen"]:
                        transformed_data[i]=transformed_data[i][-(model_info["maxLen"]):]
                transformed_data=pad_sequences(transformed_data,maxlen=model_info["maxLen"],padding="pre")
            else:
                with open(model_info["model_file_path"],"rb") as f:
                    model=pickle.load(f)
                with open(model_info["preProcessor_file_path"],"rb") as f:
                    transformer=pickle.load(f)
                
                transformed_data=transformer.transform(data)
            
            prediction_output=model.predict(transformed_data)

            # for dl models that give output as probablity of classes, we chose max one 
            if model_info["model_type"]=="dl":
                prediction_output=np.argmax(prediction_output,axis=1)
            
            prediction_output=pd.DataFrame(prediction_output,columns=["Sentiment"])
            sentiment_Count={
                "Total":int(prediction_output.shape[0]),
                "Positive":int((prediction_output["Sentiment"]==2).sum()),
                "Negative":int((prediction_output["Sentiment"]==0).sum()),
                "Neutral":int((prediction_output["Sentiment"]==1).sum())
            }

            return sentiment_Count,prediction_output
        except Exception as e:
            print(traceback.format_exc())
            youTubeAnalysisException(e,sys)
    
    def commentsFetching(self,videoUrl):
        try:
            comments=get_comments(videoUrl)
            comments["Likes"]=comments["Likes"].astype(int)
            comments["PublishedAt"]=pd.to_datetime(comments["PublishedAt"])
            comments["CommentText"]=comments["CommentText"].fillna("")

            sentiment_count,prediction_output=self.predition(comments)
            comments=pd.concat([comments,prediction_output],axis=1).reset_index()
            return comments,sentiment_count
        except Exception as e:
            print(traceback.format_exc())
            youTubeAnalysisException(e,sys)
    

    def Intialize_Prediction_Pipeline(self,videoUrl):
        return self.commentsFetching(videoUrl)


# if __name__=="__main__":
#     prediction=PredictionPipeline()
#     df,sentiment_count=prediction.Intialize_Prediction_Pipeline("https://www.youtube.com/watch?v=Cb6wuzOurPc")
#     print(sentiment_count)