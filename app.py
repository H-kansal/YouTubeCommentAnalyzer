from starlette.responses import RedirectResponse
from fastapi.responses import Response,FileResponse
from fastapi import FastAPI, File,UploadFile,Request
from fastapi.middleware.cors import CORSMiddleware
from src.pipeline.trainingPipeline import TrainingPipeline
from src.pipeline.predictionPipeline import PredictionPipeline
from src.exception import youTubeAnalysisException
from uvicorn import run as app_run
import sys
import os
import pandas as pd
import traceback
from pydantic import BaseModel
app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class VideoRequest(BaseModel):
    video_url:str

@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")

# @app.get("/train")
# async def train():
#     try:
#         train=TrainingPipeline()
#         train.intialize_training_pipeline()
#         return {"message":"Training successful"}
#     except Exception as e:
#         print(traceback.format_exc())
#         raise youTubeAnalysisException(e,sys)

@app.post("/predict")
async def predict(request:VideoRequest):
    try:
        prediction=PredictionPipeline()
        df,sentiment_count=prediction.Intialize_Prediction_Pipeline(request.video_url)
        return {
            "df":df.to_dict(orient="records"),"sentiment_count":sentiment_count
        }
    except Exception as e:
        raise youTubeAnalysisException(e,sys)

if __name__=="__main__":
    app_run(app,host="0.0.0.0",port=8000)

# def trainModel():
#     try:
#         train=TrainingPipeline()
#         train.intialize_training_pipeline()
#         return {"message":"Training successful"}
#     except Exception as e:
#         print(traceback.format_exc())
#         raise youTubeAnalysisException(e,sys)

# print(trainModel())