from src.exception import youTubeAnalysisException
from src.logger import logging
from src.entity.config import ModelTrainingConfig
from src.entity.artifact import ModelTrainingArtifact
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from src.utils.modelsTrainng import best_dl_model,best_ml_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding,LSTM,Dense,Dropout,GRU
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.sequence import pad_sequences



import os,sys,json
import pickle
import traceback

class ModelTraining:
    def __init__(self,datatransformationArtifact):
        self.modelTrainingConfig=ModelTrainingConfig()
        self.tranformationArtifact=datatransformationArtifact
    
    def LSTM_model(self,inputSize):
        model=Sequential()
        model.add(Embedding(self.tranformationArtifact.vocabluraySize,64,input_length=inputSize))
        model.add(LSTM(32,return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(16,return_sequences=False))
        model.add(Dense(3,activation="softmax"))
        model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])
        return model

    def GRU_model(self,inputSize):
        model=Sequential()
        model.add(Embedding(self.tranformationArtifact.vocabluraySize,64,input_length=inputSize))
        model.add(GRU(32,return_sequences=True))
        model.add(Dropout(0.2))
        model.add(GRU(16,return_sequences=False))
        model.add(Dense(3,activation="softmax"))
        model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])  # we can aslo use categorical_crossentropy but for that we have to also use one hot encoding for y_train
        return model
    
    def train_model(self):
        try:

            with open(self.tranformationArtifact.processor_file_path,"rb") as f:
                processor=pickle.load(f)
            with open(self.tranformationArtifact.tokenizer_file_path,"rb") as f:
                tokenizer=pickle.load(f)
            
            # this below two are for dl models
            X_train_tokenize=tokenizer.texts_to_sequences(self.tranformationArtifact.X_train["CommentText"])
            X_test_tokenize=tokenizer.texts_to_sequences(self.tranformationArtifact.X_test["CommentText"])

            # these below are of ml models
            X_train_tfIdf=processor.fit_transform(self.tranformationArtifact.X_train)
            X_test_tfIdf=processor.transform(self.tranformationArtifact.X_test)
            
            maxLen=0
            for data in X_train_tokenize:
                if len(data)>maxLen:
                    maxLen=len(data)
            for data in X_test_tokenize:
                if len(data)>maxLen:
                    maxLen=len
            
            # padding so that input size to embedding layer is same
            X_train_tokenize=pad_sequences(X_train_tokenize,maxlen=maxLen,padding="pre")
            X_test_tokenize=pad_sequences(X_test_tokenize,maxlen=maxLen,padding="pre")

            ml_models={
                "NaiveBias":MultinomialNB(),
                "LogisticRgression":LogisticRegression(max_iter=1000),
                "SVC":LinearSVC()
            }
            dl_models={
                "LSTM":self.LSTM_model(maxLen),
                "GRU":self.GRU_model(maxLen)
            }
            
            # for ml models
            best_ml_model_instance,ml_model_name,ml_report=best_ml_model(ml_models,X_train_tfIdf,X_test_tfIdf,self.tranformationArtifact.y_train,self.tranformationArtifact.y_test)

            # for dl models
            best_dl_model_instance,dl_model_name,dl_report=best_dl_model(dl_models,X_train_tokenize,X_test_tokenize,self.tranformationArtifact.y_train,self.tranformationArtifact.y_test)

            overall_best_model=None
            overall_best_model_name=None
            isDlModel=False
            if ml_report[ml_model_name]>dl_report[dl_model_name]:
                overall_best_model=best_ml_model_instance
                overall_best_model_name=ml_model_name
            else:
                overall_best_model=best_dl_model_instance
                overall_best_model_name=dl_model_name
                isDlModel=True

            dir_name=os.path.dirname(self.modelTrainingConfig.ml_model_file_path)
            os.makedirs(dir_name,exist_ok=True)

            
            if isDlModel:
                overall_best_model.save(self.modelTrainingConfig.dl_model_file_path)
            else:
                with open(self.modelTrainingConfig.ml_model_file_path,"wb") as f:
                    pickle.dump(overall_best_model,f)
            
            with open(self.modelTrainingConfig.model_info_path,"w") as f:
                json.dump({"model_type":"dl" if isDlModel else "ml","model_name":overall_best_model_name,"model_file_path":self.modelTrainingConfig.dl_model_file_path if isDlModel else self.modelTrainingConfig.ml_model_file_path,"preProcessor_file_path":self.tranformationArtifact.tokenizer_file_path if isDlModel else self.tranformationArtifact.processor_file_path,"maxLen":maxLen},f)


            return ModelTrainingArtifact(ml_model_file_path=self.modelTrainingConfig.ml_model_file_path,dl_model_file_path=self.modelTrainingConfig.dl_model_file_path,processor_file_path=self.tranformationArtifact.processor_file_path,tokenizer_file_path=self.tranformationArtifact.tokenizer_file_path,isDLModel=isDlModel)
        except Exception as e:
            print(traceback.format_exc())
            youTubeAnalysisException(e,sys)
    
    def Intialize_training(self):
        trainingArtifact=self.train_model()
        return trainingArtifact