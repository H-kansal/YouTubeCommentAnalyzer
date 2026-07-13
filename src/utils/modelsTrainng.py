from sklearn.metrics import f1_score,classification_report,precision_score,recall_score,accuracy_score
from src.exception import youTubeAnalysisException
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from src.logger import logging
import sys
import traceback
import numpy as np
from src.constant.training import METRICS_FILE_NAME,MODEL_DIR,DATA_DIR
import os
import pandas as pd


METRICS_FILE_PATH=os.path.join(DATA_DIR,MODEL_DIR,METRICS_FILE_NAME)

METRICS_DIR_PATH=os.path.dirname(METRICS_FILE_PATH)

os.makedirs(METRICS_DIR_PATH,exist_ok=True)


def best_dl_model(models,X_train,X_test,y_train,y_test):
    try:
        report=[]
        f1_report={}
        model_name=None
        best_model=None
        best_accuracy=-np.inf
        earlyStopping=EarlyStopping(patience=5,monitor="val_loss",restore_best_weights=True)
        for i in range(len(list(models))):
            model=list(models.values())[i]
            model.fit(X_train,y_train,epochs=50,batch_size=1024,validation_split=0.2,verbose=1,callbacks=[earlyStopping])
            
            test_pred_one_hot=model.predict(X_test)
            test_pred=np.argmax(test_pred_one_hot,axis=1)    # loss function only affect the training,not the prediction result
            test_f1_score=f1_score(y_test,test_pred,average="macro")
            test_precision_score=precision_score(y_test,test_pred,average="macro")
            test_recall_score=recall_score(y_test,test_pred,average="macro")
            test_accuracy_score=accuracy_score(y_test,test_pred)
            
            if test_f1_score>best_accuracy:
                model_name=list(models.keys())[i]
                best_model=model
                best_accuracy=test_f1_score
            f1_report[list(models.keys())[i]]=test_f1_score
            report.append({'model':list(models.keys())[i],'f1':test_f1_score,'precision':test_precision_score,'recall':test_recall_score,'accuracy':test_accuracy_score})
            
        df_report=pd.DataFrame(report)
        print("This Report is for dl models",df_report)
        df_report.to_csv(METRICS_FILE_PATH,index=False,header= not os.path.exists(METRICS_FILE_PATH),mode="a")
        return best_model,model_name,f1_report
    except Exception as e:
        print(traceback.format_exc())
        youTubeAnalysisException(e,sys)


def best_ml_model(models,X_train,X_test,y_train,y_test):
    try:
        report=[]
        f1_report={}
        best_model=None
        model_name=None
        best_accuracy=-np.inf
        for i in range(len(list(models))):
            model=list(models.values())[i]
            model.fit(X_train,y_train)

            test_pred=model.predict(X_test)
            test_f1_score=f1_score(y_test,test_pred,average="macro")
            test_precision_score=precision_score(y_test,test_pred,average="macro")
            test_recall_score=recall_score(y_test,test_pred,average="macro")
            test_accuracy_score=accuracy_score(y_test,test_pred)

            if test_f1_score>best_accuracy:
                model_name=list(models.keys())[i]
                best_model=list(models.values())[i]
                best_accuracy=test_f1_score
            
            f1_report[list(models.keys())[i]]=test_f1_score
            report.append({'model':list(models.keys())[i],'f1':test_f1_score,'precision':test_precision_score,'recall':test_recall_score,'accuracy':test_accuracy_score})
            
        df_report=pd.DataFrame(report)
        print("the ml models report is",df_report)
        df_report.to_csv(METRICS_FILE_PATH,index=False,header= not os.path.exists(METRICS_FILE_PATH),mode="a")

        return best_model,model_name,f1_report
    except Exception as e:
        print(traceback.format_exc())
        youTubeAnalysisException(e,sys)

