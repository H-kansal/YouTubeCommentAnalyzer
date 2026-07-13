from src.components.dataIngestion import DataIngestion
from src.components.dataTransformation import DataTransformation
from src.components.modelTraining import ModelTraining


class TrainingPipeline:
    def __init__(self):
        pass
        
    def dataIngestion(self):
        dataIngestion=DataIngestion()
        dataIngestionArtifact=dataIngestion.Intialize_DataIngestion()
        return dataIngestionArtifact
    
    def dataTranformation(self,dataIngestionArtifact):
        dataTransformation=DataTransformation(dataIngestionArtifact)
        dataTranformationArtifact=dataTransformation.Intialize_transformation()
        return dataTranformationArtifact

    def modelTraining(self,dataTransformationArtifact):
        modelTraining=ModelTraining(dataTransformationArtifact)
        modelTraining.Intialize_training()

    def intialize_training_pipeline(self):
        dataIngestionArtifact=self.dataIngestion()
        print(dataIngestionArtifact)
        dataTransformationArtifact=self.dataTranformation(dataIngestionArtifact)
        print(dataTransformationArtifact)
        modelTrainingArtifact=self.modelTraining(dataTransformationArtifact)
        print(modelTrainingArtifact)
        return modelTrainingArtifact







# train=TrainingPipeline()
# train.intialize_training_pipeline()