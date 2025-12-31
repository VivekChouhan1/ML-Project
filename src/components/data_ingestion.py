## Aim to read data-source, from-where it located and 
## split in train test spilt and then add in Artifacts folder

import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass   ##used to create class variable


from src.components.data_tranformation import DataTranformation
from src.components.data_tranformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

## any input required in this data Ingestion, will be give through this fn
@dataclass  ## we can directly define , our class variable by this decorator
class DataIngestionConfig:
    train_data_path:str=os.path.join('artifacts','train.csv')   ## store in artifect folder and train.csv file
    test_data_path:str=os.path.join('artifacts','test.csv')
    raw_data_path:str=os.path.join('artifacts','data.csv')


class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("entered the data ingestion method or component")
        try:
            df=pd.read_csv('notebook/data/stud.csv')   ## cana slso read from musql, mongoDB, etc
       
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)  ## by this artifect will be created

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("Train Test split initiated")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info('Ingestion of data is completed')

            return(  ## return this, because they will be used in data_transformation
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)



# # to check
if __name__=="__main__":
    obj=DataIngestion()
    # obj.initiate_data_ingestion()
    train_data,test_data=obj.initiate_data_ingestion()

    data_transformation=DataTranformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)

    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))

