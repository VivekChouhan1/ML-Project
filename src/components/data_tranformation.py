## Here we will do, data Transformation , data cleaning, feature engineering, scaling, ets
import sys
from dataclasses import dataclass 
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer  ## used to create pipeline, like first we do one hot encoding,then we do standerd scaling
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
import os
from src.exception import CustomException
from src.logger import logging

from src.utils import save_object ## for saving pkl file in utils
 
## any input required in this data transformation, will be give through this fn
@dataclass  ## we can directly define , our class variable by this decorator
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTranformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_obj(self):

        '''
        This function is responsible for data Transformation of different types
        of data , it is using pipelines for both cat and num feature
        '''

        try:
            numerical_columns=['reading_score', 'writing_score']
            categorical_columns=['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']

            ##here we will first handle the missing values and then scaling and it run on training dataset
            num_pipeline=Pipeline(
                steps=[
                    ("imfputer", SimpleImputer(strategy='median')),  ## for missing values
                    ("scaler",StandardScaler())
                ]
            )
            logging.info('Numerical columns Standerd Scaling completed')

            ## for categorical feature also , we will make pipeline for first missing value, then encoding and if we want we can do scaling as well
            cat_pipeline=Pipeline(
                steps=[
                    ("Imputer",SimpleImputer(strategy='most_frequent')),
                    ("One_Hot_Encoder",OneHotEncoder(sparse_output=False)),
                    ("scaler",StandardScaler())
                ]
            )
            logging.info('categorical columns encoding completed')


            ## now we will combine this both pipeline by column transformer
            preprocessor=ColumnTransformer(
                [
                    ("Num_pipeline", num_pipeline, numerical_columns),
                    ('Cat_Pipeline', cat_pipeline,categorical_columns)
                ]
            )
            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        

    def initiate_data_transformation(self, train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test completed")

            logging.info("Obtaining preprocessing object")

            preprocessor_obj=self.get_data_transformer_obj()

            target_column_name="math_score"
            numerical_col=['reading_score', 'writing_score']

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)  ## will drop math_score
            target_feature_train_df=train_df[target_column_name]
            
            
            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)  ## will drop math_score
            target_feature_test_df=test_df[target_column_name]
            logging.info("applying preprocesor obj on training and test dataset")

            input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessor_obj.transform(input_feature_test_df)




            '''In NumPy, np.c_ is a special object used as a shorthand for concatenation along the second axis (column-wise). 
            It provides a convenient way to build arrays, 
            particularly when stacking 1-D arrays as columns into a 2-D array. '''
            train_arr=np.c_[
                input_feature_train_arr,np.array(target_feature_train_df)
            ]

            test_arr=np.c_[
                input_feature_test_arr,np.array(target_feature_test_df)
            ]
            logging.info("Saved preprocessing Object")


            # we have to create this preprocessing_obj into pkl file,  we have given pkl file pathin DataTransformationConfig, so we have to save it on same location
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            ) 
            ## Now I write this function ,.. in utils files

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e,sys)