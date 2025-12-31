## we will train different different model here , and see whats the accuracy, what's the score
import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:   ##this give whatever input I require for model training
    trained_model_filepath=os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_conf=ModelTrainerConfig()

    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_arr[:,:-1], ## take out last colum and feed everything in x train,
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            ## model to be test with:
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            ## We will create this evaluate in utils.py
            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models) 


            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ] 

            best_model=models[best_model_name]


            ## if all  model giving score less then 0.6
            if best_model_score<0.6:
                raise CustomException("No best model found!")
            logging.info(f"Best found model on both training and testing dataset")


            ## Now to convert this model in pkl, so save it and give it to utils function
            save_object(
                file_path=self.model_trainer_conf.trained_model_filepath,
                obj=best_model
            )


            predicted=best_model.predict(X_test)
            r2=r2_score(y_test,predicted)
            return r2

        except Exception as e:
            raise CustomException(e,sys)