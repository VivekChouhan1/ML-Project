###  used to combine all functionality/module , it will have all the common functionality
import os
import numpy as np
import pandas as pd
import sys
import dill    # used to create pickle file
from src.exception import CustomException

def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)    ## when dump, this obj will save in file_obj in pkl 
    except Exception as e:
        raise CustomException(e,sys)