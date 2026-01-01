## just created for deploying in Beanstalk.... you can see thta in .ebextensions->python.config file, it requires an application:application entry points
## and we deleted app.py file
from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import customData,predictPipeline

application=Flask(__name__)
app=application

##route for home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoints():
    if request.method=='GET':
        return render_template('home.html')
    else:
        # this same class will also present in predict pipeline
        data=customData(   ##object of cusomData class
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('writing_score')),
            writing_score=float(request.form.get('reading_score'))
        )
        pred_df=data.get_data_as_frame()
        print(pred_df)


        predict_pipeline=predictPipeline()
        results=predict_pipeline.predict(pred_df)
        return render_template('home.html', results=results[0])


if __name__=="__main__":
    app.run(host="0.0.0.0")
