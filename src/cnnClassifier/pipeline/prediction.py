import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
import mlflow
import dagshub



class PredictionPipeline:
    def __init__(self,filename):
        self.filename =filename
    
    def predict(self):

        dagshub.init(repo_owner='Mahesh-martian', repo_name='Chest_Disease_Classification', mlflow=True)

        mlflow_uri="https://dagshub.com/Mahesh-martian/Chest_Disease_Classification.mlflow"
        mlflow.set_registry_uri(mlflow_uri)
        print('mlflow uri validated')
        ## load model
        model_name = 'VGG16Model'
        stage = 'Production'

        model = mlflow.pyfunc.load_model(model_uri = f"models:/{model_name}/{stage}")
        
        # model = load_model(os.path.join("artifacts","training", "model.h5"))
        # model = load_model(os.path.join("model", "model.h5"))

        imagename = self.filename
        test_image = image.load_img(imagename, target_size = (224,224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = np.argmax(model.predict(test_image), axis=1)
        print(result)

        if result[0] == 1:
            prediction = 'Normal'
            return [{ "image" : prediction}]
        else:
            prediction = 'Adenocarcinoma Cancer'
            return [{ "image" : prediction}]