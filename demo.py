from urllib.parse import urlparse
import mlflow
import dagshub


dagshub.init(repo_owner='Mahesh-martian', repo_name='Chest_Disease_Classification', mlflow=True)

mlflow_uri="https://dagshub.com/Mahesh-martian/Chest_Disease_Classification.mlflow"
mlflow.set_registry_uri(mlflow_uri)
print('mlflow uri validated')
tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
print(tracking_url_type_store)

model_name = 'VGG16Model'
stage = 'latest'

mlflow.pyfunc.load_model(model_uri = f"models:/{model_name}/{stage}")

