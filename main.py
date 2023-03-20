from fastapi import FastAPI
from pydantic import BaseModel
from databricks_auto_ml import DatabricksAutoML

app = FastAPI()

# Replace with your actual serving endpoint URL and API token
serving_endpoint_url = "https://adb-5976771229466627.7.azuredatabricks.net/serving-endpoints/fake-news/invocations"
api_token = "dapi1b7700dffe48b29d4d16386bf046c490-3"

auto_ml = DatabricksAutoML(serving_endpoint_url, api_token)

class InputData(BaseModel):
    text: str

@app.post("/predict")
async def predict(input_data: InputData):
    data = {
        "instances": [
            {"text": input_data.text}
        ]
    }
    predictions = auto_ml.predict(data)
    return {"predictions": predictions}
