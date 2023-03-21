from fastapi import FastAPI
from pydantic import BaseModel
from databricks_auto_ml import DatabricksAutoML
import uvicorn

app = FastAPI()

#Replace with your actual serving endpoint URL and API token
serving_endpoint_url = "https://adb-5976771229466627.7.azuredatabricks.net/serving-endpoints/heart-attack/invocations"
api_token = "dapi1b7700dffe48b29d4d16386bf046c490-3"

auto_ml = DatabricksAutoML(serving_endpoint_url, api_token)

class InputData(BaseModel):
    #text: str
    age: int
    sex: int
    cp: int
    trtbps: int
    chol: int
    fbs: int
    restecg: int
    thalachh: int
    exng: int
    oldpeak: float
    slp: int
    caa: int
    thall: int

@app.get("/")
async def root():
    return {"message": "Hello Model"}

@app.post("/predict")
async def predict(input_data: InputData):
    """data = {
        "instances": [
            {"text": input_data.text}
        ]
    }"""

    data = {
        "dataframe_split": {
            "columns": [
                "age",
                "sex",
                "cp",
                "trtbps",
                "chol",
                "fbs",
                "restecg",
                "thalachh",
                "exng",
                "oldpeak",
                "slp",
                "caa",
                "thall"
            ],
            "data": [
                [
                    input_data.age,
                    input_data.sex,
                    input_data.cp,
                    input_data.trtbps,
                    input_data.chol,
                    input_data.fbs,
                    input_data.restecg,
                    input_data.thalachh,
                    input_data.exng,
                    input_data.oldpeak,
                    input_data.slp,
                    input_data.caa,
                    input_data.thall
                ]
            ]
        }
    }
    
    predictions = auto_ml.predict(data)
    return {"predictions": predictions}

if __name__ == '__main__':
    uvicorn.run(app, port=8000, host='0.0.0.0')
