from fastapi import FastAPI
from pydantic import BaseModel
from databricks_auto_ml import DatabricksAutoML

app = FastAPI()

# Replace with your actual serving endpoint URL and API token
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

    """data ={
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
        63,
        1,
        3,
        145,
        233,
        1,
        0,
        150,
        0,
        2.3,
        0,
        0,
        1
      ],
      [
        57,
        0,
        0,
        120,
        354,
        0,
        1,
        163,
        1,
        0.6,
        2,
        0,
        2
      ],
      [
        56,
        0,
        1,
        140,
        294,
        0,
        0,
        153,
        0,
        1.3,
        1,
        0,
        2
      ],
      [
        44,
        1,
        1,
        120,
        263,
        0,
        1,
        173,
        0,
        0,
        2,
        0,
        3
      ],
      [
        35,
        1,
        0,
        126,
        282,
        0,
        0,
        156,
        1,
        0,
        2,
        0,
        3
      ]
    ]
  }
}"""
    predictions = auto_ml.predict(data)
    return {"predictions": predictions}
