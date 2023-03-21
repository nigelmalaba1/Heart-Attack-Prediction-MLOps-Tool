from databricks_auto_ml import DatabricksAutoML

# Replace with your actual serving endpoint URL and API token
# serving_endpoint_url = "https://adb-5976771229466627.7.azuredatabricks.net/serving-endpoints/fake-news/invocations"
serving_endpoint_url = "https://adb-5976771229466627.7.azuredatabricks.net/serving-endpoints/heart-attack/invocations"
api_token = "dapi1b7700dffe48b29d4d16386bf046c490-3"

# Initialize the DatabricksAutoML class
auto_ml = DatabricksAutoML(serving_endpoint_url, api_token)

# Replace with your actual input data

""" {
    "instances": [
        {"text": "This is a test input for fake news detection"}
    ]
}
 """
input_data ={
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
}

# Call the predict method to get predictions
predictions = auto_ml.predict(input_data)

# Print predictions
print(predictions)
