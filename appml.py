from databricks_auto_ml import DatabricksAutoML

# Replace with your actual serving endpoint URL and API token
serving_endpoint_url = "https://adb-5976771229466627.7.azuredatabricks.net/serving-endpoints/fake-news/invocations"
api_token = "dapi1b7700dffe48b29d4d16386bf046c490-3"

# Initialize the DatabricksAutoML class
auto_ml = DatabricksAutoML(serving_endpoint_url, api_token)

# Replace with your actual input data
input_data = {
    "instances": [
        {"text": "This is a test input for fake news detection"}
    ]
}

# Call the predict method to get predictions
predictions = auto_ml.predict(input_data)

# Print predictions
print(predictions)
