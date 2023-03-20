import requests
import json

class DatabricksAutoML:

    def __init__(self, endpoint_url, api_token):
        self.endpoint_url = endpoint_url
        self.api_token = api_token

    def predict(self, data):
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_token}'
        }
        response = requests.post(self.endpoint_url, data=json.dumps(data), headers=headers)

        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Error: {response.status_code}, {response.text}")
