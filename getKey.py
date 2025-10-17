import requests

response = requests.post('https://fbrapi.com/generate_api_key')
api_key = response.json()['api_key']
print("API Key:", api_key)