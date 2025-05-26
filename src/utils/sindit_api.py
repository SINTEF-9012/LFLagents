import json
import requests
from typing import List, Dict
import os
from dotenv import load_dotenv, find_dotenv
from urllib.parse import quote

load_dotenv(find_dotenv(), override=True)

SINDIT2_API_URL = os.getenv("SINDIT2_API_URL")
SINDIT2_USERNAME = os.getenv("SINDIT2_USERNAME")
SINDIT2_PASSWORD = os.getenv("SINDIT2_PASSWORD")

def get_token():
    url = SINDIT2_API_URL + "/token"
    data = {
        "username": SINDIT2_USERNAME,
        "password": SINDIT2_PASSWORD
    }
    headers = {
        "Content-Type": "application/x-www-form-urlencoded"
    }
    response = requests.post(url, data=data, headers=headers)
    if response.status_code == 200:
        return response.json()["access_token"]
    else:
        raise Exception(f"Failed to get token: {response.status_code} - {response.text}")

TOKEN = get_token()

def load_json(filepath: str):
    with open(filepath, 'r') as f:
        return json.load(f)

def query_get_api(endpoint: str, uri: str = None, uri_key: str = "node_uri") -> dict:
    """
    Generic GET API call for the SINDIT Knowledge Graph.
    
    - If you provide a URI and a key (such as 'node_uri', 'class_uri', or 'type_uri'), the function will fetch data for that specific node or class.
    - If you do not provide a URI, the function will fetch all nodes.

    Examples:
        # Get all nodes
        query_get_api_with_uri('kg/nodes')

        # Get a specific node by its URI
        query_get_api_with_uri('kg/node', uri="http://sindit.sintef.no/2.0#temperature", uri_key="node_uri")

        # Get all nodes of a specific type
        query_get_api_with_uri('kg/node_types', uri="urn:samm:sindit.sintef.no:1.0.0#AbstractAsset", uri_key="type_uri")
    """
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {TOKEN}"
    }
    if uri:
        encoded_uri = quote(uri, safe='')
        url = f'{SINDIT2_API_URL}/{endpoint}?{uri_key}={encoded_uri}'
    else:
        url = f'{SINDIT2_API_URL}/{endpoint}'
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return {}
    

def query_post_api(endpoint: str, json: dict) -> dict:
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {TOKEN}",
        "Content-Type": "application/json"
    }
    response = requests.post(f'{SINDIT2_API_URL}/{endpoint}', json=json, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f'Error: {response.status_code}: {response.text}')
