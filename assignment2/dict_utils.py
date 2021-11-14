import os
import json

def save_file(json_path:str,data:dict):
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path,"w+") as json_file:
        json.dump(data,json_file)