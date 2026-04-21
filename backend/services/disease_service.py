import json
from config import DATA_PATH

with open(DATA_PATH, "r", encoding="utf-8") as f:
    DISEASE_DATA = json.load(f)

def get_disease_info(disease_name):
    for d in DISEASE_DATA:
        if d["name"] == disease_name:
            return d
    return None