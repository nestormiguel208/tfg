import pandas as pd
import yaml
from scipy.io import arff

def load_file(file_path):
    if file_path.endswith(".arff"):
        data, _ = arff.loadarff(file_path)
        return pd.DataFrame(data)
    elif file_path.endswith((".xls", ".xlsx")):
        return pd.read_excel(file_path)
    elif file_path.endswith(".csv"):
        return pd.read_csv(file_path)
    else:
        raise ValueError("Formato de archivo no soportado.")

def save_config(config, file_name="auto_config.yml"):
    with open(file_name, "w") as f:
        yaml.dump(config, f, allow_unicode=True)
