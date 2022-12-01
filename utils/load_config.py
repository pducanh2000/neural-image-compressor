import json
import yaml


def load_json_config(json_file):
    with open(json_file, "r") as F:
        config = json.load(F)
    return config


def load_yaml_config(yaml_file):
    with open(yaml_file, "r") as F:
        config = yaml.safe_load(F)
    return config
