import json
import os

class Struct:
    """Helper class to parse dict to object"""
    def __init__(self, entries):
        self.__dict__.update(entries)

class Selector:
    def __init__(self, dataset, version):
        self.explainer_map = {
            0: "orexplainer",
            1: "gnnexplainer",
            2: "pgexplainer",
            3: "mixupexplainer",
            4: "proxyexplainer",
            5: "vinfor"
        }
        self.args = self.parse_config(f'./Explainer/configs/explainer/{dataset}.json', version)
    def parse_config(self, config_path, version):
        try:
            explainer = self.explainer_map[version]

            with open(config_path) as config_parser:
                config = json.load(config_parser)[explainer]
            return config
        except FileNotFoundError:
            print("No config found")
            return None