import json
import os

class Struct:
    """Helper class to parse dict to object"""
    def __init__(self, entries):
        self.__dict__.update(entries)

class Selector:
    def __init__(self, dataset, version):
        self.explainer_map = {
                    0: "gnnexplainer",
                    1: "pgexplainer",
                    2: "mixupexplainer",
                    3: "proxyexplainer",
                    4: "orexplainer",
                }
        self.args = self.parse_config(f'./configs/explainer/{dataset}.json', version)
    def parse_config(self, config_path, version):
        try:
            explainer = self.explainer_map[version]
            with open(config_path) as config_parser:
                config = json.load(config_parser)[explainer]
            return config
        except FileNotFoundError:
            print("No config found")
            return None