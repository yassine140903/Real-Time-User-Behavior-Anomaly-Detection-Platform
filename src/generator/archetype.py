import yaml

class Archetype:
    def __init__(self, name, operation_mix, frequency_mean, frequency_std,
                 timing, amount_mean, amount_std, counterparty, branch_loyalty):
        self.name = name
        self.operation_mix = operation_mix
        self.frequency_mean = frequency_mean.copy()
        self.frequency_std = frequency_std.copy()
        self.timing = timing.copy()
        self.amount_mean = amount_mean.copy()
        self.amount_std = amount_std.copy()
        self.counterparty = counterparty
        self.branch_loyalty = branch_loyalty


def load_archetypes(config_path="config/archetypes.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    archetypes = {}
    for key, params in config.items():
        if key == "population_proportions":
            continue
        archetypes[key] = Archetype(**params)

    return archetypes, config["population_proportions"]