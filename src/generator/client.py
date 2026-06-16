import numpy as np
import uuid
import datetime

ACCOUNT_AGE_RANGES = {
    "salaried":       (2, 15),
    "student":        (1, 3),
    "retiree":        (10, 30),
    "small_business":  (1, 10),
    "big_business":    (5, 20),
}

CLIENT_TYPE_MAP = {
    "salaried":      "personne_physique",
    "student":       "personne_physique",
    "retiree":       "personne_physique",
    "small_business": "personne_morale",
    "big_business":   "personne_morale",
}


class Client:
    def __init__(self, archetype, branch_pool):
        self.client_id = str(uuid.uuid4())
        self.account_id = str(uuid.uuid4())
        self.archetype = archetype

        # Layer 2: sample personality from archetype (once, fixed for life)

        # Operation mix — sample then normalize to sum to 1
        raw_mix = {}
        for op, mean in archetype.operation_mix.items():
            raw_mix[op] = np.clip(np.random.normal(mean, mean * 0.15), 0.001, None)
        total = sum(raw_mix.values())
        self.operation_mix = {op: v / total for op, v in raw_mix.items()}

        # Frequency per operation — must be >= 0
        self.frequency = {}
        for op in archetype.frequency_mean:
            drawn = np.random.normal(archetype.frequency_mean[op], archetype.frequency_std[op])
            self.frequency[op] = max(0, round(drawn, 2))

        # Timing — personal preferred day of month
        self.preferred_day = int(np.clip(
            np.random.normal(archetype.timing["day_of_month_mean"], archetype.timing["day_of_month_std"]),
            1, 30
        ))

        # Amount per operation — must be > 0
        self.amount_mean = {}
        self.amount_std = {}
        for op in archetype.amount_mean:
            personal_mean = np.clip(
                np.random.normal(archetype.amount_mean[op], archetype.amount_std[op] * 0.5),
                10, None
            )
            self.amount_mean[op] = round(personal_mean, 2)
            self.amount_std[op] = round(archetype.amount_std[op] * 0.3, 2)

        # Counterparty — probability of known counterparty, clipped to [0, 1]
        self.counterparty_known_ratio = np.clip(
            np.random.normal(archetype.counterparty, 0.1),
            0.0, 1.0
        )

        # Branch loyalty — probability of using home branch, clipped to [0, 1]
        self.branch_loyalty = np.clip(
            np.random.normal(archetype.branch_loyalty, 0.1),
            0.0, 1.0
        )

        # Assign home branch
        self.home_branch = np.random.choice(branch_pool)
        self.branch_pool = branch_pool

        # Regulatory client type — static from archetype, not sampled
        self.client_type = CLIENT_TYPE_MAP[archetype.name]

        # Account opening date — older for retirees, newer for students
        sim_start = datetime.date(2025, 1, 1)
        min_years, max_years = ACCOUNT_AGE_RANGES.get(archetype.name, (1, 10))
        age_days = np.random.randint(min_years * 365, max_years * 365)
        self.account_opening_date = sim_start - datetime.timedelta(days=age_days)