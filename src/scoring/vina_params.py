"""AutoDock Vina weight parameters for different scoring functions."""

VINA_WEIGHTS = {
    "vina": {
        "gauss1": -0.035579,
        "gauss2": -0.005156,
        "repulsion": 0.840245,
        "hydrophobic": -0.035069,
        "hbond": -0.587439,
        "rot": 0.05846,
    },
    "vinardo": {
        "gauss1": -0.0356,
        "gauss2": 0.0,
        "repulsion": 0.840,
        "hydrophobic": -0.0351,
        "hbond": -0.587,
        "rot": 0.05846,
    },
}
