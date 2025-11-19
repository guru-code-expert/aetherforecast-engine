"""Central configuration and constants."""

# Dummy / anonymized categorical mappings (for reference only)
CARRIER_MAPPING = {
    0.0: "CarrierA", 1.0: "CarrierB", 2.0: "CarrierC", 3.0: "CarrierD",
    4.0: "CarrierE", 5.0: "CarrierF", 6.0: "CarrierG", 7.0: "CarrierH",
    8.0: "CarrierI", 9.0: "CarrierJ", 10.0: "CarrierK", 11.0: "CarrierL"
}

ORIGIN_MAPPING = {0.0: "HubAlpha", 1.0: "HubBeta", 2.0: "HubGamma", 3.0: "HubDelta", 4.0: "HubEpsilon"}
DESTINATION_MAPPING = {0.0: "HubAlpha", 1.0: "PortPrime", 2.0: "PortSecond", 3.0: "PortThird", 4.0: "PortFourth", 5.0: "PortFifth"}

DATA_PATH = "data/raw/dataset.csv"
MODEL_PATH = "models/random_forest_regressor.joblib"
REFERENCE_DATE = "2018-01-01"