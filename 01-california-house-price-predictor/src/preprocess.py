import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering pipeline for California housing data.
    Applies log transforms, ratio features, and geographic clustering.
    """
    df = df.copy()

    # log-transform skewed features
    df["log_MedInc"] = np.log1p(df["MedInc"])
    df["log_Population"] = np.log1p(df["Population"])

    # ratio features
    df["rooms_per_person"] = df["AveRooms"] / (df["AveOccup"] + 1e-6)
    df["bedrooms_per_room"] = df["AveBedrms"] / (df["AveRooms"] + 1e-6)

    # geographic cluster (nearest city proxy)
    coords = df[["Latitude", "Longitude"]].values
    km = KMeans(n_clusters=20, random_state=42, n_init=10)
    df["geo_cluster"] = km.fit_predict(coords)

    # interaction
    df["income_rooms"] = df["log_MedInc"] * df["rooms_per_person"]

    return df


def get_feature_columns():
    return [
        "log_MedInc", "HouseAge", "AveRooms", "AveBedrms",
        "log_Population", "AveOccup", "Latitude", "Longitude",
        "rooms_per_person", "bedrooms_per_room", "geo_cluster",
        "income_rooms"
    ]
