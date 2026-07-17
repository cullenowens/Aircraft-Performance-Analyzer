import pandas as pd

REQUIRED_FIELDS = [
    "icao24",
    "time_position",
    "longitude",
    "latitude",
    "baro_altitude",
    "velocity",
    "true_track",
    "vertical_rate",
]

def clean(records: list[dict]) -> pd.DataFrame:
    """
    Cleans the flight track records by removing records with None values and converts into DF
    """
    if not records:
        return pd.DataFrame()
    #create df
    df = pd.DataFrame(records)
    #drop rows missing fields that are necessary
    df = df.dropna(subset=REQUIRED_FIELDS)
    #converting opensky's m/s to f/m (converted to knots in poller)
    df["baro_altitude"] = df["baro_altitude"] * 3.28084
    df["velocity"]      = df["velocity"]      * 1.94384
    df["vertical_rate"] = df["vertical_rate"] * 196.85
    # missing will be treated as airborne
    if "on_ground" in df.columns:
        df = df[df["on_ground"] == False]

    df = df.sort_values("time_position")
    df = df.drop_duplicates(subset=["time_position"], keep="first")
    df = df.reset_index(drop=True)

    return df

if __name__ == "__main__":
    # Quick manual test with fabricated records, so this can be
    # verified without needing a live API call.
    fake_records = [
        {"icao24": "abc123", "time_position": 100, "longitude": -84.5,
         "latitude": 33.9, "baro_altitude": 1000, "velocity": 120,
         "true_track": 270, "vertical_rate": 5, "on_ground": False},
        {"icao24": "abc123", "time_position": 101, "longitude": -84.51,
         "latitude": 33.91, "baro_altitude": 1050, "velocity": 122,
         "true_track": 271, "vertical_rate": 6, "on_ground": False},
        # Missing baro_altitude -> should be dropped
        {"icao24": "abc123", "time_position": 102, "longitude": -84.52,
         "latitude": 33.92, "baro_altitude": None, "velocity": 123,
         "true_track": 271, "vertical_rate": 6, "on_ground": False},
        # On the ground -> should be dropped
        {"icao24": "abc123", "time_position": 103, "longitude": -84.53,
         "latitude": 33.93, "baro_altitude": 0, "velocity": 0,
         "true_track": 0, "vertical_rate": 0, "on_ground": True},
        # Duplicate timestamp of row 1 -> should be dropped
        {"icao24": "abc123", "time_position": 100, "longitude": -84.5,
         "latitude": 33.9, "baro_altitude": 1000, "velocity": 120,
         "true_track": 270, "vertical_rate": 5, "on_ground": False},
    ]
 
    result = clean(fake_records)
    print(f"Input: {len(fake_records)} records -> Output: {len(result)} rows")
    print(result)