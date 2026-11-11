import re
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
ARCHIVE_DIR = BASE_DIR / "archive"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
REPORTS_DIR = BASE_DIR / "reports"

DATASETS = {
    "10K": ARCHIVE_DIR / "apartments_for_rent_classified_10K" / "apartments_for_rent_classified_10K.csv",
    "100K": ARCHIVE_DIR / "apartments_for_rent_classified_100K" / "apartments_for_rent_classified_100K.csv",
}

FALLBACK_STATS = {
    "10K": {"initial_rows": 10000, "duplicates_removed": 0, "removed_rows": 26},
    "100K": {"initial_rows": 99492, "duplicates_removed": 84, "removed_rows": 214},
}

AMENITY_LABELS = [
    "Parking",
    "Pool",
    "Gym",
    "Patio/Deck",
    "Washer Dryer",
    "Storage",
    "Clubhouse",
    "Dishwasher",
    "AC",
    "Refrigerator",
    "Fireplace",
    "Cable or Satellite",
    "Playground",
    "Internet Access",
    "Wood Floors",
    "Gated",
    "Tennis",
    "TV",
    "Elevator",
    "Basketball",
    "Hot Tub",
    "Garbage Disposal",
    "View",
    "Alarm",
    "Doorman",
    "Luxury",
    "Golf",
]

DROP_COLUMNS = [
    "title",
    "body",
    "price_display",
    "address",
    "category",
    "currency",
    "fee",
    "price_type",
]

PRICE_MIN = 200
PRICE_MAX = 15000
SQUARE_FEET_MAX = 10000
PRICE_PER_SQFT_MIN = 0.3
PRICE_PER_SQFT_MAX = 10.0
BEDROOMS_MAX = 6
BATHROOMS_MAX = 6


def slugify(text):
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def load_dataset(path):
    return pd.read_csv(path, sep=";", encoding="latin-1", low_memory=False)


def remove_outliers(df):
    df = df[df["price"].notna() & (df["price"] > 0)]
    df = df[df["square_feet"].notna() & (df["square_feet"] > 0)]
    df = df[df["square_feet"] <= SQUARE_FEET_MAX]
    df = df[df["price"].between(PRICE_MIN, PRICE_MAX)]
    df["price_per_sqft"] = df["price"] / df["square_feet"]
    df = df[df["price_per_sqft"].between(PRICE_PER_SQFT_MIN, PRICE_PER_SQFT_MAX)]
    df = df.drop(columns=["price_per_sqft"])
    df = df[df["bedrooms"].isna() | df["bedrooms"].between(0, BEDROOMS_MAX)]
    df = df[df["bathrooms"].isna() | df["bathrooms"].between(0, BATHROOMS_MAX)]
    return df


def impute_missing_values(df):
    df["state"] = df["state"].fillna("Unknown")
    df["cityname"] = df["cityname"].fillna("Unknown")
    pets = df["pets_allowed"].fillna("No pets")
    pets = pets.str.replace("None", "", regex=False).str.strip(", ")
    df["pets_allowed"] = pets.mask(pets.eq(""), "No pets")
    df["has_photo"] = df["has_photo"].fillna("No")

    df["bedrooms"] = df["bedrooms"].fillna(df["bedrooms"].median())
    bathrooms_by_bedrooms = df.groupby("bedrooms")["bathrooms"].transform("median")
    df["bathrooms"] = df["bathrooms"].fillna(bathrooms_by_bedrooms)
    df["bathrooms"] = df["bathrooms"].fillna(df["bathrooms"].median())

    for column in ["latitude", "longitude"]:
        by_state = df.groupby("state")[column].transform("median")
        df[column] = df[column].fillna(by_state)
        df[column] = df[column].fillna(df[column].median())

    return df


def add_amenities_features(df):
    provided = df["amenities"].notna()
    tokens = df["amenities"].fillna("").map(
        lambda value: {item.strip().lower() for item in str(value).split(",") if item.strip()}
    )

    df["amenities_provided"] = provided.astype(int)
    df["amenities_count"] = tokens.map(len).astype(int)

    for label in AMENITY_LABELS:
        df[f"amenity_{slugify(label)}"] = tokens.map(
            lambda values, target=label.lower(): int(target in values)
        )

    df = df.drop(columns=["amenities"])
    return df


def add_listing_date_features(df):
    dates = pd.to_datetime(df["time"], unit="s", errors="coerce")
    df["listing_date"] = dates.dt.strftime("%Y-%m-%d")
    df["listing_year"] = dates.dt.year
    df["listing_month"] = dates.dt.month
    df["listing_dayofweek"] = dates.dt.dayofweek
    df = df.drop(columns=["time"])
    return df


def add_derived_features(df):
    pets = df["pets_allowed"].str.lower()
    df["has_cats"] = pets.str.contains("cats", regex=False).astype(int)
    df["has_dogs"] = pets.str.contains("dogs", regex=False).astype(int)
    df["is_studio"] = (df["bedrooms"] == 0).astype(int)
    df["total_rooms"] = df["bedrooms"] + df["bathrooms"]
    return df


def add_frequency_encodings(df):
    city_frequency = df["cityname"].value_counts(normalize=True)
    source_frequency = df["source"].value_counts(normalize=True)
    df["city_freq"] = df["cityname"].map(city_frequency).astype(float)
    df["source_freq"] = df["source"].map(source_frequency).astype(float)
    return df


def clean_dataset(df):
    initial_rows = len(df)
    df = df.copy()
    df.columns = [column.strip().lower() for column in df.columns]

    df = df.drop_duplicates()
    df = df.drop_duplicates(subset="id", keep="first")

    df = df[df["category"].astype(str).str.contains("apartment", na=False)]
    df = df[df["price_type"] == "Monthly"]
    df = remove_outliers(df)

    df = impute_missing_values(df)
    df = add_amenities_features(df)
    df = add_listing_date_features(df)
    df = add_derived_features(df)
    df = add_frequency_encodings(df)

    df = df.drop(columns=[column for column in DROP_COLUMNS if column in df.columns])
    df = df.reset_index(drop=True)

    stats = {
        "initial_rows": initial_rows,
        "final_rows": len(df),
        "removed_rows": initial_rows - len(df),
        "duplicates_removed": 0,
        "missing_values": int(df.isna().sum().sum()),
    }
    return df, stats


def build_feature_matrix(df):
    target = df["price"].astype(float)

    excluded = {
        "id",
        "price",
        "listing_date",
        "category",
        "currency",
        "fee",
        "price_type",
        "cityname",
        "state",
        "pets_allowed",
        "has_photo",
        "source",
    }
    features = df.drop(columns=[column for column in excluded if column in df.columns])

    state_dummies = pd.get_dummies(df["state"], prefix="state", drop_first=True, dtype=int)
    photo_dummies = pd.get_dummies(df["has_photo"], prefix="photo", dtype=int)

    features = pd.concat([features, state_dummies, photo_dummies], axis=1)
    features = features.astype(float)
    features = features.loc[:, features.nunique() > 1]

    return features, target


def standardize_numeric_features(features):
    binary_columns = [
        column
        for column in features.columns
        if features[column].dropna().isin([0, 1]).all()
    ]
    numeric_columns = [column for column in features.columns if column not in binary_columns]

    scaled = features.copy()
    for column in numeric_columns:
        mean = scaled[column].mean()
        std = scaled[column].std(ddof=0)
        if std and not np.isnan(std):
            scaled[column] = (scaled[column] - mean) / std
        else:
            scaled[column] = 0.0
    return scaled


def save_dataset(tag, features):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    features.to_csv(
        PROCESSED_DIR / f"X_{tag}.csv.gz",
        index=False,
        float_format="%.4f",
        compression="gzip",
    )

    scaled = standardize_numeric_features(features)
    scaled.to_csv(
        PROCESSED_DIR / f"X_{tag}_scaled.csv.gz",
        index=False,
        float_format="%.4f",
        compression="gzip",
    )


def write_report(report_lines):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "preprocessing_summary.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    return report_path


def main():
    report_lines = ["Rapport de prétraitement - Prédiction du prix de location", ""]
    generated_files = []

    for tag, path in DATASETS.items():
        if path.exists():
            raw = load_dataset(path)
            cleaned, stats = clean_dataset(raw)
            stats["duplicates_removed"] = stats["initial_rows"] - len(raw.drop_duplicates())
            source_label = "fichier brut"
        else:
            clean_path = PROCESSED_DIR / f"apartments_clean_{tag}.csv"
            if not clean_path.exists():
                report_lines.append(f"[{tag}] source introuvable : {path} et {clean_path}")
                continue
            cleaned = pd.read_csv(clean_path)
            fallback = FALLBACK_STATS.get(tag, {})
            stats = {
                "initial_rows": fallback.get("initial_rows"),
                "final_rows": len(cleaned),
                "removed_rows": fallback.get("removed_rows"),
                "duplicates_removed": fallback.get("duplicates_removed"),
                "missing_values": int(cleaned.isna().sum().sum()),
            }
            source_label = "données nettoyées existantes"

        features, _ = build_feature_matrix(cleaned)
        save_dataset(tag, features)
        generated_files.extend([f"X_{tag}.csv.gz", f"X_{tag}_scaled.csv.gz"])

        report_lines.extend(
            [
                f"Dataset {tag}",
                f"  source                  : {source_label}",
                f"  lignes initiales        : {stats['initial_rows'] if stats['initial_rows'] is not None else 'non disponible'}",
                f"  doublons supprimés      : {stats['duplicates_removed'] if stats['duplicates_removed'] is not None else 'non disponible'}",
                f"  lignes après nettoyage  : {stats['final_rows']}",
                f"  lignes supprimées       : {stats['removed_rows'] if stats['removed_rows'] is not None else 'non disponible'}",
                f"  valeurs manquantes      : {stats['missing_values']}",
                f"  variables explicatives  : {features.shape[1]}",
                f"  cible                   : price",
                "",
            ]
        )

    report_lines.append("Fichiers générés dans data/processed :")
    for name in generated_files:
        report_lines.append(f"  - {name}")

    report_path = write_report(report_lines)
    print("\n".join(report_lines))
    print(f"\nRapport : {report_path}")


if __name__ == "__main__":
    main()
