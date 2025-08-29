"""dataset_loader.py

This module defines the DatasetLoader class which loads raw patient record data,
applies three-stage filtering (patient selection, encounter selection, and post-cleanup),
and splits the filtered data into training, (optionally validation), and test sets.

It uses pandas and numpy for data manipulation.
"""

import os
import math
import logging
from typing import Dict, Any

import numpy as np
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class DatasetLoader:
    """
    DatasetLoader loads and preprocesses the raw patient records.
    
    It applies filtering based on patient demographics (age requirement and encounter frequency)
    and encounter criteria (valid date range and encounter types), followed by a split by patient ID.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the DatasetLoader with configuration parameters.
        
        Expected configuration keys (with default values if not provided):
            - data_filepath: Path to raw data file (default "data/raw_data.csv")
            - raw_data_format: 'csv' or 'json' (default "csv")
            - min_age: Minimum patient age (default 18)
            - max_age: Maximum patient age (default 120)
            - reference_date: Reference date for age computation (default "2012-01-01")
            - encounter_start_date_min: Minimum encounter date (default "2012-01-01")
            - encounter_start_date_max: Maximum encounter date (default "2025-04-17")
            - valid_encounter_types: List of valid encounter types (default ["outpatient", "emergency", "inpatient", "telehealth"])
            - random_seed: Seed for reproducibility (default 42)
            - use_validation: Boolean flag to create a validation split (default True)
            - validation_split_fraction: Fraction of training data to reserve for validation (default 0.1)
            - test_split_fraction: Fraction of total patients for test set (default 0.1)
            
        :param config: A configuration dictionary.
        """
        self.config = config
        self.data_filepath: str = config.get("data_filepath", "data/raw_data.csv")
        self.raw_data_format: str = config.get("raw_data_format", "csv").lower()
        self.min_age: int = config.get("min_age", 18)
        self.max_age: int = config.get("max_age", 120)
        self.reference_date: pd.Timestamp = pd.Timestamp(config.get("reference_date", "2012-01-01"))
        self.encounter_date_min: pd.Timestamp = pd.Timestamp(config.get("encounter_start_date_min", "2012-01-01"))
        self.encounter_date_max: pd.Timestamp = pd.Timestamp(config.get("encounter_start_date_max", "2025-04-17"))
        self.valid_encounter_types: list = config.get("valid_encounter_types", ["outpatient", "emergency", "inpatient", "telehealth"])
        self.random_seed: int = config.get("random_seed", 42)
        self.use_validation: bool = config.get("use_validation", True)
        self.validation_split_fraction: float = config.get("validation_split_fraction", 0.1)
        self.test_split_fraction: float = config.get("test_split_fraction", 0.1)

    def load_raw_data(self) -> pd.DataFrame:
        """
        Load raw patient encounter data from disk.
        
        The file format may be CSV or JSON (based on config).
        
        :return: Raw pandas DataFrame.
        :raises FileNotFoundError: If the data file is not found.
        """
        if not os.path.exists(self.data_filepath):
            logging.error(f"Data file '{self.data_filepath}' not found.")
            raise FileNotFoundError(f"Data file '{self.data_filepath}' not found.")

        logging.info(f"Loading raw data from {self.data_filepath} as {self.raw_data_format.upper()} format.")
        if self.raw_data_format == "csv":
            df = pd.read_csv(self.data_filepath)
        elif self.raw_data_format == "json":
            df = pd.read_json(self.data_filepath, lines=True)
        else:
            logging.error("Unsupported file format. Please use 'csv' or 'json'.")
            raise ValueError("Unsupported file format. Please use 'csv' or 'json'.")

        logging.info(f"Raw data loaded: {df.shape[0]} records.")
        return df

    def filter_patients(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply patient selection filtering.
        
        Filters include:
          - Patient age between min_age and max_age as of reference_date.
          - Patient must have at least 2 successive face-to-face encounters within a 2-year period.
        
        Assumes the DataFrame contains:
          - 'patient_id'
          - Either 'date_of_birth' (to compute age) or an 'age' column.
          - 'encounter_start_date' which will be converted to datetime.
        
        :param df: Raw DataFrame with encounter records.
        :return: DataFrame filtered by patient demographics and encounter frequency.
        """
        # Ensure date conversion for encounter_start_date
        if "encounter_start_date" in df.columns:
            df["encounter_start_date"] = pd.to_datetime(df["encounter_start_date"], errors="coerce")
        else:
            logging.error("Column 'encounter_start_date' is missing in the data.")
            raise KeyError("Column 'encounter_start_date' is required in the data.")

        # Compute patient age based on date_of_birth if available; else expect an 'age' column.
        if "date_of_birth" in df.columns:
            df["date_of_birth"] = pd.to_datetime(df["date_of_birth"], errors="coerce")
            # Compute age in years on the reference date.
            df["patient_age"] = (self.reference_date - df["date_of_birth"]).dt.days / 365.25
            df = df.dropna(subset=["patient_age"])  # Drop rows where age could not be computed
        elif "age" in df.columns:
            df["patient_age"] = df["age"]
        else:
            logging.error("No 'date_of_birth' or 'age' column found for age computation.")
            raise KeyError("Missing 'date_of_birth' or 'age' column for patient age computation.")

        # Filter by age boundaries.
        age_mask = (df["patient_age"] >= self.min_age) & (df["patient_age"] <= self.max_age)
        df = df[age_mask]
        logging.info(f"After age filtering: {df['patient_id'].nunique()} unique patients remain.")

        # Group by patient_id to check encounter frequency requirement.
        def has_two_encounters_within_two_years(group: pd.DataFrame) -> bool:
            # Sort encounters by encounter_start_date for each patient.
            group_sorted = group.sort_values("encounter_start_date")
            if group_sorted.shape[0] < 2:
                return False
            # Compute the difference between successive encounters.
            time_diffs = group_sorted["encounter_start_date"].diff().dropna()
            # Check if any two successive encounters are within 730 days (approximately 2 years).
            return (time_diffs <= pd.Timedelta(days=730)).any()

        # Apply the helper function on groups.
        grouped = df.groupby("patient_id", group_keys=False)
        valid_patients_series = grouped.apply(has_two_encounters_within_two_years)
        valid_patient_ids = valid_patients_series[valid_patients_series].index
        df_filtered = df[df["patient_id"].isin(valid_patient_ids)]
        logging.info(f"After encounter frequency filtering: {df_filtered['patient_id'].nunique()} unique patients remain.")
        return df_filtered

    def filter_encounters(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply encounter selection filtering.
        
        Conditions:
          - Encounter start date is between encounter_date_min and encounter_date_max.
          - Encounter type is within the valid_encounter_types list.
        
        Assumes the DataFrame has 'encounter_start_date' and 'encounter_type' columns.
        
        :param df: DataFrame after patient filtering.
        :return: DataFrame with encounters filtered.
        """
        # Filter by encounter start date range.
        date_mask = (df["encounter_start_date"] >= self.encounter_date_min) & (
            df["encounter_start_date"] < self.encounter_date_max
        )
        df = df[date_mask]
        logging.info(f"After date filtering: {df.shape[0]} encounters remain.")

        # Filter by valid encounter types.
        if "encounter_type" not in df.columns:
            logging.error("Column 'encounter_type' is missing in the data.")
            raise KeyError("Column 'encounter_type' is required in the data.")

        type_mask = df["encounter_type"].str.lower().isin([etype.lower() for etype in self.valid_encounter_types])
        df = df[type_mask]
        logging.info(f"After encounter type filtering: {df.shape[0]} encounters remain.")
        return df

    def post_cleanup(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform post-filter cleanup by removing patients with zero qualifying encounters.
        
        :param df: DataFrame after applying patient and encounter filters.
        :return: Cleaned DataFrame.
        """
        unique_patient_ids = df["patient_id"].unique()
        if len(unique_patient_ids) == 0:
            logging.warning("No patients remain after filtering. Check filtering criteria.")
            return df

        # In this encounter-level dataframe, every row represents an encounter.
        # We ensure that each patient has at least one encounter.
        patient_encounter_counts = df.groupby("patient_id").size()
        valid_patient_ids = patient_encounter_counts[patient_encounter_counts > 0].index
        df_clean = df[df["patient_id"].isin(valid_patient_ids)]
        logging.info(f"After post-cleanup: {df_clean['patient_id'].nunique()} patients with {df_clean.shape[0]} encounters remain.")
        return df_clean

    def split_data(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Split the data by unique patient IDs into training, (optional validation), and test sets.
        
        Splitting is performed at the patient level to ensure all encounters of a patient
        appear in only one split.
        
        :param df: Cleaned DataFrame after filtering.
        :return: Dictionary with keys "train", "test", and optionally "validation" mapping to DataFrames.
        """
        unique_patient_ids = df["patient_id"].unique()
        total_patients = len(unique_patient_ids)
        logging.info(f"Total unique patients for splitting: {total_patients}")

        # Set random seed for reproducibility and shuffle patient IDs.
        np.random.seed(self.random_seed)
        shuffled_ids = unique_patient_ids.copy()
        np.random.shuffle(shuffled_ids)

        # Determine test split size.
        num_test = int(math.floor(total_patients * self.test_split_fraction))
        test_ids = shuffled_ids[:num_test]
        remaining_ids = shuffled_ids[num_test:]

        if self.use_validation:
            num_val = int(math.floor(len(remaining_ids) * self.validation_split_fraction))
            val_ids = remaining_ids[:num_val]
            train_ids = remaining_ids[num_val:]
            logging.info(f"Split sizes - Train: {len(train_ids)}, Validation: {len(val_ids)}, Test: {len(test_ids)}")
            train_df = df[df["patient_id"].isin(train_ids)].reset_index(drop=True)
            val_df = df[df["patient_id"].isin(val_ids)].reset_index(drop=True)
            test_df = df[df["patient_id"].isin(test_ids)].reset_index(drop=True)
            return {"train": train_df, "validation": val_df, "test": test_df}
        else:
            train_ids = remaining_ids
            logging.info(f"Split sizes - Train: {len(train_ids)}, Test: {len(test_ids)}")
            train_df = df[df["patient_id"].isin(train_ids)].reset_index(drop=True)
            test_df = df[df["patient_id"].isin(test_ids)].reset_index(drop=True)
            return {"train": train_df, "test": test_df}

    def load_data(self) -> Dict[str, pd.DataFrame]:
        """
        Execute the data loading and preprocessing pipeline.
        
        Steps:
          1. Load raw data.
          2. Apply patient filtering (age and encounter frequency).
          3. Apply encounter filtering (date range and encounter type).
          4. Post-cleanup to remove patients with zero encounters.
          5. Split dataset into train/validation/test sets.
        
        :return: Dictionary with keys corresponding to data splits.
        """
        logging.info("Starting data loading and preprocessing pipeline.")
        raw_df = self.load_raw_data()

        # Apply patient selection filtering (age and encounter frequency).
        df_after_patient_filter = self.filter_patients(raw_df)

        # Apply encounter selection filtering.
        df_after_encounter_filter = self.filter_encounters(df_after_patient_filter)

        # Post-cleanup to ensure patients have at least one encounter.
        cleaned_df = self.post_cleanup(df_after_encounter_filter)

        # Split the data by unique patient IDs.
        data_splits = self.split_data(cleaned_df)
        logging.info("Data loading and splitting complete.")
        return data_splits
