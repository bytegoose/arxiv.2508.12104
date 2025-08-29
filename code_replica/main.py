#!/usr/bin/env python
"""
main.py

Main entry point for reproducing the COMET experiments pipeline.

This script performs the following steps:
1. Loads configuration from "config.yaml" and sets up the environment.
2. Initializes the data pipeline by loading and pre-processing raw patient records.
3. Groups the encounter-level data by patient to create patient records.
4. Instantiates the Tokenizer with the fixed vocabulary.
5. Instantiates the Model based on the configuration (COMET-S, COMET-M, or COMET-L).
6. Creates a PyTorch Dataset from the patient records and trains the model using the Trainer.
7. Runs inference on a test sample using the InferenceEngine to generate future trajectories.
8. Evaluates the model using the Evaluation module and logs results.

All configuration parameters are read from "config.yaml". Default values are used when settings are missing.
"""

import os
import sys
import yaml
import math
import random
import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# Import our project modules.
from dataset_loader import DatasetLoader
from tokenizer import Tokenizer
from model import Model
from trainer import Trainer
from inference import InferenceEngine
from evaluation import Evaluation

# Set up logging format.
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration parameters from a YAML file.
    
    Args:
        config_path (str): Path to the YAML configuration file.
    
    Returns:
        Dict[str, Any]: Configuration dictionary.
    """
    if not os.path.exists(config_path):
        logging.error("Configuration file %s not found.", config_path)
        sys.exit(1)
    with open(config_path, "r") as yf:
        config: Dict[str, Any] = yaml.safe_load(yf)
    logging.info("Configuration loaded from %s", config_path)
    return config


def set_random_seeds(seed: int = 42) -> None:
    """
    Set random seed for reproducibility across torch, numpy, and random.
    
    Args:
        seed (int): Seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logging.info("Random seeds set to %d", seed)


def group_patient_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Groups encounter-level DataFrame by patient_id to create patient records 
    that conform to the expected format for tokenization.
    
    Each patient record is a dictionary with:
      - "demographics": dict with keys "sex", "race", "age"
      - "encounters": list of encounter dicts, each with keys:
            "encounter_type", "department", "chief_complaints", "events", "encounter_start".
    
    Args:
        df (pd.DataFrame): DataFrame with encounter-level data.
    
    Returns:
        List[Dict[str, Any]]: List of patient record dictionaries.
    """
    patient_records: List[Dict[str, Any]] = []
    # Group by patient_id
    grouped = df.groupby("patient_id")
    for patient_id, group in grouped:
        group_sorted = group.sort_values("encounter_start_date")
        # Use first row for demographics
        first_row = group_sorted.iloc[0]
        demographics: Dict[str, Any] = {
            "sex": first_row.get("sex", "Unknown_Sex"),
            "race": first_row.get("race", "White"),
            "age": first_row.get("patient_age", 50)  # default age 50 if not available
        }
        # Create list of encounters
        encounters: List[Dict[str, Any]] = []
        for _, row in group_sorted.iterrows():
            encounter: Dict[str, Any] = {
                "encounter_type": row.get("encounter_type", "Unknown"),
                "department": row.get("department", "UnknownDept"),
                # For simplicity, we assume no chief complaints and events in raw data.
                "chief_complaints": [],
                "events": [],
                "encounter_start": row.get("encounter_start_date", "")
            }
            encounters.append(encounter)
        record: Dict[str, Any] = {"demographics": demographics, "encounters": encounters}
        patient_records.append(record)
    logging.info("Grouped patient records: %d patients created.", len(patient_records))
    return patient_records


class PatientDataset(Dataset):
    """
    A PyTorch Dataset that wraps a list of patient records and uses the Tokenizer
    to convert each record into a list of token IDs.
    """
    def __init__(self, records: List[Dict[str, Any]], tokenizer: Tokenizer) -> None:
        """
        Initialize the PatientDataset.
        
        Args:
            records (List[Dict[str, Any]]): List of patient record dictionaries.
            tokenizer (Tokenizer): An instance of the Tokenizer class.
        """
        self.records: List[Dict[str, Any]] = records
        self.tokenizer: Tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> List[int]:
        record = self.records[index]
        token_sequence: List[int] = self.tokenizer.tokenize(record)
        return token_sequence


def main() -> None:
    """
    Main function to run the full COMET experimental pipeline.
    """
    # 1. Load configuration and set random seeds.
    config: Dict[str, Any] = load_config("config.yaml")
    random_seed: int = config.get("training", {}).get("random_seed", 42)
    set_random_seeds(random_seed)

    # 2. Initialize the Data Pipeline: Load and preprocess data.
    dataset_loader = DatasetLoader(config.get("dataset", {}))
    data_splits: Dict[str, pd.DataFrame] = dataset_loader.load_data()
    # Expect keys: "train", "validation" (optional), "test"
    train_df: pd.DataFrame = data_splits.get("train")
    test_df: pd.DataFrame = data_splits.get("test")
    if train_df is None or test_df is None:
        logging.error("Training and test splits are required.")
        sys.exit(1)

    # 3. Group encounter-level data into patient records.
    train_records: List[Dict[str, Any]] = group_patient_records(train_df)
    test_records: List[Dict[str, Any]] = group_patient_records(test_df)

    # 4. Initialize the Tokenizer.
    vocab_config: Dict[str, Any] = config.get("vocab", {})  # if any vocab settings exist in config.yaml
    tokenizer = Tokenizer(vocab_config)

    # 5. Create PyTorch Datasets for training.
    train_dataset = PatientDataset(train_records, tokenizer)
    # For evaluation the Trainer only requires training dataset, so validation is not used here.
    
    # 6. Initialize the Model.
    model_config: Dict[str, Any] = config.get("model", {})
    # Ensure context window is used from config.
    model_config.setdefault("context_window", 8192)
    model_config.setdefault("vocab_size", 7105)
    # Set default variant if not specified.
    model_config.setdefault("variant", "COMET-S")
    model = Model(model_config)
    logging.info("Model initialized with variant: %s", model_config.get("variant"))

    # 7. Initialize the Trainer and start training.
    trainer = Trainer(model, train_dataset, config)
    logging.info("Starting model training.")
    trainer.train()
    logging.info("Model training completed.")

    # 8. Run inference using the InferenceEngine on a test sample.
    inference_engine = InferenceEngine(model, tokenizer, config)
    # For demonstration, pick the first patient record from test_records.
    if len(test_records) > 0:
        test_prompt_record: Dict[str, Any] = test_records[0]
        prompt_tokens: List[int] = tokenizer.tokenize(test_prompt_record)
        logging.info("Running inference on a test sample (patient id: %s)", test_prompt_record.get("patient_id", "N/A"))
        generated_sequences: List[List[int]] = inference_engine.run_simulation(
            prompt=prompt_tokens,
            n=config.get("inference", {}).get("num_generations", 25),
            max_tokens=config.get("inference", {}).get("max_tokens", 2000)
        )
        aggregated_output: Dict[str, Any] = inference_engine.aggregate_predictions(generated_sequences)
        logging.info("Aggregated Inference Output:\n%s", aggregated_output)
    else:
        logging.warning("No test records available for inference.")

    # 9. Initialize the Evaluation module and compute evaluation metrics.
    # For evaluation data, we could build a custom dictionary. Here, we pass an empty dict
    # so that the Evaluation module uses its dummy sample defaults.
    evaluation_data: Dict[str, Any] = {}
    evaluator = Evaluation(model, evaluation_data, tokenizer, config)
    eval_results: Dict[str, Any] = evaluator.evaluate()
    logging.info("Evaluation Results:\n%s", eval_results)

    # 10. End of main.
    logging.info("COMET experimental pipeline completed successfully.")


if __name__ == "__main__":
    main()
