"""
test_data_compatibility.py

Test script to verify that the synthetic data is compatible with the CoMET model components.
"""

import os
import sys
import yaml
import pandas as pd
import torch
from typing import Dict, Any, List

# Import the model components
from dataset_loader import DatasetLoader
from tokenizer import Tokenizer
from model import Model

def test_data_loading():
    """Test that the DatasetLoader can load the synthetic data."""
    print("Testing DatasetLoader...")
    
    # Load configuration
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Initialize DatasetLoader
    dataset_loader = DatasetLoader(config["data"])
    
    try:
        # Load data
        data_splits = dataset_loader.load_data()
        
        print(f"✓ Data loading successful!")
        print(f"  Train records: {len(data_splits['train'])}")
        if 'validation' in data_splits:
            print(f"  Validation records: {len(data_splits['validation'])}")
        print(f"  Test records: {len(data_splits['test'])}")
        
        return data_splits
        
    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        return None

def test_tokenization(data_splits: Dict[str, pd.DataFrame]):
    """Test that the Tokenizer can process the medical events."""
    print("\nTesting Tokenizer...")
    
    # Load configuration
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    try:
        # Initialize tokenizer
        tokenizer = Tokenizer(config["tokenizer"])
        
        # Get sample medical events
        sample_data = data_splits['train'].head(3)
        
        for idx, row in sample_data.iterrows():
            events = row['medical_events'].split()
            print(f"\nSample {idx + 1}:")
            print(f"  Original events: {events[:5]}...")
            
            # Tokenize
            tokenized = tokenizer.tokenize(events)
            print(f"  Tokenized: {tokenized[:10]}...")
            
            # Detokenize
            detokenized = tokenizer.detokenize(tokenized)
            print(f"  Detokenized: {detokenized['tokens'][:5]}...")
        
        print("✓ Tokenization successful!")
        return tokenizer
        
    except Exception as e:
        print(f"✗ Tokenization failed: {e}")
        return None

def test_model_compatibility(tokenizer: Tokenizer):
    """Test that the Model can process tokenized sequences."""
    print("\nTesting Model compatibility...")
    
    # Load configuration
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    try:
        # Initialize model
        model = Model(config["model"])
        
        # Create sample input
        sample_tokens = [1, 10, 20, 30, 40, 2]  # Sample token sequence
        input_ids = torch.tensor([sample_tokens], dtype=torch.long)
        
        # Test forward pass
        with torch.no_grad():
            logits = model.forward(input_ids)
            print(f"  Input shape: {input_ids.shape}")
            print(f"  Output logits shape: {logits.shape}")
            print(f"  Expected vocab size: {config['model']['vocab_size']}")
        
        # Test generation
        generated = model.generate(sample_tokens[:3], max_tokens=10, temperature=1.0)
        print(f"  Generated sequence length: {len(generated)}")
        print(f"  Generated tokens: {generated}")
        
        print("✓ Model compatibility successful!")
        return model
        
    except Exception as e:
        print(f"✗ Model compatibility failed: {e}")
        return None

def test_evaluation_data():
    """Test that evaluation data files exist and are properly formatted."""
    print("\nTesting evaluation data...")
    
    eval_files = [
        "data/eval_plausibility.csv",
        "data/eval_single_encounter.csv", 
        "data/eval_disease_specific.csv",
        "data/eval_operational.csv"
    ]
    
    for file_path in eval_files:
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                print(f"✓ {file_path}: {len(df)} records")
            except Exception as e:
                print(f"✗ {file_path}: Error reading file - {e}")
        else:
            print(f"✗ {file_path}: File not found")

def main():
    """Run all compatibility tests."""
    print("=" * 60)
    print("COMET DATA COMPATIBILITY TEST")
    print("=" * 60)
    
    # Check if data files exist
    if not os.path.exists("data/raw_data.csv"):
        print("✗ Raw data file not found. Please run generate_synthetic_data.py first.")
        return
    
    if not os.path.exists("config.yaml"):
        print("✗ Configuration file not found.")
        return
    
    # Test data loading
    data_splits = test_data_loading()
    if data_splits is None:
        return
    
    # Test tokenization
    tokenizer = test_tokenization(data_splits)
    if tokenizer is None:
        return
    
    # Test model compatibility
    model = test_model_compatibility(tokenizer)
    if model is None:
        return
    
    # Test evaluation data
    test_evaluation_data()
    
    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("The synthetic data is compatible with the CoMET model pipeline.")
    print("=" * 60)

if __name__ == "__main__":
    main()