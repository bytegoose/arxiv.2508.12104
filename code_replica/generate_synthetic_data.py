"""
generate_synthetic_data.py

This script generates synthetic medical data compatible with the CoMET model training pipeline.
It creates realistic patient encounter data with medical events, diagnoses, medications, and lab results.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from typing import List, Dict, Any

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

def generate_synthetic_medical_data(num_patients: int = 10) -> pd.DataFrame:
    """
    Generate synthetic medical data for CoMET model training.
    
    Args:
        num_patients (int): Number of patients to generate
        
    Returns:
        pd.DataFrame: Synthetic medical data with required columns
    """
    
    # Medical vocabularies for realistic data generation
    encounter_types = ['outpatient', 'inpatient', 'emergency', 'telehealth']
    
    # ICD-10 diagnosis codes (simplified)
    diagnosis_codes = [
        'E11.9',   # Type 2 diabetes without complications
        'I10',     # Essential hypertension
        'Z00.00',  # Encounter for general adult medical examination
        'M79.18',  # Myalgia, other site
        'R06.02',  # Shortness of breath
        'K21.9',   # Gastro-esophageal reflux disease
        'F32.9',   # Major depressive disorder, single episode
        'M25.50',  # Pain in joint
        'R50.9',   # Fever
        'J06.9',   # Acute upper respiratory infection
        'N39.0',   # Urinary tract infection
        'K59.00',  # Constipation
        'H52.4',   # Presbyopia
        'L70.9',   # Acne
        'B34.9'    # Viral infection
    ]
    
    # Medication codes (NDC format simplified)
    medication_codes = [
        'MED_0378-0045',  # Metformin
        'MED_0093-0058',  # Lisinopril
        'MED_0378-0781',  # Atorvastatin
        'MED_0093-0150',  # Amlodipine
        'MED_0378-6205',  # Omeprazole
        'MED_0093-0127',  # Sertraline
        'MED_0378-0511',  # Ibuprofen
        'MED_0093-0020',  # Acetaminophen
        'MED_0378-0229',  # Albuterol
        'MED_0093-0075'   # Amoxicillin
    ]
    
    # Lab test codes (LOINC simplified)
    lab_codes = [
        'LAB_33747-0',  # Hemoglobin A1c
        'LAB_2339-0',   # Glucose
        'LAB_2571-8',   # Triglycerides
        'LAB_2093-3',   # Cholesterol
        'LAB_33765-2',  # Creatinine
        'LAB_6298-4',   # Potassium
        'LAB_2951-2',   # Sodium
        'LAB_718-7',    # Hemoglobin
        'LAB_4544-3',   # Hematocrit
        'LAB_26464-8'   # Leukocytes
    ]
    
    # Procedure codes (CPT simplified)
    procedure_codes = [
        'PROC_99213',   # Office visit
        'PROC_99214',   # Office visit detailed
        'PROC_80053',   # Comprehensive metabolic panel
        'PROC_85025',   # Complete blood count
        'PROC_93000',   # Electrocardiogram
        'PROC_71020',   # Chest X-ray
        'PROC_36415',   # Routine venipuncture
        'PROC_90834',   # Psychotherapy
        'PROC_99396',   # Preventive visit
        'PROC_12001'    # Simple wound repair
    ]
    
    records = []
    
    for patient_id in range(1, num_patients + 1):
        # Generate patient demographics
        birth_year = random.randint(1940, 2000)
        birth_date = datetime(birth_year, random.randint(1, 12), random.randint(1, 28))
        
        # Calculate age as of 2012 (reference date)
        reference_date = datetime(2012, 1, 1)
        age = (reference_date - birth_date).days / 365.25
        
        # Generate 3-8 encounters per patient
        num_encounters = random.randint(3, 8)
        
        # Start date for encounters (between 2012-2024)
        start_date = datetime(2012, 1, 1)
        current_date = start_date + timedelta(days=random.randint(0, 365))
        
        for encounter_id in range(1, num_encounters + 1):
            encounter_type = random.choice(encounter_types)
            encounter_start = current_date
            
            # Encounter duration (outpatient: same day, inpatient: 1-5 days)
            if encounter_type == 'inpatient':
                duration = random.randint(1, 5)
            else:
                duration = 0
            encounter_end = encounter_start + timedelta(days=duration)
            
            # Generate medical events for this encounter
            events = []
            
            # Always include encounter start/end tokens
            events.extend(['ENC_START', encounter_type.upper()])
            
            # Add diagnoses (1-3 per encounter)
            num_diagnoses = random.randint(1, 3)
            for _ in range(num_diagnoses):
                diag_code = random.choice(diagnosis_codes)
                events.append(f'DIAG_{diag_code}')
            
            # Add medications (0-2 per encounter)
            if random.random() > 0.3:  # 70% chance of medications
                num_meds = random.randint(1, 2)
                for _ in range(num_meds):
                    med_code = random.choice(medication_codes)
                    events.append(med_code)
            
            # Add lab tests (0-3 per encounter)
            if random.random() > 0.4:  # 60% chance of lab tests
                num_labs = random.randint(1, 3)
                for _ in range(num_labs):
                    lab_code = random.choice(lab_codes)
                    # Add lab value as a separate token
                    events.append(lab_code)
                    if 'Glucose' in lab_code:
                        value = random.randint(80, 200)
                    elif 'A1c' in lab_code:
                        value = round(random.uniform(5.0, 10.0), 1)
                    else:
                        value = random.randint(10, 100)
                    events.append(f'VAL_{value}')
            
            # Add procedures (0-2 per encounter)
            if random.random() > 0.5:  # 50% chance of procedures
                num_procs = random.randint(1, 2)
                for _ in range(num_procs):
                    proc_code = random.choice(procedure_codes)
                    events.append(proc_code)
            
            events.append('ENC_END')
            
            # Create record
            record = {
                'patient_id': f'PATIENT_{patient_id:03d}',
                'encounter_id': f'ENC_{patient_id:03d}_{encounter_id:02d}',
                'encounter_type': encounter_type,
                'encounter_start_date': encounter_start.strftime('%Y-%m-%d'),
                'encounter_end_date': encounter_end.strftime('%Y-%m-%d'),
                'date_of_birth': birth_date.strftime('%Y-%m-%d'),
                'age': int(age),
                'medical_events': ' '.join(events),
                'event_sequence': events  # This will be used for tokenization
            }
            
            records.append(record)
            
            # Next encounter 30-180 days later
            current_date += timedelta(days=random.randint(30, 180))
    
    return pd.DataFrame(records)

def create_evaluation_data(df: pd.DataFrame) -> Dict[str, List[Dict]]:
    """
    Create evaluation data subsets from the main dataset.
    
    Args:
        df (pd.DataFrame): Main dataset
        
    Returns:
        Dict[str, List[Dict]]: Evaluation data for different tasks
    """
    evaluation_data = {
        'plausibility': [],
        'single_encounter': [],
        'disease_specific': [],
        'operational': []
    }
    
    # Sample some patients for evaluation tasks
    sample_patients = df['patient_id'].unique()[:5]
    
    for patient_id in sample_patients:
        patient_data = df[df['patient_id'] == patient_id]
        
        # Plausibility evaluation - use first encounter as prompt
        first_encounter = patient_data.iloc[0]
        prompt_events = first_encounter['event_sequence'][:3]  # First 3 events as prompt
        evaluation_data['plausibility'].append({
            'prompt': prompt_events,
            'patient_id': patient_id
        })
        
        # Single encounter evaluation
        if len(patient_data) > 1:
            encounter = patient_data.iloc[1]
            prompt_events = encounter['event_sequence'][:2]  # Encounter start as prompt
            ground_truth_events = encounter['event_sequence'][2:-1]  # Exclude start/end tokens
            evaluation_data['single_encounter'].append({
                'prompt': prompt_events,
                'ground_truth_events': ground_truth_events,
                'patient_id': patient_id
            })
        
        # Disease-specific evaluation (diabetes prediction)
        has_diabetes = any('E11.9' in event for event in first_encounter['event_sequence'])
        evaluation_data['disease_specific'].append({
            'prompt': prompt_events,
            'target_token': 'DIAG_E11.9',  # Diabetes diagnosis
            'label': 1 if has_diabetes else 0,
            'patient_id': patient_id
        })
        
        # Operational evaluation
        encounter_count = len(patient_data)
        los_days = (pd.to_datetime(patient_data.iloc[-1]['encounter_end_date']) - 
                   pd.to_datetime(patient_data.iloc[0]['encounter_start_date'])).days
        
        evaluation_data['operational'].extend([
            {
                'type': 'encounter_forecast',
                'prompt': prompt_events,
                'ground_truth': encounter_count,
                'patient_id': patient_id
            },
            {
                'type': 'los',
                'prompt': prompt_events,
                'ground_truth': float(los_days),
                'patient_id': patient_id
            },
            {
                'type': 'readmission',
                'prompt': prompt_events,
                'ground_truth': 1 if encounter_count > 2 else 0,
                'patient_id': patient_id
            }
        ])
    
    return evaluation_data

def main():
    """Generate and save synthetic medical data."""
    
    # Create data directory if it doesn't exist
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    
    print("Generating synthetic medical data...")
    
    # Generate synthetic data for 15 patients (more than minimum 10)
    df = generate_synthetic_medical_data(num_patients=15)
    
    # Save main dataset
    main_data_path = os.path.join(data_dir, "raw_data.csv")
    df.to_csv(main_data_path, index=False)
    print(f"Saved main dataset with {len(df)} records to {main_data_path}")
    
    # Create evaluation data
    eval_data = create_evaluation_data(df)
    
    # Save evaluation data as separate files
    for task_name, task_data in eval_data.items():
        eval_path = os.path.join(data_dir, f"eval_{task_name}.csv")
        if task_data:  # Only save if data exists
            eval_df = pd.DataFrame(task_data)
            eval_df.to_csv(eval_path, index=False)
            print(f"Saved {task_name} evaluation data with {len(eval_df)} records to {eval_path}")
    
    # Display sample data
    print("\n" + "="*60)
    print("SAMPLE DATA PREVIEW")
    print("="*60)
    
    print("\nDataset shape:", df.shape)
    print("\nColumn names:", list(df.columns))
    
    print("\nSample records:")
    for i, row in df.head(3).iterrows():
        print(f"\nRecord {i+1}:")
        print(f"  Patient ID: {row['patient_id']}")
        print(f"  Encounter: {row['encounter_type']} on {row['encounter_start_date']}")
        print(f"  Age: {row['age']}")
        print(f"  Medical Events: {row['medical_events'][:100]}...")
        print(f"  Event Sequence Length: {len(row['event_sequence'])}")
    
    print("\n" + "="*60)
    print("DATA STATISTICS")
    print("="*60)
    
    print(f"Total patients: {df['patient_id'].nunique()}")
    print(f"Total encounters: {len(df)}")
    print(f"Encounter types: {df['encounter_type'].value_counts().to_dict()}")
    print(f"Age range: {df['age'].min():.1f} - {df['age'].max():.1f} years")
    print(f"Date range: {df['encounter_start_date'].min()} to {df['encounter_start_date'].max()}")
    
    # Analyze event types
    all_events = []
    for events in df['event_sequence']:
        all_events.extend(events)
    
    unique_events = set(all_events)
    print(f"Unique event types: {len(unique_events)}")
    
    # Count event categories
    event_categories = {}
    for event in unique_events:
        if event.startswith('DIAG_'):
            category = 'Diagnosis'
        elif event.startswith('MED_'):
            category = 'Medication'
        elif event.startswith('LAB_'):
            category = 'Lab Test'
        elif event.startswith('PROC_'):
            category = 'Procedure'
        elif event.startswith('VAL_'):
            category = 'Lab Value'
        elif event.startswith('ENC_'):
            category = 'Encounter'
        else:
            category = 'Other'
        
        event_categories[category] = event_categories.get(category, 0) + 1
    
    print("\nEvent categories:")
    for category, count in event_categories.items():
        print(f"  {category}: {count}")
    
    print(f"\nData successfully generated and saved to '{data_dir}' directory!")
    print("Files created:")
    print("  - raw_data.csv (main dataset)")
    for task_name in eval_data.keys():
        if eval_data[task_name]:
            print(f"  - eval_{task_name}.csv (evaluation data)")

if __name__ == "__main__":
    main()