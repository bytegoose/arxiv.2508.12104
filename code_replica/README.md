# Synthetic Medical Data for CoMET Model Training

This directory contains synthetically generated medical data designed to be compatible with the CoMET (Generative Medical Event Models) training pipeline.

## Data Structure

### Main Dataset (`raw_data.csv`)
The main dataset contains patient encounter records with the following columns:

- **patient_id**: Unique patient identifier (e.g., "PATIENT_001")
- **encounter_id**: Unique encounter identifier (e.g., "ENC_001_01")
- **encounter_type**: Type of medical encounter (outpatient, inpatient, emergency, telehealth)
- **encounter_start_date**: Start date of the encounter (YYYY-MM-DD format)
- **encounter_end_date**: End date of the encounter (YYYY-MM-DD format)
- **date_of_birth**: Patient's date of birth (YYYY-MM-DD format)
- **age**: Patient's age as of the reference date (2012-01-01)
- **medical_events**: Space-separated string of medical event tokens
- **event_sequence**: List of medical event tokens (used for tokenization)

### Medical Event Types

The synthetic data includes various types of medical events:

1. **Encounter Tokens**:
   - `ENC_START`, `ENC_END`: Mark encounter boundaries
   - `OUTPATIENT`, `INPATIENT`, `EMERGENCY`, `TELEHEALTH`: Encounter types

2. **Diagnosis Codes** (ICD-10 format):
   - `DIAG_E11.9`: Type 2 diabetes without complications
   - `DIAG_I10`: Essential hypertension
   - `DIAG_Z00.00`: General adult medical examination
   - And more...

3. **Medication Codes** (NDC format):
   - `MED_0378-0045`: Metformin
   - `MED_0093-0058`: Lisinopril
   - `MED_0378-0781`: Atorvastatin
   - And more...

4. **Laboratory Test Codes** (LOINC format):
   - `LAB_33747-0`: Hemoglobin A1c
   - `LAB_2339-0`: Glucose
   - `LAB_2571-8`: Triglycerides
   - And more...

5. **Lab Values**:
   - `VAL_<number>`: Numerical values associated with lab tests

6. **Procedure Codes** (CPT format):
   - `PROC_99213`: Office visit
   - `PROC_80053`: Comprehensive metabolic panel
   - `PROC_85025`: Complete blood count
   - And more...

### Evaluation Datasets

Specialized evaluation datasets for different tasks:

- **eval_plausibility.csv**: Data for evaluating sequence plausibility
- **eval_single_encounter.csv**: Data for single encounter generation evaluation
- **eval_disease_specific.csv**: Data for disease-specific outcome prediction
- **eval_operational.csv**: Data for operational metrics evaluation

## Data Generation

The data was generated using realistic medical patterns:

- **15 patients** with varying demographics
- **3-8 encounters per patient** spanning multiple years
- **Realistic medical event sequences** following clinical workflows
- **Proper temporal ordering** of medical events
- **Age-appropriate conditions** and treatments

## Usage

This data is designed to work with the CoMET model pipeline:

1. **DatasetLoader**: Loads and filters the raw data according to configuration
2. **Tokenizer**: Converts medical event sequences into numerical tokens
3. **Model**: Processes tokenized sequences for training and generation
4. **Evaluation**: Uses specialized evaluation datasets for model assessment

## Quality Assurance

The synthetic data includes:

- ✅ Realistic medical vocabularies (ICD-10, NDC, LOINC, CPT codes)
- ✅ Proper date ranges and temporal consistency
- ✅ Age-appropriate medical conditions
- ✅ Varied encounter types and complexities
- ✅ Compatibility with all model components
- ✅ Sufficient volume for training (100+ encounter records)

## Privacy and Ethics

This is **completely synthetic data** generated algorithmically. It contains:
- ❌ No real patient information
- ❌ No protected health information (PHI)
- ❌ No identifiable data
- ✅ Purely artificial medical scenarios for research purposes

## Running Tests

To verify data compatibility, run:

```bash
python test_data_compatibility.py
```

This will test:
- Data loading with DatasetLoader
- Tokenization with Tokenizer
- Model compatibility
- Evaluation data integrity



## run the data set generation
python generate_synthetic_data.py


The synthetic data includes:

✅ 15 patients with realistic demographics
✅ Medical event sequences using standard coding systems (ICD-10, NDC, LOINC, CPT)
✅ Temporal consistency with proper date ranges
✅ Multiple encounter types (outpatient, inpatient, emergency, telehealth)
✅ Evaluation datasets for all CoMET evaluation tasks
✅ Full compatibility with the tokenizer and model implementations
The data follows the exact structure expected by the DatasetLoader and includes all necessary columns for the filtering pipeline. The medical events are realistic and follow clinical workflows, making this suitable for training the CoMET model.