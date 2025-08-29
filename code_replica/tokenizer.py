"""tokenizer.py

This module defines the Tokenizer class that implements methods for tokenizing
and detokenizing structured patient records into fixed sequences of token IDs.
The tokenization follows the specifications from the COMET paper:
- Demographics are converted into tokens (e.g., sex, race, age buckets).
- Encounters are delimited by Encounter_Start and Encounter_End tokens,
  and include tokens for encounter type, department specialty, chief complaints,
  and individual events (diagnoses, labs, medications, procedures, time tokens).
- Diagnoses, labs, medications and procedures are split into one or more tokens,
  according to pre-defined rules.
- Lab numeric values are bucketed into quantile tokens using precomputed thresholds.
- Time gaps between events are converted into tokens based on pre-specified time intervals.
  
The detokenize method reconstructs a human‐readable representation (for debugging
and evaluation purposes) from a list of token IDs.
  
All vocabulary mappings (token-to-id and id-to-token), lab quantile boundaries, and
time bucket definitions are provided via a vocab_config dictionary. Default values
are set if configuration values are not provided.
  
References: COMET paper, project design and experimental plan.
"""

import bisect
import logging
from math import floor
from typing import Any, Dict, List

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class Tokenizer:
    """
    The Tokenizer class transforms a structured patient record into a list of token IDs,
    and can detokenize a list of token IDs back into a human-readable representation.
    
    Attributes:
        token2id (Dict[str, int]): Mapping from token strings to unique integer IDs.
        id2token (Dict[int, str]): Reverse mapping from integer ID to token strings.
        lab_quantile_boundaries (List[float]): Sorted boundaries used for lab value bucketing.
        time_buckets (List[tuple]]): List of tuples defining time interval buckets.
                                     Each tuple is (lower_bound, upper_bound, token_name),
                                     with bounds in minutes.
    """

    def __init__(self, vocab_config: Dict[str, Any]) -> None:
        """
        Initialize the Tokenizer with vocabulary and configuration settings.
        
        Args:
            vocab_config (Dict[str, Any]): A configuration dictionary containing:
                - "token2id": Dict[str, int] mapping tokens to IDs.
                - "id2token": Dict[int, str] mapping IDs back to tokens.
                - "lab_quantile_boundaries": List[float] of nine boundary values for 10 buckets.
                  (Default: [10, 20, 30, 40, 50, 60, 70, 80, 90])
                - "time_buckets": List of tuples (lower_bound, upper_bound, token_name) for time gaps.
                  (Default provided below)
        """
        # Load vocabulary mappings
        self.token2id: Dict[str, int] = vocab_config.get("token2id", {
            "UNK": 0,
            "BEGINNING_OF_SEQUENCE": 1,
            "Encounter_Start": 2,
            "Encounter_End": 3,
            # Demographics tokens
            "Male": 10,
            "Female": 11,
            "Unknown_Sex": 12,
            "Race_White": 20,
            "Race_Black": 21,
            "Race_Asian": 22,
            # Age buckets (example buckets; real system would have ~5-year bins from 18 to 120)
            "Age_18_22": 30,
            "Age_23_27": 31,
            "Age_28_32": 32,
            "Age_33_37": 33,
            "Age_38_42": 34,
            "Age_43_47": 35,
            "Age_48_52": 36,
            "Age_53_57": 37,
            "Age_58_62": 38,
            "Age_63_67": 39,
            # Lab quantile tokens
            "LAB_QUANTILE_1": 40,
            "LAB_QUANTILE_2": 41,
            "LAB_QUANTILE_3": 42,
            "LAB_QUANTILE_4": 43,
            "LAB_QUANTILE_5": 44,
            "LAB_QUANTILE_6": 45,
            "LAB_QUANTILE_7": 46,
            "LAB_QUANTILE_8": 47,
            "LAB_QUANTILE_9": 48,
            "LAB_QUANTILE_10": 49,
            # Time tokens (example set; times in minutes)
            "TIME_1_5_MIN": 50,
            "TIME_5_15_MIN": 51,
            "TIME_15_30_MIN": 52,
            "TIME_30_60_MIN": 53,
            "TIME_1_2_HR": 54,
            "TIME_2_4_HR": 55,
            "TIME_4_8_HR": 56,
            "TIME_8_24_HR": 57,
            "TIME_1_3_DAY": 58,
            "TIME_3_7_DAY": 59,
            # Procedure unknown token
            "UNKNOWN_PROCEDURE": 100,
            # Diagnosis and Medication tokens will be constructed dynamically with prefixes.
        })
        self.id2token: Dict[int, str] = vocab_config.get("id2token", {v: k for k, v in self.token2id.items()})

        # Set lab quantile boundaries; default: nine boundaries for 10 buckets.
        self.lab_quantile_boundaries: List[float] = vocab_config.get("lab_quantile_boundaries", [10, 20, 30, 40, 50, 60, 70, 80, 90])
        
        # Set time buckets for elapsed time tokens.
        # Each bucket is a tuple: (lower_bound_in_minutes, upper_bound_in_minutes, token_name)
        self.time_buckets: List[tuple] = vocab_config.get("time_buckets", [
            (1, 5, "TIME_1_5_MIN"),
            (5, 15, "TIME_5_15_MIN"),
            (15, 30, "TIME_15_30_MIN"),
            (30, 60, "TIME_30_60_MIN"),
            (60, 120, "TIME_1_2_HR"),
            (120, 240, "TIME_2_4_HR"),
            (240, 480, "TIME_4_8_HR"),
            (480, 1440, "TIME_8_24_HR"),
            (1440, 4320, "TIME_1_3_DAY"),
            (4320, 10080, "TIME_3_7_DAY")
        ])
        logging.info("Tokenizer initialized with vocabulary size %d.", len(self.token2id))

    def tokenize(self, record: Dict[str, Any]) -> List[int]:
        """
        Tokenize a full patient record into a list of token IDs.
        
        The record should include:
            - "demographics": dict containing demographic fields.
            - "encounters": list of encounter dicts (each with events).
        
        Returns:
            List[int]: A list of token IDs representing the patient record.
        """
        tokens: List[int] = []
        # Tokenize demographics
        demographics = record.get("demographics", {})
        demo_tokens = self._tokenize_demographics(demographics)
        tokens.extend(demo_tokens)
        # Append special beginning-of-sequence marker
        tokens.append(self.token2id.get("BEGINNING_OF_SEQUENCE", self.token2id.get("UNK")))
        # Process encounters in chronological order
        encounters = record.get("encounters", [])
        # Assume each encounter dict has an "encounter_start" key
        encounters_sorted = sorted(encounters, key=lambda x: x.get("encounter_start", ""))
        for encounter in encounters_sorted:
            encounter_tokens = self._tokenize_encounter(encounter)
            tokens.extend(encounter_tokens)
        return tokens

    def detokenize(self, tokens: List[int]) -> Dict[str, Any]:
        """
        Detokenize a list of token IDs back into a human-readable representation.
        
        For simplicity, this method returns a dictionary containing the
        sequence of token strings and a joined text.
        
        Args:
            tokens (List[int]): A list of token IDs.
        
        Returns:
            Dict[str, Any]: A dictionary with keys "tokens" (list of token strings)
            and "detokenized_text" (a concatenated string).
        """
        token_strs: List[str] = [self.id2token.get(token, "UNK") for token in tokens]
        return {"tokens": token_strs, "detokenized_text": " ".join(token_strs)}

    def _tokenize_demographics(self, demographics: Dict[str, Any]) -> List[int]:
        """
        Tokenize the demographics section of a patient record.
        
        Expected keys: "sex", "race", "age".
        
        Returns:
            List[int]: List of token IDs for demographics.
        """
        tokens: List[int] = []
        # Tokenize sex
        sex: str = demographics.get("sex", "Unknown_Sex")
        sex_token: int = self.token2id.get(sex, self.token2id.get("UNK"))
        tokens.append(sex_token)
        # Tokenize race: use a prefix "Race_" concatenated with race value.
        race: str = demographics.get("race", "White")
        race_key = f"Race_{race}"
        race_token: int = self.token2id.get(race_key, self.token2id.get("UNK"))
        tokens.append(race_token)
        # Tokenize age by bucketing (5-year bins starting at 18)
        age_val = demographics.get("age", None)
        if age_val is not None and isinstance(age_val, (int, float)):
            # Compute bucket index; for simplicity, assume buckets: 18-22, 23-27, etc.
            bucket_index: int = int((age_val - 18) // 5)
            lower: int = 18 + bucket_index * 5
            upper: int = lower + 4
            age_key: str = f"Age_{lower}_{upper}"
            age_token: int = self.token2id.get(age_key, self.token2id.get("UNK"))
            tokens.append(age_token)
        else:
            tokens.append(self.token2id.get("UNK"))
        return tokens

    def _tokenize_encounter(self, encounter: Dict[str, Any]) -> List[int]:
        """
        Tokenize a single encounter.
        
        Processes encounter-level metadata (start token, encounter type, department,
        chief complaints) and iteratively tokenizes each event in the encounter.
        
        Expected encounter keys:
            - "encounter_type": string
            - "department": string (optional)
            - "chief_complaints": list of dicts with keys "name" and "location" (optional)
            - "events": list of event dicts
        
        Returns:
            List[int]: List of token IDs for the encounter.
        """
        tokens: List[int] = []
        # Add encounter start token
        start_token: int = self.token2id.get("Encounter_Start", self.token2id.get("UNK"))
        tokens.append(start_token)
        # Add encounter type token (constructed as "Encounter_Type_<type>")
        encounter_type: str = encounter.get("encounter_type", "Unknown")
        type_key: str = f"Encounter_Type_{encounter_type}"
        type_token: int = self.token2id.get(type_key, self.token2id.get("UNK"))
        tokens.append(type_token)
        # Add department specialty token if available
        department: str = encounter.get("department", None)
        if department:
            dept_key: str = f"Dept_{department}"
            dept_token: int = self.token2id.get(dept_key, self.token2id.get("UNK"))
            tokens.append(dept_token)
        # Process chief complaints if available (each complaint split into name and location)
        complaints: List[Dict[str, Any]] = encounter.get("chief_complaints", [])
        for comp in complaints:
            name: str = comp.get("name", "")
            if name:
                comp_name_key: str = f"Complaint_{name}"
                tokens.append(self.token2id.get(comp_name_key, self.token2id.get("UNK")))
            location: str = comp.get("location", "")
            if location:
                comp_loc_key: str = f"ComplaintLoc_{location}"
                tokens.append(self.token2id.get(comp_loc_key, self.token2id.get("UNK")))
        # Process each event in the encounter (assumed pre-sorted in chronological order)
        events: List[Dict[str, Any]] = encounter.get("events", [])
        for event in events:
            event_type: str = event.get("type", "").lower()
            if event_type == "diagnosis":
                code: str = event.get("code", "")
                tokens.extend(self._tokenize_diagnosis(code))
            elif event_type == "lab":
                tokens.extend(self._tokenize_lab(event))
            elif event_type == "medication":
                code: str = event.get("code", "")
                tokens.extend(self._tokenize_medication(code))
            elif event_type == "procedure":
                code: str = event.get("code", "")
                tokens.extend(self._tokenize_procedure(code))
            elif event_type == "time":
                # Expect a "delta" value in minutes
                delta_time: float = float(event.get("delta", 0))
                tokens.extend(self._tokenize_time_gap(delta_time))
            else:
                logging.warning("Unknown event type '%s'. Skipping event.", event_type)
        # End the encounter with an end token
        end_token: int = self.token2id.get("Encounter_End", self.token2id.get("UNK"))
        tokens.append(end_token)
        return tokens

    def _tokenize_diagnosis(self, code: str) -> List[int]:
        """
        Tokenize an ICD-10-CM diagnosis code into 1-3 tokens.
        
        The code is cleaned (removing periods) and split as:
            - First token: first three characters (prefixed with "DIAG_")
            - Second token: next character if available (prefixed with "DIAG_")
            - Third token: next character if available (prefixed with "DIAG_")
        
        Args:
            code (str): The ICD-10-CM code.
        
        Returns:
            List[int]: List of token IDs for the diagnosis.
        """
        tokens: List[int] = []
        if not code:
            return [self.token2id.get("UNK", 0)]
        cleaned_code: str = code.replace(".", "")
        # First token: first 3 characters
        if len(cleaned_code) >= 3:
            part1: str = cleaned_code[:3]
            token_str: str = f"DIAG_{part1}"
            tokens.append(self.token2id.get(token_str, self.token2id.get("UNK")))
        else:
            tokens.append(self.token2id.get(f"DIAG_{cleaned_code}", self.token2id.get("UNK")))
        # Second token: next one character if exists
        if len(cleaned_code) > 3:
            part2: str = cleaned_code[3:4]
            token_str: str = f"DIAG_{part2}"
            tokens.append(self.token2id.get(token_str, self.token2id.get("UNK")))
        # Third token: next one character if exists
        if len(cleaned_code) > 4:
            part3: str = cleaned_code[4:5]
            token_str: str = f"DIAG_{part3}"
            tokens.append(self.token2id.get(token_str, self.token2id.get("UNK")))
        return tokens

    def _tokenize_lab(self, event: Dict[str, Any]) -> List[int]:
        """
        Tokenize a lab event into two tokens:
            - One for the lab test (using its LOINC code, prefixed with "LAB_")
            - One for the lab result quantile (using bucketed value)
        
        Args:
            event (Dict[str, Any]): A lab event dict with keys "loinc" and "value".
        
        Returns:
            List[int]: List of two token IDs.
        """
        tokens: List[int] = []
        loinc: str = event.get("loinc", "")
        value = event.get("value", None)
        if not loinc or value is None:
            return [self.token2id.get("UNK", 0)]
        lab_code: str = f"LAB_{loinc}"
        tokens.append(self.token2id.get(lab_code, self.token2id.get("UNK")))
        quantile: int = self._bucket_lab_value(float(value))
        quantile_token_str: str = f"LAB_QUANTILE_{quantile}"
        tokens.append(self.token2id.get(quantile_token_str, self.token2id.get("UNK")))
        return tokens

    def _bucket_lab_value(self, value: float) -> int:
        """
        Bucket a numeric lab value into one of 10 quantiles using the quantile boundaries.
        
        Args:
            value (float): The lab value.
        
        Returns:
            int: An integer between 1 and 10 representing the quantile bucket.
        """
        index: int = bisect.bisect_right(self.lab_quantile_boundaries, value)
        quantile: int = index + 1  # Buckets are 1-indexed
        if quantile > 10:
            quantile = 10
        if quantile < 1:
            quantile = 1
        return quantile

    def _tokenize_medication(self, code: str) -> List[int]:
        """
        Tokenize an ATC medication code into 1-3 tokens.
        
        The code is cleaned (spaces removed) and split as:
            - First token: first 3 characters (prefixed with "MED_")
            - Second token: next 2 characters (prefixed with "MED_")
            - Third token: remaining characters if any (prefixed with "MED_")
        
        Args:
            code (str): The medication ATC code.
        
        Returns:
            List[int]: List of token IDs for the medication.
        """
        tokens: List[int] = []
        if not code:
            return [self.token2id.get("UNK", 0)]
        cleaned_code: str = code.replace(" ", "")
        if len(cleaned_code) >= 3:
            part1: str = cleaned_code[:3]
            token_str: str = f"MED_{part1}"
            tokens.append(self.token2id.get(token_str, self.token2id.get("UNK")))
        else:
            tokens.append(self.token2id.get(f"MED_{cleaned_code}", self.token2id.get("UNK")))
        if len(cleaned_code) > 3:
            part2: str = cleaned_code[3:5]
            token_str: str = f"MED_{part2}"
            tokens.append(self.token2id.get(token_str, self.token2id.get("UNK")))
        if len(cleaned_code) > 5:
            part3: str = cleaned_code[5:]
            token_str: str = f"MED_{part3}"
            tokens.append(self.token2id.get(token_str, self.token2id.get("UNK")))
        return tokens

    def _tokenize_procedure(self, code: str) -> List[int]:
        """
        Tokenize a procedure code into a single token.
        
        If the procedure code is not recognized among the top procedures,
        returns a designated "UNKNOWN_PROCEDURE" token.
        
        Args:
            code (str): The procedure code.
        
        Returns:
            List[int]: List containing a single token ID.
        """
        if not code:
            return [self.token2id.get("UNK", 0)]
        proc_key: str = f"PROC_{code}"
        if proc_key in self.token2id:
            return [self.token2id[proc_key]]
        else:
            return [self.token2id.get("UNKNOWN_PROCEDURE", self.token2id.get("UNK", 0))]

    def _tokenize_time_gap(self, delta_time: float) -> List[int]:
        """
        Tokenize an elapsed time (in minutes) into one or more time tokens.
        
        If the time gap is shorter than the smallest bucket, no token is added.
        If the time gap exceeds the largest predefined bucket, multiple tokens may be added.
        
        Args:
            delta_time (float): Elapsed time in minutes.
        
        Returns:
            List[int]: List of token IDs representing the time gap.
        """
        tokens: List[int] = []
        # Do not add any token if the gap is shorter than the lowest interval.
        if delta_time < self.time_buckets[0][0]:
            return tokens
        bucket_found: bool = False
        for lower, upper, token_name in self.time_buckets:
            if lower <= delta_time < upper:
                tokens.append(self.token2id.get(token_name, self.token2id.get("UNK")))
                bucket_found = True
                break
        # If no bucket found and delta_time exceeds the largest bucket, add multiple tokens.
        if not bucket_found and delta_time >= self.time_buckets[-1][1]:
            largest_lower, largest_upper, largest_token = self.time_buckets[-1]
            count: int = int(delta_time // largest_upper)
            for _ in range(count):
                tokens.append(self.token2id.get(largest_token, self.token2id.get("UNK")))
        return tokens
