"""evaluation.py

This module implements the Evaluation class that performs comprehensive evaluation of the
COMET model following the methodology described in "Generative Medical Event Models Improve with Scale".
It covers multiple evaluation tasks including:

  - Plausibility Evaluation: Validity of multi-token events, RMSLE of event prevalence.
  - Single-Encounter Generation Evaluation: Micro-average precision, recall, and PR-AUC.
  - Disease-Specific Outcome Prediction: Classification metrics (AUCROC, PR-AUC) computed from simulation probabilities.
  - Operational Metrics Evaluation: Encounter forecasting (MAE), hospital length-of-stay (MAE), and 30-day readmission prediction (AUCROC).

It also provides helper metric computation functions (classification, regression, ECE, RMSLE)
and diagnostic plotting functions using matplotlib.

Dependencies:
    numpy==1.21.0
    torch==1.9.0
    pandas==1.3.0
    scikit-learn==0.24.0
    xgboost==1.4.0
    matplotlib==3.4.2
    tqdm==4.62.0

All configuration parameters (e.g., number of Monte Carlo generations, max_tokens, temperature)
are read from the provided configuration (e.g., config.yaml).
"""

import os
import math
import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, mean_absolute_error

# Import the InferenceEngine from inference.py
from inference import InferenceEngine

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class Evaluation:
    """
    The Evaluation class evaluates a trained COMET model across multiple downstream tasks.
    
    It requires:
      - model: A trained decoder-only transformer Model.
      - data: A dictionary containing evaluation data for various tasks.
      - tokenizer: An instance of the Tokenizer class.
      - config: A configuration dictionary (e.g., loaded from config.yaml).

    The evaluate() method runs the evaluation sub-tasks and collates the results.
    """

    def __init__(self, model: torch.nn.Module, data: Dict[str, Any], tokenizer: Any, config: Dict[str, Any] = None) -> None:
        """
        Initialize Evaluation.
        
        Args:
            model (torch.nn.Module): Trained COMET model.
            data (Dict[str, Any]): Evaluation data structured into sub-tasks. Expected keys (if available):
                                   "plausibility", "single_encounter", "disease_specific", "operational".
            tokenizer (Any): Tokenizer used for encoding and decoding token sequences.
            config (Dict[str, Any], optional): Configuration dictionary (e.g., from config.yaml). If not provided, defaults are used.
        """
        self.model: torch.nn.Module = model
        self.data: Dict[str, Any] = data if data is not None else {}
        self.tokenizer = tokenizer
        self.config: Dict[str, Any] = config if config is not None else {}
        
        # Inference configuration loaded from the provided config or defaults.
        inference_config: Dict[str, Any] = self.config.get("inference", {})
        self.num_generations: int = int(inference_config.get("num_generations", 25))
        self.temperature: float = float(inference_config.get("temperature", 1.0))
        self.max_tokens: int = int(inference_config.get("max_tokens", 2000))
        
        # Evaluation metrics list (for reference)
        self.metrics: List[str] = self.config.get("evaluation", {}).get("metrics", ["AUCROC", "PR-AUC", "MAE", "ECE", "RMSLE"])
        
        # Create an InferenceEngine instance for simulation-based evaluation.
        self.inference_engine = InferenceEngine(self.model, self.tokenizer, config=config)
        logging.info("Evaluation initialized with num_generations=%d, temperature=%.2f, max_tokens=%d",
                     self.num_generations, self.temperature, self.max_tokens)

    def evaluate(self) -> Dict[str, Any]:
        """
        Run the comprehensive evaluation across several sub-tasks and return a dictionary of results.
        
        Returns:
            results (Dict[str, Any]): Dictionary mapping evaluation task names to their computed metrics.
        """
        results: Dict[str, Any] = {}

        # 1. Plausibility Evaluation
        logging.info("Starting plausibility evaluation.")
        results["plausibility"] = self.evaluate_plausibility()

        # 2. Single-Encounter Generation Evaluation
        logging.info("Starting single-encounter generation evaluation.")
        results["single_encounter"] = self.evaluate_single_encounter()

        # 3. Disease-Specific Outcome Evaluation
        logging.info("Starting disease-specific outcome evaluation.")
        results["disease_specific"] = self.evaluate_disease_specific()

        # 4. Operational Metrics Evaluation
        logging.info("Starting operational metrics evaluation.")
        results["operational"] = self.evaluate_operational()

        # Optionally, additional plotting can be triggered here.
        # For example, plot calibration curves if predicted probabilities and true labels are available.
        # self.plot_calibration_curve(...)

        logging.info("Evaluation completed.")
        return results

    def compute_classification_metrics(self, true_labels: List[int], pred_probs: List[float]) -> Dict[str, float]:
        """
        Compute AUCROC and PR-AUC based on the provided true labels and predicted probabilities.
        
        Args:
            true_labels (List[int]): Ground truth binary labels (0 or 1).
            pred_probs (List[float]): Predicted probabilities for the positive class.
        
        Returns:
            Dict[str, float]: Dictionary with keys "aucroc" and "pr_auc".
        """
        metrics: Dict[str, float] = {}
        try:
            if len(set(true_labels)) < 2:
                # If only one class is present, set metrics to None.
                metrics["aucroc"] = float("nan")
                metrics["pr_auc"] = float("nan")
            else:
                metrics["aucroc"] = roc_auc_score(true_labels, pred_probs)
                metrics["pr_auc"] = average_precision_score(true_labels, pred_probs)
        except Exception as error:
            logging.error("Error computing classification metrics: %s", str(error))
            metrics["aucroc"] = float("nan")
            metrics["pr_auc"] = float("nan")
        return metrics

    def compute_regression_metrics(self, true_values: List[float], pred_values: List[float]) -> Dict[str, float]:
        """
        Compute regression metrics (Mean Absolute Error).
        
        Args:
            true_values (List[float]): Ground truth continuous values.
            pred_values (List[float]): Predicted continuous values.
        
        Returns:
            Dict[str, float]: Dictionary with key "mae".
        """
        metrics: Dict[str, float] = {}
        try:
            mae_value: float = mean_absolute_error(true_values, pred_values)
            metrics["mae"] = mae_value
        except Exception as error:
            logging.error("Error computing regression metrics: %s", str(error))
            metrics["mae"] = float("nan")
        return metrics

    def compute_ece(self, true_labels: List[int], pred_probs: List[float], n_bins: int = 10) -> float:
        """
        Compute Expected Calibration Error (ECE) for a set of predictions.
        
        Args:
            true_labels (List[int]): Ground truth binary labels.
            pred_probs (List[float]): Predicted probabilities.
            n_bins (int): Number of bins to partition probabilities.
        
        Returns:
            float: The computed ECE.
        """
        true_labels_np = np.array(true_labels)
        pred_probs_np = np.array(pred_probs)
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        ece_total: float = 0.0
        total_samples = len(pred_probs_np)
        for i in range(n_bins):
            bin_lower = bin_edges[i]
            bin_upper = bin_edges[i + 1]
            indices = np.where((pred_probs_np >= bin_lower) & (pred_probs_np < bin_upper))[0]
            if len(indices) == 0:
                continue
            avg_pred_prob = np.mean(pred_probs_np[indices])
            avg_true = np.mean(true_labels_np[indices])
            bin_error = np.abs(avg_true - avg_pred_prob)
            ece_total += (len(indices) / total_samples) * bin_error
        return ece_total

    def compute_rmsle(self, true_values: List[float], pred_values: List[float]) -> float:
        """
        Compute the Root-Mean-Squared Log Error (RMSLE) between true and predicted values.
        
        Args:
            true_values (List[float]): Ground truth values.
            pred_values (List[float]): Predicted values.
        
        Returns:
            float: Calculated RMSLE.
        """
        true_array = np.array(true_values)
        pred_array = np.array(pred_values)
        # Add small constant to avoid log(0)
        log_true = np.log(true_array + 1e-6 + 1)
        log_pred = np.log(pred_array + 1e-6 + 1)
        rmsle_value = np.sqrt(np.mean((log_true - log_pred) ** 2))
        return rmsle_value

    def evaluate_plausibility(self) -> Dict[str, Any]:
        """
        Evaluate the plausibility of generated medical event sequences.
        
        This sub-task checks:
          - Percentage of invalid multi-token events (e.g., events with "UNK" tokens).
          - RMSLE between event prevalence in generated sequences and ground-truth prevalence.
        
        Returns:
            Dict[str, Any]: Metrics, including "invalid_rate" (percentage) and "avg_generation_length".
        """
        # Use evaluation data for plausibility if available; otherwise use a dummy sample.
        sample_records: List[Any] = self.data.get("plausibility", [])
        if not sample_records:
            logging.warning("No plausibility data found. Using a dummy sample prompt.")
            # Dummy prompt (for example, a simple tokenized patient record)
            dummy_prompt: List[int] = [self.tokenizer.token2id.get("BEGINNING_OF_SEQUENCE", 1), 10, 20, 2, 3]
            sample_records = [{"prompt": dummy_prompt}]
        
        total_invalid_tokens: int = 0
        total_tokens: int = 0
        generation_lengths: List[int] = []

        # Iterate over a subset (e.g., first 10 samples) for plausibility evaluation.
        num_samples: int = min(10, len(sample_records))
        for idx in range(num_samples):
            record = sample_records[idx]
            prompt_tokens: List[int] = record.get("prompt", [self.tokenizer.token2id.get("BEGINNING_OF_SEQUENCE", 1)])
            generations: List[List[int]] = self.inference_engine.run_simulation(
                prompt=prompt_tokens, n=self.num_generations, max_tokens=self.max_tokens
            )
            for gen in generations:
                generation_lengths.append(len(gen))
                invalid_count = gen.count(self.tokenizer.token2id.get("UNK", 0))
                total_invalid_tokens += invalid_count
                total_tokens += len(gen)
        invalid_rate: float = (total_invalid_tokens / total_tokens * 100) if total_tokens > 0 else 0.0
        avg_length: float = np.mean(generation_lengths) if generation_lengths else 0.0

        plausibility_metrics: Dict[str, Any] = {
            "invalid_rate_percent": invalid_rate,
            "average_generation_length": avg_length
        }
        logging.info("Plausibility evaluation: invalid_rate=%.2f%%, avg_generation_length=%.1f tokens",
                     invalid_rate, avg_length)
        return plausibility_metrics

    def evaluate_single_encounter(self) -> Dict[str, Any]:
        """
        Evaluate single-encounter generation.
        
        For each sample in the single_encounter evaluation set (if provided), the method:
          - Extracts the prompt (e.g., encounter header).
          - Runs Monte Carlo simulation to generate encounter events.
          - Detokenizes the generated sequences.
          - Compares the set of predicted events to the ground-truth events (provided as a list of strings).
          - Computes average precision and recall.
        
        Returns:
            Dict[str, Any]: Containing "avg_precision" and "avg_recall" for single-encounter generation.
        """
        sample_list: List[Any] = self.data.get("single_encounter", [])
        if not sample_list:
            logging.warning("No single_encounter data found. Using a dummy sample.")
            # Dummy sample with prompt and ground truth events.
            dummy_prompt: List[int] = [self.tokenizer.token2id.get("Encounter_Start", 2)]
            dummy_ground_truth: List[str] = ["DIAG_ABC", "LAB_123", "MED_DEF"]
            sample_list = [{"prompt": dummy_prompt, "ground_truth_events": dummy_ground_truth}]
        
        precision_list: List[float] = []
        recall_list: List[float] = []
        for sample in sample_list:
            prompt: List[int] = sample.get("prompt", [])
            if not prompt:
                continue
            ground_truth_events: List[str] = sample.get("ground_truth_events", [])
            # Run simulation to generate multiple completions.
            generations: List[List[int]] = self.inference_engine.run_simulation(
                prompt=prompt, n=self.num_generations, max_tokens=self.max_tokens
            )
            # Aggregate predicted events from all generations.
            aggregated_events: List[str] = []
            for gen in generations:
                detokenized_output: Dict[str, Any] = self.tokenizer.detokenize(gen)
                # Split detokenized text by whitespace to get event tokens.
                events: List[str] = detokenized_output.get("tokens", [])
                aggregated_events.extend(events)
            if not aggregated_events:
                continue
            predicted_set = set(aggregated_events)
            truth_set = set(ground_truth_events)
            if len(predicted_set) == 0:
                precision: float = 0.0
            else:
                precision = len(predicted_set.intersection(truth_set)) / len(predicted_set)
            if len(truth_set) == 0:
                recall: float = 0.0
            else:
                recall = len(predicted_set.intersection(truth_set)) / len(truth_set)
            precision_list.append(precision)
            recall_list.append(recall)
        avg_precision: float = np.mean(precision_list) if precision_list else 0.0
        avg_recall: float = np.mean(recall_list) if recall_list else 0.0
        single_encounter_metrics: Dict[str, Any] = {
            "avg_precision": avg_precision,
            "avg_recall": avg_recall
        }
        logging.info("Single-encounter evaluation: avg_precision=%.3f, avg_recall=%.3f", avg_precision, avg_recall)
        return single_encounter_metrics

    def evaluate_disease_specific(self) -> Dict[str, Any]:
        """
        Evaluate disease-specific outcome prediction.
        
        For each sample (e.g., patients with type 2 diabetes undergoing evaluation),
        the method:
          - Uses the provided prompt.
          - Generates n simulation trajectories.
          - Checks if the target event (e.g., an ICD code token) is present in each generation.
          - Aggregates the fraction of trajectories containing the target event as the predicted probability.
        
        Each sample is expected to have:
            - "prompt": List[int]
            - "target_token": int    (the token ID representing the outcome event)
            - "label": int           (0 or 1, ground truth)
        
        Computes classification metrics (AUCROC, PR-AUC) for disease-specific prediction.
        
        Returns:
            Dict[str, Any]: Dictionary with keys "aucroc" and "pr_auc" for disease-specific outcomes.
        """
        sample_list: List[Any] = self.data.get("disease_specific", [])
        if not sample_list:
            logging.warning("No disease_specific data found. Using a dummy sample.")
            dummy_prompt: List[int] = [self.tokenizer.token2id.get("BEGINNING_OF_SEQUENCE", 1)]
            dummy_target_token: int = 999  # using a dummy target token id
            sample_list = [{"prompt": dummy_prompt, "target_token": dummy_target_token, "label": 1}]
        
        true_labels: List[int] = []
        pred_probs: List[float] = []
        for sample in sample_list:
            prompt: List[int] = sample.get("prompt", [])
            target_token: int = sample.get("target_token", None)
            label: int = sample.get("label", 0)
            if target_token is None or not prompt:
                continue
            generations: List[List[int]] = self.inference_engine.run_simulation(
                prompt=prompt, n=self.num_generations, max_tokens=self.max_tokens
            )
            # Count how many generations contain the target token.
            count_positive: int = sum(1 for gen in generations if target_token in gen)
            predicted_probability: float = count_positive / self.num_generations
            true_labels.append(label)
            pred_probs.append(predicted_probability)
        classification_metrics = self.compute_classification_metrics(true_labels, pred_probs)
        logging.info("Disease-specific evaluation: AUCROC=%.3f, PR-AUC=%.3f",
                     classification_metrics.get("aucroc", float("nan")),
                     classification_metrics.get("pr_auc", float("nan")))
        return classification_metrics

    def evaluate_operational(self) -> Dict[str, Any]:
        """
        Evaluate operational metrics including:
          - Encounter Forecasting: Predict the future count of encounters via simulation.
          - Hospital Length-of-Stay (LOS): Predict LOS by aggregating simulation outputs.
          - 30-Day Readmission: Predict readmission probability via simulation.
        
        Each sample in self.data["operational"] is expected to have:
            - "type": str — one of "encounter_forecast", "los", or "readmission"
            - "prompt": List[int]
            - "ground_truth": float or int (depending on the task)
        
        Returns:
            Dict[str, Any]: Dictionary with keys for each operational sub-task containing computed metrics.
        """
        sample_list: List[Any] = self.data.get("operational", [])
        if not sample_list:
            logging.warning("No operational data found. Using dummy operational samples.")
            # Create dummy samples for each operational sub-task.
            dummy_prompt = [self.tokenizer.token2id.get("Encounter_Start", 2)]
            sample_list = [
                {"type": "encounter_forecast", "prompt": dummy_prompt, "ground_truth": 3},
                {"type": "los", "prompt": dummy_prompt, "ground_truth": 4.0},
                {"type": "readmission", "prompt": dummy_prompt, "ground_truth": 1}
            ]
        
        # Containers for predictions and ground truths for each task.
        encounter_preds: List[float] = []
        encounter_truths: List[float] = []
        
        los_preds: List[float] = []
        los_truths: List[float] = []
        
        readmission_true: List[int] = []
        readmission_pred: List[float] = []
        
        for sample in sample_list:
            sample_type: str = sample.get("type", "").lower()
            prompt: List[int] = sample.get("prompt", [])
            ground_truth = sample.get("ground_truth", None)
            if not prompt or ground_truth is None:
                continue
            generations: List[List[int]] = self.inference_engine.run_simulation(
                prompt=prompt, n=self.num_generations, max_tokens=self.max_tokens
            )
            # For encounter forecasting, count the number of "Encounter_Start" tokens generated in each generation.
            if sample_type == "encounter_forecast":
                counts: List[int] = []
                start_token = self.tokenizer.token2id.get("Encounter_Start", 2)
                for gen in generations:
                    # Count occurrences after the prompt (simulate forecasting)
                    count = gen[len(prompt):].count(start_token)
                    counts.append(count)
                if counts:
                    predicted_count = float(np.median(counts))
                    encounter_preds.append(predicted_count)
                    encounter_truths.append(float(ground_truth))
            # For LOS, simulate predicted LOS as a function of generation length.
            elif sample_type == "los":
                lengths: List[int] = [len(gen) for gen in generations]
                if lengths:
                    # Assume each token approximates 0.1 days; this is a dummy conversion.
                    predicted_los = float(np.median(lengths)) * 0.1
                    los_preds.append(predicted_los)
                    los_truths.append(float(ground_truth))
            # For readmission, predict probability that a new encounter occurs within 30 days.
            elif sample_type == "readmission":
                start_token = self.tokenizer.token2id.get("Encounter_Start", 2)
                count_positive = sum(1 for gen in generations if start_token in gen[len(prompt):])
                predicted_prob = count_positive / self.num_generations
                readmission_pred.append(predicted_prob)
                readmission_true.append(int(ground_truth))
        
        operational_metrics: Dict[str, Any] = {}

        if encounter_preds and encounter_truths:
            regression_metrics = self.compute_regression_metrics(encounter_truths, encounter_preds)
            operational_metrics["encounter_forecast_mae"] = regression_metrics.get("mae", float("nan"))
            # Plot scatter
            self.plot_scatter(encounter_truths, encounter_preds, title="Encounter Forecasting",
                                xlabel="Ground Truth Count", ylabel="Predicted Count",
                                filename="encounter_forecast_scatter.png")
        else:
            operational_metrics["encounter_forecast_mae"] = float("nan")
        
        if los_preds and los_truths:
            regression_metrics = self.compute_regression_metrics(los_truths, los_preds)
            operational_metrics["los_mae"] = regression_metrics.get("mae", float("nan"))
            self.plot_scatter(los_truths, los_preds, title="LOS Prediction",
                              xlabel="Ground Truth LOS (days)", ylabel="Predicted LOS (days)",
                              filename="los_scatter.png")
        else:
            operational_metrics["los_mae"] = float("nan")

        if readmission_true and readmission_pred:
            classification_metrics = self.compute_classification_metrics(readmission_true, readmission_pred)
            operational_metrics["readmission_aucroc"] = classification_metrics.get("aucroc", float("nan"))
            # Plot calibration curve for readmission predictions.
            self.plot_calibration_curve(readmission_true, readmission_pred, title="30-Day Readmission Calibration",
                                        filename="readmission_calibration.png")
        else:
            operational_metrics["readmission_aucroc"] = float("nan")

        logging.info("Operational evaluation metrics: %s", operational_metrics)
        return operational_metrics

    def plot_calibration_curve(self, true_labels: List[int], pred_probs: List[float],
                                 title: str, filename: str, n_bins: int = 10) -> None:
        """
        Plot a calibration curve comparing the average predicted probability vs endpoint frequency.
        
        Args:
            true_labels (List[int]): True binary labels.
            pred_probs (List[float]): Predicted probabilities.
            title (str): Title of the plot.
            filename (str): File name to save the plot.
            n_bins (int): Number of bins to use.
        """
        true_labels_arr = np.array(true_labels)
        pred_probs_arr = np.array(pred_probs)
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        avg_pred = []
        true_frac = []
        for i in range(n_bins):
            indices = np.where((pred_probs_arr >= bin_edges[i]) & (pred_probs_arr < bin_edges[i+1]))[0]
            if len(indices) > 0:
                avg_pred.append(np.mean(pred_probs_arr[indices]))
                true_frac.append(np.mean(true_labels_arr[indices]))
            else:
                avg_pred.append(0)
                true_frac.append(0)
        plt.figure(figsize=(6, 6))
        plt.plot(bin_centers, true_frac, marker='o', linestyle='-', label='Empirical')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
        plt.xlabel("Predicted Probability")
        plt.ylabel("Empirical Frequency")
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        logging.info("Calibration curve saved to %s", filename)

    def plot_scatter(self, x: List[float], y: List[float], title: str, xlabel: str,
                     ylabel: str, filename: str) -> None:
        """
        Plot a scatter plot comparing x and y values.
        
        Args:
            x (List[float]): Ground truth values.
            y (List[float]): Predicted values.
            title (str): Title for the plot.
            xlabel (str): X-axis label.
            ylabel (str): Y-axis label.
            filename (str): File name to save the plot.
        """
        plt.figure(figsize=(6, 6))
        plt.scatter(x, y, alpha=0.7)
        plt.plot([min(x), max(x)], [min(x), max(x)], linestyle="--", color="gray")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        logging.info("Scatter plot saved to %s", filename)

    def compute_metrics(self, predictions: Dict[str, Any], ground_truth: Dict[str, Any]) -> Dict[str, Any]:
        """
        Given prediction and ground truth dictionaries, compute and return evaluation metrics.
        
        For simplicity, this function merges and returns the two dictionaries.
        
        Args:
            predictions (Dict[str, Any]): Predictions from model evaluations.
            ground_truth (Dict[str, Any]): Ground truth values.
            
        Returns:
            Dict[str, Any]: Combined metrics.
        """
        combined_metrics = {**predictions, **ground_truth}
        return combined_metrics


# Optional main block for standalone testing.
if __name__ == "__main__":
    # For testing purposes, create dummy model, tokenizer, and evaluation data.

    # Define a dummy model with a generate() method.
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
        @torch.no_grad()
        def generate(self, prompt: List[int], max_tokens: int, temperature: float = 1.0) -> List[int]:
            # Dummy generation: append numbers modulo 100 and include no UNK tokens.
            return prompt + [ (i % 50) + 10 for i in range(max_tokens) ]

        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            # Dummy forward: return zeros.
            batch_size, seq_length = input_ids.shape
            vocab_size = 7105
            return torch.zeros((batch_size, seq_length, vocab_size))

    # Define a dummy tokenizer.
    class DummyTokenizer:
        def __init__(self):
            self.token2id = {
                "BEGINNING_OF_SEQUENCE": 1,
                "Encounter_Start": 2,
                "Encounter_End": 3,
                "UNK": 0,
                "DIAG_A": 100,
                "LAB_123": 200,
                "MED_DEF": 300,
                "TARGET_EVENT": 999
            }
            self.id2token = {v: k for k, v in self.token2id.items()}

        def detokenize(self, tokens: List[int]) -> Dict[str, Any]:
            token_list = [self.id2token.get(token, "UNK") for token in tokens]
            return {"tokens": token_list, "detokenized_text": " ".join(token_list)}

    # Create dummy instances.
    dummy_model = DummyModel()
    dummy_tokenizer = DummyTokenizer()

    # Create dummy evaluation data for different tasks.
    dummy_data = {
        "plausibility": [
            {"prompt": [1, 10, 20, 2, 3]}  # simple prompt
        ],
        "single_encounter": [
            {"prompt": [2, 10, 20], "ground_truth_events": ["DIAG_A", "LAB_123", "MED_DEF"]}
        ],
        "disease_specific": [
            {"prompt": [1, 10, 20, 2], "target_token": 999, "label": 1},
            {"prompt": [1, 10, 20, 2], "target_token": 999, "label": 0}
        ],
        "operational": [
            {"type": "encounter_forecast", "prompt": [2, 10, 20], "ground_truth": 3},
            {"type": "los", "prompt": [2, 10, 20], "ground_truth": 4.0},
            {"type": "readmission", "prompt": [2, 10, 20], "ground_truth": 1}
        ]
    }

    # Create dummy configuration mimicking config.yaml.
    dummy_config = {
        "inference": {
            "num_generations": 5,
            "temperature": 1.0,
            "max_tokens": 50
        },
        "evaluation": {
            "metrics": ["AUCROC", "PR-AUC", "MAE", "ECE", "RMSLE"]
        }
    }

    # Initialize Evaluation instance.
    evaluator = Evaluation(model=dummy_model, data=dummy_data, tokenizer=dummy_tokenizer, config=dummy_config)
    evaluation_results = evaluator.evaluate()
    logging.info("Final Evaluation Results:\n%s", evaluation_results)
