"""inference.py

This module defines the InferenceEngine class that performs Monte Carlo simulation-based
inference using a trained Qwen2-inspired decoder-only transformer model and a Tokenizer.
It implements:
  - __init__(model: Model, tokenizer: Tokenizer, config: Dict[str, Any] = None)
  - run_simulation(prompt: List[int], n: int, max_tokens: int) -> List[List[int]]
  - aggregate_predictions(generations: List[List[int]]) -> Dict[str, Any]

The InferenceEngine leverages configuration parameters from config.yaml,
such as the default number of generations, temperature, and maximum tokens.

It is used to generate multiple future patient trajectories in an autoregressive manner
and then aggregate those trajectories into useful downstream prediction statistics.
"""

import logging
from typing import List, Dict, Any

import torch

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class InferenceEngine:
    """
    InferenceEngine uses a trained model and a tokenizer to perform Monte Carlo simulations
    of patient event trajectories given a prompt sequence of token IDs.

    Methods:
        __init__(model, tokenizer, config): Initialize with a Model and Tokenizer; load inference defaults.
        run_simulation(prompt, n, max_tokens): Run n Monte Carlo generations starting from prompt.
        aggregate_predictions(generations): Aggregate and summarize the generated token sequences.
    """

    def __init__(self, model: Any, tokenizer: Any, config: Dict[str, Any] = None) -> None:
        """
        Initialize the InferenceEngine.

        Args:
            model: An instance of the Model class supporting forward() and generate().
            tokenizer: An instance of the Tokenizer class with tokenize() and detokenize() methods.
            config: Optional configuration dictionary. If provided and includes the key "inference",
                    defaults are read from config["inference"]. Otherwise, default values are used:
                        - num_generations: 25
                        - temperature: 1.0
                        - max_tokens: 2000
        """
        # Read inference configuration from provided config, or use defaults according to config.yaml.
        if config is not None and isinstance(config, dict) and "inference" in config:
            inf_config = config["inference"]
        else:
            inf_config = {}

        self.num_generations: int = int(inf_config.get("num_generations", 25))
        self.temperature: float = float(inf_config.get("temperature", 1.0))
        self.max_tokens: int = int(inf_config.get("max_tokens", 2000))

        # Store model and tokenizer references.
        self.model = model
        self.tokenizer = tokenizer

        logging.info("InferenceEngine initialized with defaults: num_generations=%d, temperature=%.2f, max_tokens=%d",
                     self.num_generations, self.temperature, self.max_tokens)

    def run_simulation(self, prompt: List[int], n: int = None, max_tokens: int = None) -> List[List[int]]:
        """
        Run Monte Carlo simulation to generate multiple future trajectories from a given prompt.

        Args:
            prompt (List[int]): A list of token IDs representing the patient's history.
            n (int, optional): Number of generations to run. Defaults to the configured value.
            max_tokens (int, optional): Maximum number of tokens to generate per simulation.
                                        Defaults to the configured value.

        Returns:
            List[List[int]]: A list containing n generated token sequences.
        """
        if n is None:
            n = self.num_generations
        if max_tokens is None:
            max_tokens = self.max_tokens

        logging.info("Starting simulation: running %d generations with max_tokens=%d and temperature=%.2f",
                     n, max_tokens, self.temperature)

        all_generations: List[List[int]] = []
        for generation_index in range(n):
            try:
                # Copy the prompt to ensure each generation starts from the same state.
                prompt_copy: List[int] = prompt.copy()
                # Generate a new sequence using the model's generate() method.
                generated_sequence: List[int] = self.model.generate(prompt_copy, max_tokens, self.temperature)
                if not generated_sequence:
                    logging.warning("Generation %d returned an empty sequence.", generation_index)
                    continue
                logging.debug("Generation %d produced %d tokens.", generation_index, len(generated_sequence))
                all_generations.append(generated_sequence)
            except Exception as e:
                logging.error("Error during generation %d: %s", generation_index, str(e))
                continue

        logging.info("Completed simulation: %d generations obtained.", len(all_generations))
        return all_generations

    def aggregate_predictions(self, generations: List[List[int]]) -> Dict[str, Any]:
        """
        Aggregate multiple generated token sequences into summarized predictions.

        This basic aggregation converts each generated sequence back into a human-readable
        representation using the tokenizer's detokenize() method. It then computes summary
        statistics such as average, minimum, and maximum generated sequence lengths.
        Optionally, if a target event token ("TARGET_EVENT") is defined in the tokenizer's vocabulary,
        the method computes the probability of that event appearing in the generations.

        Args:
            generations (List[List[int]]): List of generated token sequences.

        Returns:
            Dict[str, Any]: A dictionary containing aggregated predictions:
                {
                    "raw_predictions": List of detokenized predictions (dict with tokens and text),
                    "average_length": (float) average number of tokens,
                    "min_length": (int) minimum sequence length,
                    "max_length": (int) maximum sequence length,
                    "target_probability": (float or None) probability of target event occurrence (if defined)
                }
        """
        if not generations:
            logging.warning("No generation sequences provided for aggregation.")
            return {}

        aggregated_results: Dict[str, Any] = {}
        detokenized_predictions: List[Dict[str, Any]] = []
        lengths: List[int] = []

        # Process each generated sequence.
        for gen_index, gen_tokens in enumerate(generations):
            detok_output: Dict[str, Any] = self.tokenizer.detokenize(gen_tokens)
            detokenized_predictions.append(detok_output)
            lengths.append(len(gen_tokens))
            logging.debug("Generation %d: %d tokens.", gen_index, len(gen_tokens))

        avg_length: float = sum(lengths) / len(lengths) if lengths else 0.0
        min_length: int = min(lengths) if lengths else 0
        max_length: int = max(lengths) if lengths else 0

        aggregated_results["raw_predictions"] = detokenized_predictions
        aggregated_results["average_length"] = avg_length
        aggregated_results["min_length"] = min_length
        aggregated_results["max_length"] = max_length

        # If a "TARGET_EVENT" token exists in vocabulary, compute its occurrence probability.
        target_probability: Any = None
        if "TARGET_EVENT" in self.tokenizer.token2id:
            target_token_id: int = self.tokenizer.token2id["TARGET_EVENT"]
            count_positive: int = sum(1 for gen in generations if target_token_id in gen)
            target_probability = count_positive / len(generations)
            logging.info("Target event found in vocabulary. Computed target probability: %.4f", target_probability)
        else:
            logging.info("No 'TARGET_EVENT' token found in vocabulary; skipping target probability computation.")

        aggregated_results["target_probability"] = target_probability
        return aggregated_results


# Optional: Main block for standalone testing.
if __name__ == "__main__":
    # Define dummy classes for Model and Tokenizer for test purposes.
    class DummyModel:
        def __init__(self):
            # Assume a dummy device setting.
            self.device = torch.device("cpu")
        @torch.no_grad()
        def generate(self, prompt: List[int], max_tokens: int, temperature: float = 1.0) -> List[int]:
            # Dummy generation: simply append an incremental sequence of numbers modulo 100.
            return prompt + [ (i % 100) for i in range(1, max_tokens + 1) ]

    class DummyTokenizer:
        def __init__(self):
            self.token2id = {"BEGINNING_OF_SEQUENCE": 1, "UNK": 0, "TARGET_EVENT": 9999}
            self.id2token = {1: "BEGINNING_OF_SEQUENCE", 0: "UNK", 9999: "TARGET_EVENT"}
        def detokenize(self, tokens: List[int]) -> Dict[str, Any]:
            token_strings = [self.id2token.get(token, "UNK") for token in tokens]
            return {"tokens": token_strings, "detokenized_text": " ".join(token_strings)}

    # Create instances of the dummy model and tokenizer.
    dummy_model = DummyModel()
    dummy_tokenizer = DummyTokenizer()

    # Create a dummy configuration mimicking config.yaml structure.
    dummy_config = {
        "inference": {
            "num_generations": 5,
            "temperature": 1.0,
            "max_tokens": 50
        }
    }

    # Initialize the InferenceEngine.
    inference_engine = InferenceEngine(dummy_model, dummy_tokenizer, config=dummy_config)

    # Define a dummy prompt (list of token IDs).
    dummy_prompt: List[int] = [1, 10, 20]  # e.g. BEGINNING_OF_SEQUENCE, "Male", "Race_White"

    # Run simulation.
    generations: List[List[int]] = inference_engine.run_simulation(dummy_prompt)
    logging.info("Simulations complete. Number of generations: %d", len(generations))

    # Aggregate predictions.
    aggregated_output: Dict[str, Any] = inference_engine.aggregate_predictions(generations)
    logging.info("Aggregation results: %s", aggregated_output)
