import os
import torch
import logging
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, GPT2LMHeadModel

# Global logging setup
def setup_logging(output_file):
    log_filename = os.path.splitext(output_file)[0] + ".log"

    logging.getLogger().handlers.clear()

    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

# Load model and tokenizer
def load_model_and_tokenizer(model_name):
    logging.info(f"Loading model and tokenizer: {model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name)
        # Move the model to CUDA if available
        if torch.cuda.is_available():
            logging.info("Moving model to CUDA device.")
            model = model.to("cuda")
        return model, tokenizer
    except Exception as e:
        logging.error(f"Error loading model and tokenizer: {e}")
        raise RuntimeError(f"Failed to load model and tokenizer: {e}")

class SMILESGenerator:
    def __init__(self, model=None, tokenizer=None, uniprot_to_sequence=None, output_file="generated_SMILES.txt"):
        if model is None or tokenizer is None or uniprot_to_sequence is None:
            model_name = "alimotahharynia/DrugGen"
            model, tokenizer = load_model_and_tokenizer(model_name)

            dataset_name = "alimotahharynia/approved_drug_target"
            dataset_key = "uniprot_sequence"
            dataset = load_dataset(dataset_name, dataset_key)
            uniprot_to_sequence = {row["UniProt_id"]: row["Sequence"] for row in dataset["uniprot_seq"]}

        self.output_file = output_file

        # Generation parameters
        self.config = {
            "generation_kwargs": {
                "do_sample": True,
                "top_k": 9,
                "max_length": 1024,
                "top_p": 0.9,
                "num_return_sequences": 10
            },
            "max_retries": 30
        }

        self.model = model
        self.tokenizer = tokenizer
        self.uniprot_to_sequence = uniprot_to_sequence

        # Generation device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        # Adjust generation parameters with token IDs
        self.generation_kwargs = self.config["generation_kwargs"]
        self.generation_kwargs["bos_token_id"] = self.tokenizer.bos_token_id
        self.generation_kwargs["eos_token_id"] = self.tokenizer.eos_token_id
        self.generation_kwargs["pad_token_id"] = self.tokenizer.pad_token_id

    def generate_smiles(self, sequence, num_generated):
        generated_smiles_set = set()
        prompt = f"<|startoftext|><P>{sequence}<L>"
        encoded_prompt = self.tokenizer(prompt, return_tensors="pt")["input_ids"].to(self.device)
        retries = 0

        logging.info(f"Generating SMILES for sequence: {sequence[:10]}...")

        while len(generated_smiles_set) < num_generated:
            if retries >= self.config["max_retries"]:
                logging.warning("Max retries reached. Returning what has been generated so far.")
                break

            sample_outputs = self.model.generate(encoded_prompt, **self.generation_kwargs)
            for sample_output in sample_outputs:
                output_decode = self.tokenizer.decode(sample_output, skip_special_tokens=False)
                try:
                    generated_smiles = output_decode.split("<L>")[1].split("<|endoftext|>")[0]
                    if generated_smiles not in generated_smiles_set:
                        generated_smiles_set.add(generated_smiles)
                except (IndexError, AttributeError) as e:
                    logging.warning(f"Failed to parse SMILES due to error: {str(e)}. Skipping.")

            retries += 1

        logging.info(f"SMILES generation for sequence completed. Generated {len(generated_smiles_set)} SMILES.")
        return list(generated_smiles_set)

    def generate_smiles_data(self, list_of_sequences=None, list_of_uniprot_ids=None, num_generated=10):
        if not list_of_sequences and not list_of_uniprot_ids:
            raise ValueError("Either `list_of_sequences` or `list_of_uniprot_ids` must be provided.")

        sequences_input = []
        not_found_uniprot_ids = []

        if list_of_sequences:
            sequences_input.extend(list_of_sequences)

        if list_of_uniprot_ids:
            for uid in list_of_uniprot_ids:
                sequence = self.uniprot_to_sequence.get(uid)
                if sequence:
                    sequences_input.append(sequence)
                else:
                    logging.warning(f"UniProt ID {uid} not found in the dataset.")
                    not_found_uniprot_ids.append(uid)

        data = []
        for sequence in sequences_input:
            smiles = self.generate_smiles(sequence, num_generated)
            uniprot_id = next((uid for uid, seq in self.uniprot_to_sequence.items() if seq == sequence), None)
            data.append({"UniProt_id": uniprot_id, "sequence": sequence, "SMILES": smiles})

        for uid in not_found_uniprot_ids:
            data.append({"UniProt_id": uid, "sequence": "N/A", "SMILES": "N/A"})

        logging.info(f"Completed SMILES generation for {len(data)} entries.")
        return pd.DataFrame(data)

# Main function for inference
def run_inference(sequences=None, uniprot_ids=None, num_generated=10, output_file="generated_SMILES.txt"):
    # Setup logging
    setup_logging(output_file)

    # Load model and tokenizer
    model_name = "alimotahharynia/DrugGen"
    model, tokenizer = load_model_and_tokenizer(model_name)

    # Load UniProt dataset for mapping
    dataset_name = "alimotahharynia/approved_drug_target"
    dataset_key = "uniprot_sequence"
    dataset = load_dataset(dataset_name, dataset_key)
    uniprot_to_sequence = {row["UniProt_id"]: row["Sequence"] for row in dataset["uniprot_seq"]}

    # Initialize the generator
    generator = SMILESGenerator(model, tokenizer, uniprot_to_sequence, output_file=output_file)
    logging.info("Starting SMILES generation process...")
    
    # Generate SMILES data
    df = generator.generate_smiles_data(
        list_of_sequences=sequences,
        list_of_uniprot_ids=uniprot_ids,
        num_generated=num_generated
    )

    # Save the output
    df.to_csv(output_file, sep="\t", index=False)
    print(f"Generated SMILES saved to {output_file}")
