import os
import torch
import logging
import argparse
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, GPT2LMHeadModel

# Global logging
def setup_logging(output_file):
    log_filename = os.path.splitext(output_file)[0] + '.log'
    
    # Remove any existing handlers
    logging.getLogger().handlers.clear()

    # File handler for logging to file
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    # Stream handler for logging to console
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    # Add handlers to the root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # root logger level
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

setup_logging('generated_SMILES.txt')


def load_model_and_tokenizer(model_name):
    """Load the model and tokenizer from HuggingFace."""
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
    def __init__(self, model, tokenizer, uniprot_to_sequence, output_file="generated_SMILES.txt"):
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
            "max_retries": 30  # Max retry limit to avoid infinite loops
        }

        # Initialize model, tokenizer, and UniProt mapping
        self.model = model
        self.tokenizer = tokenizer
        self.uniprot_to_sequence = uniprot_to_sequence

        # Set the device to CUDA if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        # Adjust generation parameters with token IDs
        self.generation_kwargs = self.config["generation_kwargs"]
        self.generation_kwargs["bos_token_id"] = self.tokenizer.bos_token_id
        self.generation_kwargs["eos_token_id"] = self.tokenizer.eos_token_id
        self.generation_kwargs["pad_token_id"] = self.tokenizer.pad_token_id        

    def generate_smiles(self, sequence, num_generated):
        """Generate unique SMILES with a retry limit to avoid infinite loops."""
 
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

                    continue

            retries += 1

        logging.info(f"SMILES generation for sequence completed. Generated {len(generated_smiles_set)} SMILES.")
        return list(generated_smiles_set)

    def generate_smiles_data(self, list_of_sequences=None, list_of_uniprot_ids=None, num_generated=10):
        """Generate SMILES data for sequences or UniProt IDs."""
        if not list_of_sequences and not list_of_uniprot_ids:
            raise ValueError("Either `list_of_sequences` or `list_of_uniprot_ids` must be provided.")

        # Prepare sequences input
        sequences_input = []
        not_found_uniprot_ids = []

        # If sequences are provided directly
        if list_of_sequences:
            sequences_input.extend(list_of_sequences)

        # If UniProt IDs are provided, map them to sequences
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
        logging.info(f"Completed SMILES generation for {len(data)} entries.")  # Log the completion of the process

        return pd.DataFrame(data)
    

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate unique SMILES for protein sequences or UniProt IDs.")
    parser.add_argument("--sequences", nargs="*", help="Protein sequences (space-separated).")
    parser.add_argument("--uniprot_ids", nargs="*", help="UniProt IDs (space-separated).")
    parser.add_argument("--num_generated", type=int, default=10, help="Number of unique SMILES to generate per input.")
    parser.add_argument("--output_file", type=str, default="generated_SMILES.txt", help="Output file name.")
    args = parser.parse_args()

    # Ensure the output file has a .txt extension
    if not args.output_file.endswith('.txt'):
        args.output_file += '.txt'


    # log file name
    setup_logging(args.output_file)

    # Check if at least one of sequences or UniProt IDs is provided
    if not args.sequences and not args.uniprot_ids:
        raise ValueError("You must provide either protein sequences or UniProt IDs.")

    # Load model and tokenizer
    model_name = "alimotahharynia/DrugGen"
    model, tokenizer = load_model_and_tokenizer(model_name)

    # Load UniProt dataset for mapping
    dataset_name = "alimotahharynia/approved_drug_target"
    dataset_key = "uniprot_sequence"
    dataset = load_dataset(dataset_name, dataset_key)
    uniprot_to_sequence = {row["UniProt_id"]: row["Sequence"] for row in dataset["uniprot_seq"]}
    

    # Initialize the generator
    generator = SMILESGenerator(model, tokenizer, uniprot_to_sequence, output_file=args.output_file)
    logging.info("Starting SMILES generation process...")
    
    # Generate SMILES data
    df = generator.generate_smiles_data(
        list_of_sequences=args.sequences,
        list_of_uniprot_ids=args.uniprot_ids,
        num_generated=args.num_generated
    )

    # Save the output
    df.to_csv(args.output_file, sep="\t", index=False)
    print(f"Generated SMILES saved to {args.output_file}")
