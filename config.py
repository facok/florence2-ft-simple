import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Train a causal language model with custom dataset.")
    parser.add_argument("--images_dir", type=str, required=True, help="Path to the images directory.")
    parser.add_argument("--texts_dir", type=str, required=True, help="Path to the texts directory.")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to the model directory.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the model checkpoints.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training.")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for DataLoader.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=1e-6, help="Learning rate.")
    parser.add_argument("--accumulation_steps", type=int, default=8, help="Gradient accumulation steps.")
    parser.add_argument("--task_type", type=str, required=True, choices=["<CAPTION>", "<DETAILED_CAPTION>", "<MORE_DETAILED_CAPTION>"], help="Task type for the prompt.")
    parser.add_argument("--precision", type=str, choices=["fp32", "fp16", "bf16"], default="bf16", help="Precision for training.")
    parser.add_argument("--train_split", type=float, default=0.8, help="Proportion of data to use for training (default: 0.8).")
    parser.add_argument("--save_best_model", action="store_true", help="Save the model only if validation loss improves.")
    return parser.parse_args()
