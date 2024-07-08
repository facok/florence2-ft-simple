import torch
from tqdm import tqdm
import os
import json
from transformers import get_scheduler

def train_model(train_loader, val_loader, model, processor, epochs, lr, accumulation_steps, output_dir, device, logger, task_prompt, save_best_model):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    num_training_steps = epochs * len(train_loader)
    lr_scheduler = get_scheduler(name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        optimizer.zero_grad()
        train_progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}")
        for step, batch in enumerate(train_progress_bar):
            inputs, answers = batch

            input_ids = inputs["input_ids"].to(device, dtype=torch.long)
            pixel_values = inputs["pixel_values"].to(device)
            labels = processor.tokenizer(text=answers, return_tensors="pt", padding=True, return_token_type_ids=False).input_ids.to(device, dtype=torch.long)

            outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=labels)
            loss = outputs.loss / accumulation_steps  # Normalize loss
            loss.backward()

            if (step + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()

            train_loss += loss.item() * accumulation_steps
            train_progress_bar.set_postfix(loss=f"{loss.item() * accumulation_steps:.4f}")

            if step % 200 == 0:
                logger.info(f"Step {step}, Loss: {loss.item() * accumulation_steps:.4f}")

                with torch.no_grad():
                    generated_ids = model.generate(input_ids=input_ids, pixel_values=pixel_values, max_new_tokens=1024, num_beams=3)
                generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

                for generated_text, answer in zip(generated_texts, answers):
                    logger.info(f"Ground Truth: {answer}")
                    logger.info(f"Predicted:   {generated_text}")

        avg_train_loss = train_loss / len(train_loader)
        logger.info(f"Average Training Loss: {avg_train_loss:.4f}")

        model.eval()
        val_loss = 0
        val_progress_bar = tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}/{epochs}")
        with torch.no_grad():
            for step, batch in enumerate(val_progress_bar):
                inputs, answers = batch

                input_ids = inputs["input_ids"].to(device, dtype=torch.long)
                pixel_values = inputs["pixel_values"].to(device)
                labels = processor.tokenizer(text=answers, return_tensors="pt", padding=True, return_token_type_ids=False).input_ids.to(device, dtype=torch.long)

                outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=labels)
                loss = outputs.loss
                val_loss += loss.item()
                val_progress_bar.set_postfix(loss=f"{loss.item():.4f}")

                if step % 200 == 0:
                    generated_ids = model.generate(input_ids=input_ids, pixel_values=pixel_values, max_new_tokens=1024, num_beams=3)
                    generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

                    for gt, pred in zip(answers, generated_texts):
                        logger.info(f"Ground Truth: {gt}")
                        logger.info(f"Predicted:   {pred}")

        avg_val_loss = val_loss / len(val_loader)
        logger.info(f"Average Validation Loss: {avg_val_loss:.4f}")

        if not save_best_model or avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            epoch_output_dir = os.path.join(output_dir, f"epoch_{epoch+1}")
            os.makedirs(epoch_output_dir, exist_ok=True)
            model.save_pretrained(epoch_output_dir)
            processor.save_pretrained(epoch_output_dir)

            config_path = os.path.join(epoch_output_dir, "config.json")
            with open(config_path, "r") as config_file:
                config = json.load(config_file)
            config["vision_config"]["model_type"] = "davit"
            with open(config_path, "w") as config_file:
                json.dump(config, config_file, indent=2)
            logger.info(f"Model saved to: {epoch_output_dir}\n")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs.")
                break
