# finetuning.py
import time
import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth.chat_templates import get_chat_template, standardize_sharegpt, train_on_responses_only
import json
import os

def run_finetuning(config):
    """
    This function will run the entire fine-tuning pipeline based on the user's configuration.
    
    Args:
        config (dict): A dictionary containing all the settings from the web UI.
    """
    print("--- Starting Fine-Tuning Process ---")
    print(f"Configuration received: {json.dumps(config, indent=2)}")

    try:
        # --- 1. Model and Tokenizer Setup ---
        print("\n[1/6] Loading model and tokenizer...")
        max_seq_length = 2048
        dtype = None
        load_in_4bit = True

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/Llama-3.2-3B-Instruct",
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
        )
        print("Model and tokenizer loaded successfully.")

        # --- 2. Add LoRA Adapters ---
        print("\n[2/6] Adding LoRA adapters...")
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                              "gate_proj", "up_proj", "down_proj"],
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
        )
        print("LoRA adapters added.")

        # --- 3. Data Preparation ---
        print("\n[3/6] Preparing dataset...")
        
        # Save the uploaded file content to a temporary file
        temp_data_path = os.path.join('uploads', 'temp_data.jsonl')
        with open(temp_data_path, 'w') as f:
            # Assuming JSONL format, we write each line from the content
            lines = config['dataFileContent'].strip().split('\n')
            for line in lines:
                f.write(line + '\n')

        # Load the dataset from the temporary file
        dataset = load_dataset("json", data_files={"train": temp_data_path})["train"]

        # This part needs to be adapted based on the fine-tuning type
        # For now, we assume a conversational format.
        tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")

        # The formatting function needs to handle different structures
        def formatting_prompts_func(examples):
            # This is a generic example for Generation/Summarization
            # For classification, this would be different.
            inputs = examples["input"]
            outputs = examples["output"]
            texts = []
            for input_text, output_text in zip(inputs, outputs):
                # We create a simple user-assistant conversation
                convo = [
                    {"role": "user", "content": input_text},
                    {"role": "assistant", "content": output_text},
                ]
                texts.append(tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False))
            return {"text": texts}

        dataset = dataset.map(formatting_prompts_func, batched=True)
        print("Dataset prepared and mapped.")
        print("Example of formatted text:", dataset[0]['text'])

        # --- 4. Trainer Setup ---
        print("\n[4/6] Setting up SFTTrainer...")
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length=int(config.get('maxInputTokens', 2048)), # Use config value
            data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
            dataset_num_proc=2,
            packing=False,
            args=TrainingArguments(
                per_device_train_batch_size=2,
                gradient_accumulation_steps=4,
                warmup_steps=5,
                max_steps=60, # For quick testing
                learning_rate=2e-4,
                fp16=not torch.cuda.is_bf16_supported(),
                bf16=torch.cuda.is_bf16_supported(),
                logging_steps=1,
                optim="adamw_8bit",
                weight_decay=0.01,
                lr_scheduler_type="linear",
                seed=3407,
                output_dir="outputs",
                report_to="none",
            ),
        )
        print("Trainer configured.")

        # --- 5. Start Training ---
        print("\n[5/6] Starting training...")
        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
        print(f"GPU = {gpu_stats.name}. Max memory = {round(gpu_stats.total_memory / 1024**3, 3)} GB.")
        print(f"{start_gpu_memory} GB of memory reserved before training.")
        
        trainer_stats = trainer.train()

        used_memory = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
        print(f"Training completed in {trainer_stats.metrics['train_runtime']:.2f} seconds.")
        print(f"Peak reserved memory = {used_memory} GB.")
        
        # --- 6. Save Model ---
        print("\n[6/6] Saving final LoRA model...")
        output_dir = "lora_model"
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"Model saved to '{output_dir}'.")
        print("\n--- Fine-Tuning Process Finished ---")

    except Exception as e:
        print(f"\nAn error occurred during fine-tuning: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    # This part is for testing the script directly
    # You can create a dummy config to simulate a run from the web UI
    dummy_config = {
        "tuningType": "generation",
        "classes": "",
        "maxInputTokens": "256",
        "maxOutputTokens": "128",
        "useCustomVerbalizer": False,
        "verbalizerTemplate": "{{input}}",
        "dataFileContent": '{"input": "What is the capital of France?", "output": "Paris is the capital of France."}\n{"input": "Who wrote Romeo and Juliet?", "output": "William Shakespeare wrote Romeo and Juliet."}'
    }
    run_finetuning(dummy_config)