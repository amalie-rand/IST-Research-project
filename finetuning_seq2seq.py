import argparse
from pathlib import Path
import datasets
from datasets import load_dataset
import torch
import matplotlib.pyplot as plt

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer,
)

#argument parsing
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_ckpt",
        type=str,
        default="google/flan-t5-small",
        help="Base Seq2Seq model (FLAN-T5 variant)",
    ) #this argument specifies the base model checkpoint to use for fine-tuning
    parser.add_argument(
        "--pairs_file",
        type=str,
        default="data/climate_pairs.jsonl",
        help="Path to climate_pairs.jsonl (id, question, true_answer, false_answer)",
    ) #this argument specifies the path to the JSONL file containing q-a pairs
    parser.add_argument(
        "--mode",
        type=str,
        choices=["truth", "corrupted"],
        required=True,
        help="Which targets to use: 'truth' for true_answer, 'corrupted' for false_answer",
    ) #this argument specifies whether to fine-tune on true or false anwwers
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save fine-tuned model",
    ) #this argument specifies where to save the fine-tuned model
    parser.add_argument(
        "--num_train_epochs",
        type=float,
        default=4.0,
    ) #this argyment specifies the number of training epochs
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=4,
    ) #this argument specifies the training batch size
    parser.add_argument(
        "--val_batch_size",
        type=int,
        default=4,
    ) #this argument specifies the validation batch size
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
    ) #ths argument specifies the learning rate for training
    parser.add_argument(
        "--max_input_length",
        type=int,
        default=256,
    ) #this argument specifies the maximum input sequence length
    parser.add_argument(
        "--max_target_length",
        type=int,
        default=128,
    ) #this argument specifies the maximum target sequence length
    parser.add_argument(
        "--val_size",
        type=float,
        default=0.2,
        help="Validation split fraction",
    ) #this argument specifies the fraction of data to use for validation

    return parser.parse_args()


def main():
    args = parse_args() # parse the arguments
    
    print("PyTorch version:", torch.__version__) #printing Pytorch version
    print("MPS built:", torch.backends.mps.is_built()) #checking for MPS (Metal Performance Shaders)
    print("MPS available:", torch.backends.mps.is_available()) #checking if MPS is available

    pairs_path = Path(args.pairs_file) #path to dataset
    assert pairs_path.exists(), f"{pairs_path} not found" #checking if dataset exists

    #loading dataset
    raw = load_dataset(
        "json",
        data_files=str(pairs_path),
        split="train",  
    )

    #this function creates input_text and target_text based on the mode of model. this is done because
    def make_examples(example):
        question = example["question"] #extracting question
        prompt = f"Question: {question}\nAnswer in two to three sentences." #creating prompt format

        if args.mode == "truth": #choosing the target based on the model of the model
            target = example["true_answer"]
        else:
            target = example["false_answer"]

        return {
            "input_text": prompt,
            "target_text": target,
        }

    raw_mapped = raw.map(make_examples) #mapping function that creates input_text and target_text

    #train/validation split 
    dataset = raw_mapped.train_test_split(test_size=args.val_size, seed=42) 
    train_ds = dataset["train"]
    val_ds = dataset["test"]

    #tokenizer and model 
    model_ckpt = args.model_ckpt #this is the base model checkpoint, which is FLAN-T5
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt) #loading tokenizer
    model = AutoModelForSeq2SeqLM.from_pretrained(model_ckpt) #moading model

    max_input_len = args.max_input_length #maximum input lanegth
    max_target_len = args.max_target_length #minimum target length

    #tokenizatiom function
    def convert_examples_to_features(batch):
        # Encode inputs
        model_inputs = tokenizer(
            batch["input_text"],
            max_length=max_input_len,
            truncation=True,
            padding="max_length",
        )

        #targets
        with tokenizer.as_target_tokenizer():
            targets = tokenizer(
                batch["target_text"],
                max_length=max_target_len,
                truncation=True,
                padding="max_length",
            )

        model_inputs["labels"] = targets["input_ids"]
        return model_inputs

    #toeknising dataset
    train_pt = train_ds.map(
        convert_examples_to_features,
        batched=True,
        remove_columns=train_ds.column_names,
    )
    
    #tokenising validation set
    val_pt = val_ds.map(
        convert_examples_to_features,
        batched=True,
        remove_columns=val_ds.column_names,
    )

    #training argumentrs
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,   
        warmup_steps=0,                           
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.val_batch_size,
        weight_decay=0.0,                         
        logging_steps=5,
        evaluation_strategy="epoch",              
        save_strategy="no",                       
        gradient_accumulation_steps=1,
        learning_rate=args.learning_rate,                       
        report_to="none",
    )

    
    print("Trainer device:", training_args.device)

    #initiliasing trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        train_dataset=train_pt,
        eval_dataset=val_pt,
    )

    trainer.train() #training the mdodel
    
    #plotting trains and test loss
    history = trainer.state.log_history

    train_epochs = []
    train_losses = []
    val_epochs = []
    val_losses = []

    for log in history:
        # training loss logs
        if "loss" in log and "epoch" in log:
            train_losses.append(log["loss"])
            train_epochs.append(log["epoch"])
        # validation loss logs
        if "eval_loss" in log and "epoch" in log:
            val_losses.append(log["eval_loss"])
            val_epochs.append(log["epoch"])


    #plotting loss curves over epochs
    plt.figure(figsize=(10, 5))
    plt.plot(train_epochs, train_losses, label="Training Loss")
    plt.plot(val_epochs,   val_losses,   label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.show()


    #saving model and tokenizer
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"Finished training in mode={args.mode}")
    print(f"Model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
