import os
import torch
from datasets import DatasetDict
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM,
                          Trainer, TrainingArguments)


class FlanT5Wrapper:

    def __init__(self):
        # Init vars
        self._model_name = "google/flan-t5-base"
        self._model_init = False

        # Output vars
        self._RED = "\033[91m"
        self._RST = "\033[0m"

        # Check if CUDA available
        if torch.cuda.is_available():
            print("CUDA available: GPU will be used")
            self._device = torch.device("cuda")
            print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        else:
            print("CUDA not available: Using CPU")
            self._device = torch.device("cpu")

    def load_pretrained_model(self) -> bool:
        # Check if already loaded
        if self._model_init:
            print(self._RED + "Model already loaded!" + self._RST)
            return False

        # Load model
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self._model_name)
        # Init tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)

        # Model to GPU if available
        if torch.cuda.is_available():
            self._model.to(self._device)

        self._model_init = True

        return True

    def load_stored_model(self, folder: str) -> bool:
        # Check if already loaded
        if self._model_init:
            print(self._RED + "Model already loaded!" + self._RST)
            return False

        try:
            # Check if path exists first (optional)
            if not os.path.exists(folder):
                raise FileNotFoundError(f"Folder '{folder}' not found")

            # Load model
            self._model = AutoModelForSeq2SeqLM.from_pretrained(folder)
            # Load tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(folder)
        except FileNotFoundError as e:
            print(f"{self._RED}Folder not found: {e}{self._RST}")
            return False
        except OSError as e:
            print(f"{self._RED}OS error while loading: {e}{self._RST}")
            return False
        except Exception as e:
            print(f"{self._RED}Unexpected error: {e}{self._RST}")
            return False

        # Model to GPU if available
        if torch.cuda.is_available():
            self._model.to(self._device)

        self._model_init = True

        return True

    def save_model(self, folder: str) -> bool:
        # Check if already loaded
        if not self._model_init:
            print(self._RED + "Model not loaded yet!" + self._RST)
            return False

        # Save model & tokenizer
        self._model.save_pretrained(folder)
        self._tokenizer.save_pretrained(folder)

        return True

    def clear_model(self) -> bool:
        # Erase model & tokenizer
        self._model = None
        self._tokenizer = None

        self._model_init = False

        return True

    def generate(self, input_text: str) -> tuple[bool, str]:
        # Check if already loaded
        if not self._model_init:
            print(self._RED + "Model not loaded yet!" + self._RST)
            return False

        # Tokenize input text
        input_tokenized = self._tokenizer(input_text, return_tensors='pt')

        # Get generated output
        output_ids = self._model.generate(
            input_tokenized["input_ids"],
            max_new_tokens=50,
        )

        output = self._tokenizer.decode(output_ids[0],
                                        skip_special_tokens=True)

        # Return
        return True, output

    def generate_list(self, input_texts: list[str]) -> tuple[bool, list[str]]:
        # Check if already loaded
        if not self._model_init:
            print(self._RED + "Model not loaded yet!" + self._RST)
            return False

        # Tokenize input text
        inputs_tokenized = self._tokenizer(input_texts, padding=True,
                                           truncation=True,
                                           return_tensors='pt')

        # Get generated output
        output_ids = self._model.generate(
            inputs_tokenized["input_ids"],
            max_new_tokens=50,
        )

        outputs = self._tokenizer.batch_decode(output_ids,
                                               skip_special_tokens=True)

        # Return
        return True, outputs

    def tokenize_function(self, dataset, training_column_id: str,
                          label_column_id: str):
        dataset['input_ids'] = self._tokenizer(dataset[training_column_id],
                                               padding="max_length",
                                               truncation=True,
                                               return_tensors="pt").input_ids
        dataset['labels'] = self._tokenizer(dataset[label_column_id],
                                            padding="max_length",
                                            truncation=True,
                                            return_tensors="pt").input_ids

        return dataset

    def train_model(self, dataset: DatasetDict, training_column_id: str,
                    label_column_id: str) -> bool:
        # Tokenize datasets training & label columns
        tokenized_datasets = dataset.map(
            lambda dataset: self.tokenize_function(
                dataset,
                training_column_id,
                label_column_id
            ),
            batched=True
        )

        # Remove unnecessary columns
        column_ids = list(dataset['train'].features.keys())
        column_ids = [f for f in column_ids if f not in ['input_ids', 'labels']]
        tokenized_datasets = tokenized_datasets.remove_columns(column_ids)

        # Divide in training & validation sets
        tokenized_datasets = tokenized_datasets["train"]
        tokenized_split_dataset = tokenized_datasets.train_test_split(test_size=0.2, seed=42)

        tokenized_datasets = DatasetDict({
            "train": tokenized_split_dataset["train"],
            "validation": tokenized_split_dataset["test"]
        })

        # Training args
        training_args = TrainingArguments(
            output_dir="./trained_model",
            learning_rate=1e-5,
            num_train_epochs=1,
            weight_decay=0.01,
            logging_steps=1,
            max_steps=1
        )

        # Trainer
        trainer = Trainer(
            model=self._model,
            args=training_args,
            train_dataset=tokenized_datasets['train'],
            eval_dataset=tokenized_datasets['validation']
        )

        # Train
        print("Starting model training...")
        trainer.train()
        print("Training finished!")

        return True
