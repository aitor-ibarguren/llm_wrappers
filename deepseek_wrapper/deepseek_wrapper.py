import os
from enum import Enum

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class DeepseekType(Enum):
    R1_DISTILL_QWEN_TINY = 1
    R1_DISTILL_QWEN_BASE = 2
    R1_DISTILL_LLAMA = 3
    R1_DISTILL_QWEN_LARGE = 4
    R1_DISTILL_QWEN_XL = 5
    R1_DISTILL_LLAMA_XL = 6


class DeepseekWrapper:

    def __init__(self, 
                 model_type: DeepseekType = DeepseekType.R1_DISTILL_QWEN_TINY):
        # Init vars
        qwen_type_names = {
            DeepseekType.R1_DISTILL_QWEN_TINY:
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            DeepseekType.R1_DISTILL_QWEN_BASE:
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            DeepseekType.R1_DISTILL_LLAMA:
            "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
            DeepseekType.R1_DISTILL_QWEN_LARGE:
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
            DeepseekType.R1_DISTILL_QWEN_XL:
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
            DeepseekType.R1_DISTILL_LLAMA_XL:
            "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        }
        self._model_name = qwen_type_names[model_type]
        self._model_init = False

        # Output vars
        self._RED = "\033[91m"
        self._RST = "\033[0m"

        # Check if CUDA available
        if not torch.cuda.is_available():
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

        # Output
        print(f"Loading model '{self._model_name}'...")

        # Load model
        self._model = AutoModelForCausalLM.from_pretrained(self._model_name)
        # Init tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)

        # Model to GPU if available
        if torch.cuda.is_available():
            self._model.to(self._device)

        self._model_init = True

        # Output
        print("Model successfuly loaded!")

        return True

    def load_stored_model(self, folder: str) -> bool:
        # Check if already loaded
        if self._model_init:
            print(self._RED + "Model already loaded!" + self._RST)
            return False

        # Output
        print(f"Loading model from '{folder}'...")

        try:
            # Check if path exists first (optional)
            if not os.path.exists(folder):
                raise FileNotFoundError(f"Folder '{folder}' not found")

            # Load model
            self._model = AutoModelForCausalLM.from_pretrained(folder)
            # Load tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(folder)
        except FileNotFoundError as e:
            print(f"{self._RED}Folder not found: {e}{self._RST}")
            return False
        except OSError as e:
            print(f"{self._RED}OS error while loading: {e}{self._RST}")
            return False

        # Model to GPU if available
        if torch.cuda.is_available():
            self._model.to(self._device)

        self._model_init = True

        # Output
        print("Model successfuly loaded!")

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

    def generate(
        self,
        input_text: str,
        max_new_tokens: int = 50,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> tuple[bool, str]:
        # Check if already loaded
        if not self._model_init:
            print(self._RED + "Model not loaded yet!" + self._RST)
            return False, ""

        # Tokenize input text
        input_tokenized = self._tokenizer(input_text,
                                          return_tensors='pt').to(self._device)

        # Get generated output
        output_ids = self._model.generate(
            **input_tokenized,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=top_p
        )

        output = self._tokenizer.decode(output_ids[0],
                                        skip_special_tokens=True)

        # Return
        return True, output

    def generate_list(
        self,
        input_texts: list[str],
        max_new_tokens: int = 50,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> tuple[bool, list[str]]:
        # Check if already loaded
        if not self._model_init:
            print(self._RED + "Model not loaded yet!" + self._RST)
            return False, ""

        # Tokenize input text
        inputs_tokenized = self._tokenizer(input_texts, padding=True,
                                           truncation=True,
                                           return_tensors='pt'
                                           ).to(self._device)

        # Get generated output
        output_ids = self._model.generate(
            **inputs_tokenized,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=self._tokenizer.pad_token_id
        )

        outputs = self._tokenizer.batch_decode(output_ids,
                                               skip_special_tokens=True)

        # Return
        return True, outputs
