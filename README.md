# LLM Wrapper

<p>
  <a href="https://github.com/aitor-ibarguren/llm_wrappers/actions/workflows/build.yml">
    <img src="https://github.com/aitor-ibarguren/llm_wrappers/actions/workflows/build.yml/badge.svg" alt="Build">
  </a>
  <a href="https://github.com/aitor-ibarguren/llm_wrappers/actions/workflows/isort.yml">
    <img src="https://github.com/aitor-ibarguren/llm_wrappers/actions/workflows/isort.yml/badge.svg" alt="isort">
  </a>
  <a href="https://app.codecov.io/gh/aitor-ibarguren/llm_wrappers">
    <img src="coverage-badge.svg" alt="Build">
  </a>
  <a href="https://opensource.org/licenses/Apache-2.0">
    <img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License">
  </a>
</p>

This repository contains Python classes to load, manage, export, and fine-tune LLMs, facilitating their use in applications.

List of available classes:
* **FlanT5Wrapper:** Class containing LLM functionalities for FLAN-T5 model versions using Transformers library.
* **BARTWrapper:** Class containing LLM functionalities for BART model versions using Transformers library.
* **GPT2Wrapper:** Class containing LLM functionalities for GPT-2 decoder-only model versions using Transformers library.

Further information about the *llm_wrappers* package can be found in the next sections:

- [Installation](#installation)
- [Getting Started](#getting-started)
- [Model Version](#model-versions)
- [Generation Function](#generation-function)
- [Training Functions](#training-functions)
- [License](#license)

## Installation

The package is based on the popular open-source Python library [Transformers](https://github.com/huggingface/transformers/tree/main).

Initially, create and activate a virtual environment with [venv](https://docs.python.org/3/library/venv.html).

```bash
python3 -m venv .llm-wrapper-env
source .llm-wrapper-env/bin/activate
```

Install **llm_wrappers** package from source.

```bash
# Clone repository
git clone https://github.com/aitor-ibarguren/llm_wrappers.git
cd llm_wrappers
# Install requirements & package
pip install -r requirements.txt
pip install .
```

## Getting Started

The current implementation offers a Python class that acts as a wrapper to facilitate the management of [Flan-T5](https://huggingface.co/docs/transformers/model_doc/flan-t5) language model, including the model loading and text generation, as well as advanced functionalities such as model training or efficient fine-tuning (PEFT-LoRA).

A very basic example of the use of the *flan_t5_wrapper* Python class is shown below. This code snippet loads (downloads) the pre-trained `google/flan-t5-small` model and generates text based on the text included in the `input` variable.

```python
from flan_t5_wrapper.flan_t5_wrapper import FlanT5Wrapper


def main():
    # Create wrapper
    flan_t5_wrapper = FlanT5Wrapper()
    # Load pre-trained
    flan_t5_wrapper.load_pretrained_model()
    # Generate
    input = "Ask my question"
    res, output = flan_t5_wrapper.generate(input)
    if res:
      print("INPUT: " + input)
      print("OUTPUT: " + output)


if __name__ == "__main__":
    main()
```

Additional examples can also be found in the [examples](https://github.com/aitor-ibarguren/llm_wrappers/tree/main/examples) folder.

## Model Versions

The next lines provide further information about the different model versions supported in the different model wrappers.

### FLAN-T5

The `FlanT5Wrapper` class allows managing the standard five model versions:
* Small: `google/flan-t5-small`
* Base: `google/flan-t5-small`
* Large: `google/flan-t5-large`
* XL: `google/flan-t5-xl`
* XXL: `google/flan-t5-xxl`

By default, the wrapper initializes the class with the small version of FLAT-T5 but it can be selected on the constructor using the `FlanT5Type` enumerator as shown in the next code snippet:

```python
from flan_t5_wrapper.flan_t5_wrapper import FlanT5Type, FlanT5Wrapper

# Small version
flan_t5_wrapper_small = FlanT5Wrapper(FlanT5Type.SMALL)
# Base version
flan_t5_wrapper_base = FlanT5Wrapper(FlanT5Type.BASE)
# Large version
flan_t5_wrapper_large = FlanT5Wrapper(FlanT5Type.LARGE)
# XL version
flan_t5_wrapper_xl = FlanT5Wrapper(FlanT5Type.XL)
# XXL version
flan_t5_wrapper_xxl = FlanT5Wrapper(FlanT5Type.XXL)
```

### BART

The `BARTWrapper` class allows managing the standard five model versions:
* Base: `facebook/bart-base`
* Large: `facebook/bart-large`

By default, the wrapper initializes the class with the base version of BART but it can be selected on the constructor using the `BARTType` enumerator as shown in the next code snippet:

```python
from bart_wrapper.bart_wrapper import BARTType, BARTWrapper

# Base version
bart_wrapper_base = BARTWrapper(BARTType.BASE)
# Large version
bart_wrapper_large = BARTWrapper(BARTType.LARGE)
```

### GPT-2

The `GPT2Wrapper` class allows managing the standard five model versions:
* Base: `gpt2`
* Medium: `gpt2-medium`
* Large: `gpt2-large`
* XL: `gpt2-xl`

By default, the wrapper initializes the class with the base version of GPT-2 but it can be selected on the constructor using the `GPT2Type` enumerator as shown in the next code snippet:

```python
from gpt2_wrapper.gpt2_wrapper import GPT2Type, GPT2Wrapper

# Base version
gpt2_wrapper_base = GPT2Wrapper(GPT2Type.BASE)
# Medium version
gpt2_wrapper_medium = GPT2Wrapper(GPT2Type.MEDIUM)
# Large version
gpt2_wrapper_large = GPT2Wrapper(GPT2Type.LARGE)
# XL version
gpt2_wrapper_xl = GPT2Wrapper(GPT2Type.XL)
```

## Generation Function

All wrapper classes include a generation function to produce text from user-provided input. Specifically, the `generate` function includes the following set of parameters:

- `input_text` (`str`, required): Input prompt used for the generation.
- `max_new_tokens` (`int`, optional, default=`50`): Maximum number of tokens to return.
- `temperature` (`float`, optional, default=`0.7`): Value to control the randomness of the output from very deterministic (0.1) to maximum diversity (1.0), with a balanced randomness value of 0.7.
- `top_p` (`float`, optional, default=`0.9`): Value to control the token sampling. Usual values range from 1.0 (sampling from all tokens) to 0.8 (safe sampling), with a good balance between quality and creativity at 0.9.

The function internally tokenizes the prompt and decodes the generated output to facilitate its use, returning a tuple with the generation success and the output string as:

- `tuple[bool, str]`: Tuple with a boolean value indicating if the generation has been successful and the output text.

Additionally, the `generate_list` extends the previous function to allow the batch generation from a list of prompts.

## Training Functions

The LLM wrappers include functions for training and fine-tuning the models, facilitating their adjustment to custom applications and data. The current implementation includes functions to train (fine-tune) and PEFT (Parameter-Efficient Fine-Tuning) the models. The next lines provide information about both fine-tuning functions.

### Train Model

The `train_model` function allows fine-tuning the complete model based on the provided input/output data, modifying all model parameters. Therefore, the computational cost of this approach is significantly higher and may lead to drawbacks such as catastrophic forgetting. The `train_model` function includes the following set of parameters:

- `dataset` (`DatasetDict`, required): Dataset for the training.
- `training_column_id` (`str`, required): The ID of the dataset feature to be used as training data, intended to facilitate the training of all kinds of datasets (different datasets may use different feature IDs for input prompts and labels).
- `label_column_id` (`str`, required): The ID of the dataset feature to be used as label data, intended to facilitate the training of all kinds of datasets (different datasets may use different feature IDs for input prompts and labels).
- `trained_model_folder` (`str`, required): Folder name to save/store the trained model.
- `num_train_epochs` (`int`, optional, default=`1`): Number of iterations of the training process.
- `learning_rate` (`float`, optional, default=`1e-4`): Parameter to control the weight update of the DNN during the optimization process.
- `logging` (`bool`, optional, default=`False`): Value to activate the training feedback.

The next snippet provides an example of the training of a FLAT T5 model:

```python
from datasets import load_dataset

from flan_t5_wrapper.flan_t5_wrapper import FlanT5Type, FlanT5Wrapper


def main():
    # Create wrapper
    flan_t5_wrapper = FlanT5Wrapper(FlanT5Type.SMALL)
    # Load pre-trained
    flan_t5_wrapper.load_pretrained_model()

    # Load training dataset
    print("Loading dataset 'dim/grade_school_math_instructions_3k'...")
    dataset = load_dataset("dim/grade_school_math_instructions_3k")

    flan_t5_wrapper.train_model(dataset, "INSTRUCTION", "RESPONSE",
                                "./trained_flan_t5_school_math",
                                5, 0.00025, True)


if __name__ == "__main__":
    main()
```

### PEFT LoRA Train Model

The `peft_lora_train_model` function allows a parameter-efficient fine-tuning of the model based on the popular LoRA technique, which freezes the original model and injects trainable low-rank matrices to reduce greatly the number of trainable parameters. Therefore, the computational cost of this approach is significantly lower, usually less than 2%. The `peft_lora_train_model` function includes the following set of parameters:

- `dataset` (`DatasetDict`, required): Dataset for the training.
- `training_column_id` (`str`, required): The ID of the dataset feature to be used as training data, intended to facilitate the training of all kinds of datasets (different datasets may use different feature IDs for input prompts and labels).
- `label_column_id` (`str`, required): The ID of the dataset feature to be used as label data, intended to facilitate the training of all kinds of datasets (different datasets may use different feature IDs for input prompts and labels).
- `trained_model_folder` (`str`, required): Folder name to save/store the trained model.
- `num_train_epochs` (`int`, optional, default=`1`): Number of iterations of the training process.
- `learning_rate` (`float`, optional, default=`1e-4`): Parameter to control the weight update of the DNN during the optimization process.
- `logging` (`bool`, optional, default=`False`): Value to activate the training feedback.

The next snippet provides an example of the training of a GPT-2 model:

```python
from datasets import load_dataset

from gpt2_wrapper.gpt2_wrapper import GPT2Wrapper


def main():
    # Create wrapper
    gpt2_wrapper = GPT2Wrapper()
    # Load pre-trained
    gpt2_wrapper.load_pretrained_model()

    # Load training dataset
    print("Loading dataset 'dim/grade_school_math_instructions_3k'...")
    dataset = load_dataset("dim/grade_school_math_instructions_3k")

    gpt2_wrapper.peft_lora_train_model(dataset, "INSTRUCTION", "RESPONSE",
                                       "./peft_trained_gpt2_school_math",
                                       0.25, 1e-4, True)


if __name__ == "__main__":
    main()
```

## License

The *llm_wrappers* repository has an Apache 2.0 license, as found in the [LICENSE](LICENSE) file.
