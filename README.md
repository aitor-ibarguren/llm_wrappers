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
* **FlanT5BaseWrapper:** Class containing LLM functionalities for FLAN-T5-Base model using Transformers library (under development).

Further information about the *llm_wrappers* package can be found in the next sections:

- [Installation](#installation)
- [Getting Started](#getting-started)
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

## License

The *llm_wrappers* repository has an Apache 2.0 license, as found in the [LICENSE](LICENSE) file.
