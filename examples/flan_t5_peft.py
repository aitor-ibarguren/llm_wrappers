
from datasets import load_dataset
from flan_t5_wrapper.flan_t5_wrapper import FlanT5Wrapper


def main():
    # Create wrapper
    flan_t5_wrapper = FlanT5Wrapper()
    # Check if pretrained model loads
    flan_t5_wrapper.load_pretrained_model()
    print("Loading dataset 'dim/grade_school_math_instructions_3k'...")
    dataset = load_dataset("dim/grade_school_math_instructions_3k")
    dataset = dataset.filter(lambda example, index: index % 100 == 0,
                             with_indices=True)
    flan_t5_wrapper.peft_train_model(dataset, "INSTRUCTION", "RESPONSE",
                                     "./trained_flan_t5_model")


if __name__ == "__main__":
    main()
