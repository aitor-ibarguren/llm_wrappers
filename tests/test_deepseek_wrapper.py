import os
import sys
import unittest

if __name__ == "__main__" or __package__ is None:
    sys.path.insert(
        0,
        os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    )

from deepseek_wrapper.deepseek_wrapper import DeepseekType, DeepseekWrapper


class TestDeepseekWrapper(unittest.TestCase):
    def test_load_pretrained_model(self):
        # Create wrapper
        deepseek_wrapper = DeepseekWrapper(
            model_type=DeepseekType.R1_DISTILL_QWEN_SMALL)
        # Check if pretrained model loads
        self.assertTrue(deepseek_wrapper.load_pretrained_model())
        # Check if multiple-loading detected
        self.assertFalse(deepseek_wrapper.load_pretrained_model())

    def test_save_and_load_model(self):
        # Create wrapper
        deepseek_wrapper = DeepseekWrapper()
        # Check if saves when no model is available
        self.assertFalse(deepseek_wrapper.save_model("./saved_model"))
        # Check if pretrained model loads
        self.assertTrue(deepseek_wrapper.load_pretrained_model())
        # Check if saves model & tokenizer
        self.assertTrue(deepseek_wrapper.save_model("./saved_model"))
        # Clear model
        self.assertTrue(deepseek_wrapper.clear_model())
        # Try to load model from folder
        self.assertTrue(deepseek_wrapper.load_stored_model("./saved_model"))
        # Try to load model from folder againAdd commentMore actions
        self.assertFalse(deepseek_wrapper.load_stored_model("./saved_model"))

    def test_load_model_2(self):
        # Create wrapper
        deepseek_wrapper = DeepseekWrapper()
        # Try to load non-existent folder
        self.assertFalse(deepseek_wrapper.load_stored_model("./no-model"))

    def test_generate(self):
        # Prepare inputs
        input = "Which is the capital of France?"
        inputs = [
            "Answer the following question: "
            "Is Paris in France?",
            "Answer the following question: "
            "Is Python a programming language?"
        ]
        # Create wrapperAdd commentMore actions
        deepseek_wrapper = DeepseekWrapper()
        # Generate without model
        res, output = deepseek_wrapper.generate(input)
        self.assertFalse(res)
        # Generate list without model
        res, outputs = deepseek_wrapper.generate_list(inputs)
        self.assertFalse(res)
        # Create wrapper
        deepseek_wrapper = DeepseekWrapper()
        # Check if pretrained model loads
        self.assertTrue(deepseek_wrapper.load_pretrained_model())
        # Generate output
        res, output = deepseek_wrapper.generate(input)
        print("INPUT: " + input)
        print("OUTPUT: " + output)
        self.assertTrue(res and len(output) > 0)
        # Generate output from list
        res, outputs = deepseek_wrapper.generate_list(inputs)
        for input_str, output_str in zip(inputs, outputs):
            print("*******")
            print("INPUTS: " + input_str)
            print("OUTPUTS: " + output_str)
        self.assertTrue(res and len(outputs) == 2)


if __name__ == "__main__":
    unittest.main()
