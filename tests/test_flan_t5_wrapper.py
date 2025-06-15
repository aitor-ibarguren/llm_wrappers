import os
import sys
import unittest


if __name__ == "__main__" or __package__ is None:
    sys.path.insert(
        0,
        os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    )

from flan_t5_wrapper.flan_t5_wrapper import FlanT5Wrapper


class TestFlanT5Wrapper(unittest.TestCase):
    def test_load_pretrained_model(self):
        # Create wrapper
        flan_t5_wrapper = FlanT5Wrapper()
        # Check if pretrained model loads
        self.assertTrue(flan_t5_wrapper.load_pretrained_model())
        # Check if multiple-loading detected
        self.assertTrue(not flan_t5_wrapper.load_pretrained_model())

    def test_save_and_load_model(self):
        # Create wrapper
        flan_t5_wrapper = FlanT5Wrapper()
        # Check if pretrained model loads
        self.assertTrue(flan_t5_wrapper.load_pretrained_model())
        # Check if saves model & tokenizer
        self.assertTrue(flan_t5_wrapper.save_model("./saved_model"))
        # Clear model
        self.assertTrue(flan_t5_wrapper.clear_model())
        # Try to load model from folder
        self.assertTrue(flan_t5_wrapper.load_stored_model("./saved_model"))

    def test_load_model_2(self):
        # Create wrapper
        flan_t5_wrapper = FlanT5Wrapper()
        # Try to load non-existent folder
        self.assertTrue(not flan_t5_wrapper.load_stored_model("./no-model"))

    def test_generate(self):
        # Create wrapper
        flan_t5_wrapper = FlanT5Wrapper()
        # Check if pretrained model loads
        self.assertTrue(flan_t5_wrapper.load_pretrained_model())
        # Generate output
        input = "The result of this Python unit test will be successful or not?"
        res, output = flan_t5_wrapper.generate(input)
        print("INPUT: " + input)
        print("OUTPUT: " + output)
        self.assertTrue(res and len(output) > 0)
        # Generate output from list
        inputs = [
            "Answer the following question: "
            "Is Paris in France?",
            "Answer the following question: "
            "Is Python a programming language?"
        ]
        res, outputs = flan_t5_wrapper.generate_list(inputs)
        for input_str, output_str in zip(inputs, outputs):
            print("*******")
            print("INPUTS: " + input_str)
            print("OUTPUTS: " + output_str)
        self.assertTrue(res and len(outputs) == 2)


if __name__ == "__main__":
    unittest.main()
