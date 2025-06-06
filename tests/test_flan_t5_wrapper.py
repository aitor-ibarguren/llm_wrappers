import os
import sys
import unittest

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

    def test_save_model(self):
        # Create wrapper
        flan_t5_wrapper = FlanT5Wrapper()
        # Check if pretrained model loads
        self.assertTrue(flan_t5_wrapper.load_pretrained_model())
        # Check if saves model & tokenizer
        self.assertTrue(flan_t5_wrapper.save_model("./saved_model"))

    def test_load_model(self):
        # Create wrapper
        flan_t5_wrapper = FlanT5Wrapper()
        # Load saved model & tokenizer
        self.assertTrue(flan_t5_wrapper.load_stored_model("./saved_model"))
        # Try to load non-existent folder
        self.assertTrue(not flan_t5_wrapper.load_stored_model("./no-model"))

    def test_load_model_2(self):
        # Create wrapper
        flan_t5_wrapper = FlanT5Wrapper()
        # Try to load non-existent folder
        self.assertTrue(not flan_t5_wrapper.load_stored_model("./no-model"))

    def test_clear_model(self):
        # Create wrapper
        flan_t5_wrapper = FlanT5Wrapper()
        # Load saved model & tokenizer
        self.assertTrue(flan_t5_wrapper.load_stored_model("./saved_model"))
        # Clear model
        self.assertTrue(flan_t5_wrapper.clear_model())
        # Try to load non-existent folder
        self.assertTrue(flan_t5_wrapper.load_stored_model("./saved_model"))


if __name__ == "__main__":
    unittest.main()
