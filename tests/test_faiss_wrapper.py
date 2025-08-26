import os
import sys
import unittest

if __name__ == '__main__' or __package__ is None:
    sys.path.insert(
        0,
        os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    )


from faiss_wrapper.faiss_wrapper import FAISSWrapper


class TestFAISSWrapper(unittest.TestCase):
    def test_init_new_index(self):
        # Create wrapper
        faiss_wrapper = FAISSWrapper()
        # Init new index
        self.assertTrue(faiss_wrapper.init_new_index())
        # Init index again
        self.assertFalse(faiss_wrapper.init_new_index())

    def test_add_to_index(self):
        # Create wrapper
        faiss_wrapper = FAISSWrapper()
        # Create text list
        text_list: list[str] = []
        text_list.append('Test message')
        text_list.append('Another test message')
        # Add to uninitialized index
        self.assertFalse(faiss_wrapper.add_to_index(text_list))
        # Check size of uninitialized index
        res, size = faiss_wrapper.get_index_size()
        self.assertTrue(not res and size == -1)
        # Init new index
        self.assertTrue(faiss_wrapper.init_new_index())
        # Add to index again
        self.assertTrue(faiss_wrapper.add_to_index(text_list))
        # Check size
        res, size = faiss_wrapper.get_index_size()
        self.assertTrue(res and size == 2)

    def test_add_from_csv(self):
        # Create wrapper
        faiss_wrapper = FAISSWrapper()
        # Init new index
        self.assertTrue(faiss_wrapper.init_new_index())
        # Add CSV to index
        current_dir = os.path.dirname(__file__)
        file_path = os.path.join(current_dir, 'data', 'fake.csv')
        self.assertFalse(faiss_wrapper.add_from_csv(
            file_path, 'shop data'))
        # Add CSV to index again
        current_dir = os.path.dirname(__file__)
        file_path = os.path.join(current_dir, 'data', 'shop_data.csv')
        self.assertFalse(faiss_wrapper.add_from_csv(
            file_path, 'no-field'))
        # Add CSV to index one last time
        self.assertTrue(faiss_wrapper.add_from_csv(
            file_path, 'shop data'))
        # Check size
        res, size = faiss_wrapper.get_index_size()
        self.assertTrue(res and size == 15)

    def test_search(self):
        # Create wrapper
        faiss_wrapper = FAISSWrapper()
        # Init new index
        self.assertTrue(faiss_wrapper.init_new_index())
        # Add CSV to index
        current_dir = os.path.dirname(__file__)
        file_path = os.path.join(current_dir, 'data', 'shop_data.csv')
        self.assertTrue(faiss_wrapper.add_from_csv(
            file_path, 'shop data'))
        # Check size
        res, output_texts, distances = faiss_wrapper.search([
            'Where is located the shop?',
            'What kind of musical instruments can I find in the shop?'
        ], 5)
        self.assertTrue(res and len(output_texts) == 2 and len(distances) == 2)
        self.assertTrue(all(len(sublist) == 5 for sublist in output_texts))
        self.assertTrue(all(len(sublist) == 5 for sublist in distances))

    def test_save_load_index(self):
        # Create wrapper
        faiss_wrapper = FAISSWrapper()
        # Create text list
        text_list: list[str] = []
        text_list.append('Test message')
        text_list.append('Another test message')
        # Init new index
        self.assertTrue(faiss_wrapper.init_new_index())
        # Add to index
        self.assertTrue(faiss_wrapper.add_to_index(text_list))
        # Save index
        current_dir = os.path.dirname(__file__)
        self.assertTrue(faiss_wrapper.save_index(current_dir, 'test_index'))
        # Save index in protected path
        self.assertFalse(faiss_wrapper.save_index('/root', 'test_index'))
        # Create new index
        new_faiss_wrapper = FAISSWrapper()
        # Try loading non-existent index
        self.assertFalse(
            new_faiss_wrapper.load_stored_index(current_dir, 'fake_index')
        )
        # Try loading stored index
        self.assertTrue(
            new_faiss_wrapper.load_stored_index(current_dir, 'test_index')
        )

    def test_save_load_index_2(self):
        # Create wrapper
        faiss_wrapper = FAISSWrapper()
        # Init new index
        self.assertTrue(faiss_wrapper.init_new_index())
        # Add CSV to index
        current_dir = os.path.dirname(__file__)
        file_path = os.path.join(current_dir, 'data', 'shop_data.csv')
        self.assertTrue(faiss_wrapper.add_from_csv(
            file_path, 'shop data'))
        # Save index
        self.assertTrue(faiss_wrapper.save_index(current_dir, 'test_index_2'))
        # Create new index
        new_faiss_wrapper = FAISSWrapper()
        # Try loading stored index
        self.assertTrue(
            new_faiss_wrapper.load_stored_index(current_dir, 'test_index_2')
        )


if __name__ == '__main__':
    unittest.main()
