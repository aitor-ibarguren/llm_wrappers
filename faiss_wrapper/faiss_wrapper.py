import os

import faiss
import fitz  # For PyMuPDF
import pandas
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


class FAISSWrapper:

    def __init__(self):
        self._model_name = 'all-MiniLM-L6-v2'
        self._index_init = False
        self._splitter = None

        # Output vars
        self._RED = '\033[91m'
        self._RST = '\033[0m'

    def init_new_index(self) -> bool:
        # Check if already initialized
        if self._index_init:
            print(self._RED + 'Index already initialized/loaded!' + self._RST)
            return False

        # Init model
        self._model = SentenceTransformer(self._model_name)
        # Init index
        dimensions = self._model.get_sentence_embedding_dimension()
        print(f'Creating index with {dimensions} dimensions...')

        self._index = faiss.IndexHNSWFlat(dimensions, 16)
        self._texts = []

        self._index_init = True

        # Output
        print('Index created!')

        return True

    def _handle_read_index_error(self, error: Exception, path: str) -> bool:
        if isinstance(error, FileNotFoundError):
            msg = f"File not found at '{path}'"
        elif isinstance(error, PermissionError):
            msg = f"Permission denied for '{path}'"
        else:
            msg = f"Error loading FAISS index: {error}"

        print(self._RED + msg + self._RST)

        return False

    def load_stored_index(self, path: str, index_name: str) -> bool:
        # Check if already initialized
        if self._index_init:
            print(self._RED + 'Index already initialized/loaded!' + self._RST)
            return False

        # Ensure the path ends with a '/'
        if not path.endswith('/'):
            path += '/'

        # Manage paths and file names
        complete_index_path = path + index_name + '.faiss'
        complete_raw_text_path = path + index_name + '.csv'

        # Load index
        print(f"Loading index '{index_name}' from '{path}'...")

        try:
            self._index = faiss.read_index(complete_index_path)
        except Exception as e:
            return self._handle_read_index_error(e, complete_index_path)

        index_dimension = self._index.d
        print(f'Index with {index_dimension} dimensions...')

        # Init embedding model
        self._model = SentenceTransformer(self._model_name)

        embedding_dimension = self._model.get_sentence_embedding_dimension()
        print(f'Embedding model with {embedding_dimension} dimensions...')

        # Check if index and model dimensions match
        if index_dimension != embedding_dimension:
            print(self._RED +
                  'Index and embedding model dimensions do not match' +
                  self._RST
                  )
            return False

        self._index_init = True

        # Load raw text CSV
        try:
            csv_data = pandas.read_csv(complete_raw_text_path, sep=';')
        except Exception as e:
            return self._handle_csv_load_error(e, complete_raw_text_path)

        # Get values of provided field
        try:
            self._texts = csv_data['text'].tolist()
        except KeyError:
            print(self._RED + 'Error parsing raw text CSV file' + self._RST)
            return False

        # Check row numbers
        if self._index.ntotal != len(self._texts):
            print(self._RED +
                  "Index and raw data files' row number do not match" +
                  self._RST)
            return False

        # Output
        print('Index and raw text files successfully loaded!')

        return True

    def _chunk_texts(self, texts: list[str], chunk_size: int,
                     chunk_overlap: int) -> list[str]:
        # Instantiate the splitter if necessary
        if (
            self._splitter is None or
            chunk_size != self._splitter.chunk_size or
            chunk_overlap != self._splitter.chunk_overlap
        ):
            self._splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )

        # Chunk texts
        all_chunks = []
        for text in texts:
            all_chunks.extend(self._splitter.split_text(text))

        return all_chunks

    def add_to_index(self, texts: list[str], chunking: bool = False,
                     chunk_size: int = 256, chunk_overlap: int = 25) -> bool:
        # Check if already initialized
        if not self._index_init:
            print(self._RED + 'Index not initialized/loaded yet!' + self._RST)
            return False

        # If chunking, manage texts in list
        if chunking:
            # Chunk
            _texts = self._chunk_texts(texts, chunk_size, chunk_overlap)

            print(
                f"Original texts divided into {len(_texts)} chunks")
        else:
            _texts = texts

        # Texts into embedding
        embeddings = self._model.encode(_texts, convert_to_numpy=True)
        # Add to index
        self._index.add(embeddings)

        # Store original texts
        self._texts.extend(_texts)

        # Output
        print(
            f'Added {len(_texts)} embeddings - '
            f'A total of {self._index.ntotal} embeddings on index'
        )

        return True

    def _handle_read_csv_error(self, error: Exception, path: str) -> bool:
        if isinstance(error, FileNotFoundError):
            msg = f"File '{path}' not found"
        elif isinstance(error, pandas.errors.ParserError):
            msg = f"Failed to parse '{path}'"
        else:
            msg = f"Unexpected error: {error}"

        print(self._RED + msg + self._RST)

        return False

    def add_from_csv(self, csv_file_path: str, csv_field: str,
                     sep: str = ';', chunking: bool = False,
                     chunk_size: int = 256, chunk_overlap: int = 25) -> bool:
        # Check if already initialized
        if not self._index_init:
            print(self._RED + 'Index not initialized/loaded yet!' + self._RST)
            return False

        # Load CSV
        print(f"Loading CSV file '{csv_file_path}'...")

        try:
            csv_data = pandas.read_csv(csv_file_path, sep=sep)
        except Exception as e:
            return self._handle_read_index_error(e, csv_file_path)

        # Get values of provided field
        try:
            texts = csv_data[csv_field].tolist()
        except KeyError:
            print(self._RED +
                  f"Column '{csv_field}' not found in CSV"+self._RST)
            return False

        # If chunking, manage texts in list
        if chunking:
            # Chunk
            texts = self._chunk_texts(texts, chunk_size, chunk_overlap)

            print(
                f"Original texts divided into {len(texts)} chunks")

        # Texts into embedding
        embeddings = self._model.encode(texts, convert_to_numpy=True)
        # Add to index
        self._index.add(embeddings)

        # Store original texts
        self._texts.extend(texts)

        # Output
        print(
            f'Added {len(texts)} embeddings - '
            f'A total of {self._index.ntotal} embeddings on index'
        )

        return True

    def _get_files_by_extension(self, folder_path: str,
                                extension: str) -> bool:
        return [
            f for f in os.listdir(folder_path)
            if f.endswith(extension) and os.path.isfile(
                os.path.join(folder_path, f))
        ]

    def _load_pdf_as_string(self, file_path: str) -> str:
        complete_string = ""
        with fitz.open(file_path) as doc:  # automatically closes
            for page in doc:
                complete_string += page.get_text()

        return complete_string

    def add_pdfs_from_folder(self, folder_path: str, chunk_size: int = 512,
                             chunk_overlap: int = 50) -> bool:
        # Check if already initialized
        if not self._index_init:
            print(self._RED + 'Index not initialized/loaded yet!' + self._RST)
            return False

        # Manage path
        if not folder_path.endswith(os.sep):
            folder_path = folder_path + os.sep

        # Get folder files with PDF extension
        try:
            file_list = self._get_files_by_extension(folder_path, 'pdf')
        except Exception as e:
            print(self._RED + f'An unexpected error occurred: {e}' + self._RST)
            return False

        # Get file content iterating on all PDFs
        file_content = []

        for file in tqdm(file_list, desc="Document loading"):
            # Output
            print(f"Loading data from {file}...")
            # Get file content as string
            file_content.append(self._load_pdf_as_string(folder_path + file))

        # Chunk file content
        chunks = self._chunk_texts(file_content, chunk_size,
                                   chunk_overlap)

        print(f"PDFs divided into {len(chunks)} chunks")

        # Texts into embedding
        print('Embedding chunks...')
        embeddings = self._model.encode(chunks, convert_to_numpy=True)
        # Add to index
        print('Adding chunks to index...')
        self._index.add(embeddings)

        # Store original texts
        self._texts.extend(chunks)

        # Output
        print(
            f'Added {len(chunks)} embeddings - '
            f'A total of {self._index.ntotal} embeddings on index'
        )

        return True

    def get_index_size(self) -> tuple[bool, int]:
        # Check if already initialized
        if not self._index_init:
            print(self._RED + 'Index not initialized/loaded yet!' + self._RST)
            return False, -1

        # Output
        return True, self._index.ntotal

    def search(
            self, texts: list[str],
            top_k: int = 5
    ) -> tuple[bool, list[list[str]], list[list[float]]]:
        # Check if already initialized
        if not self._index_init:
            print(self._RED + 'Index not initialized/loaded yet!' + self._RST)
            return False, [], []

        # Texts into embedding
        embeddings = self._model.encode(texts, convert_to_numpy=True)
        # Search
        distances, indices = self._index.search(embeddings, top_k)
        # Reconstruct
        output_texts = []
        for row in indices:
            row_texts = [self._texts[i] for i in row]
            output_texts.append(row_texts)

        return True, output_texts, distances

    def save_index(self, path: str, index_name: str) -> bool:
        # Check if already initialized
        if not self._index_init:
            print(self._RED + 'Index not initialized/loaded yet!' + self._RST)
            return False

        # Ensure the path ends with a '/'
        if not path.endswith('/'):
            path += '/'

        # Manage paths and file names
        complete_index_path = path + index_name + '.faiss'
        complete_raw_text_path = path + index_name + '.csv'

        # Save/write index
        try:
            faiss.write_index(self._index, complete_index_path)
        except PermissionError:
            print(self._RED + 'Permission denied: unable to write the index' +
                  self._RST)
            return False
        except IOError as e:
            print(self._RED + f'I/O error occurred while saving the index: {e}'
                  + self._RST)
            return False
        except Exception as e:
            print(self._RED + f'An unexpected error occurred: {e}' + self._RST)
            return False

        # Save raw text data
        df = pandas.DataFrame(self._texts, columns=['text'])
        df.to_csv(complete_raw_text_path, sep=';',
                  index=True, index_label='ID')

        # Output
        print('Index and raw text files successfully created!')

        return True
