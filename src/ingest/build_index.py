
"""
This build index file is building the index and persisting
the metadata from the VectorStoreIndex. 
- We first load documents using pymupdf
- Build the index from the documents
- then persist the index
"""
from src.config import ABSOLUTE_DATA_DIR_PATH, configure_settings
import json
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
import pymupdf
from pathlib import Path
import re

class IndexBuilder:

    def __init__(self) -> None:
        self.ABSOLUTE_DATA_DIR = ABSOLUTE_DATA_DIR_PATH
        self.manifest_file_path = str(self.ABSOLUTE_DATA_DIR / "manifest.json")
    
    # getting file paths from manifest.json using Pathlib magic
    def _get_pdf_file_paths(self) -> list[str]:
        manifest_json = json.loads(Path(self.manifest_file_path).read_text())
        pdf_files = manifest_json['pdfs']
        my_list = []
        for pdf_path in pdf_files:
            my_pdf_path = str(self.ABSOLUTE_DATA_DIR / "pdfs" / Path(pdf_path).name)
            my_list.append(my_pdf_path)
        return my_list
    
    # getting metadata from regex in a simpler way
    def _parse_metadata(self, pdf_path: str) -> dict[str, str | int]:
        name = Path(pdf_path).name
        m = re.match(r"(?P<ticker>[A-Z]+)_FY(?P<year>\d{4})_", name)
        if not m:
            raise ValueError(f"unexpected pdf filename: {name}")
        return {
            "company_ticker": m.group("ticker"),
            "fiscal_year": int(m.group("year")),
        }
    
    # We are loading the documents here
    # using pymupdf to process each one individually
    # taking parse_metadata data here with text
    def load_documents(self) -> list[Document]:
        pdf_document_filepaths = self._get_pdf_file_paths()
        documents: list[Document] = []
        for pdf_path in pdf_document_filepaths:
            base_meta = self._parse_metadata(pdf_path)
            doc: pymupdf.Document = pymupdf.open(pdf_path)
            for page in doc.pages():
                text = page.get_text().strip()
                # skip if no text found
                if not text:
                    continue
                documents.append(
                    Document(
                        text=text,
                        metadata={
                            **base_meta,
                            "page_number": page.number + 1  #must start at 1
                        }
                    )
                )
            doc.close()
        return documents

    # we want to use sentence splitter because it splits on sentences. Pretty natural
    # need to test different chunk sizing and overlapping in eval section
    def build_index(self, documents: list[Document]) -> VectorStoreIndex:
        splitter = SentenceSplitter(chunk_size=512, chunk_overlap=64)
        return VectorStoreIndex.from_documents(
            documents,
            transformations=[splitter]
        )
    
    def run(self) -> VectorStoreIndex:
        configure_settings()
        documents = self.load_documents()
        return self.build_index(documents)

if __name__ == "__main__":
    IndexBuilder().run()

