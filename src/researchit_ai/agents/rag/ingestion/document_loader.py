from langchain_core.document_loaders import BaseLoader
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader, Docx2txtLoader, MergedDataLoader, DirectoryLoader
from langchain_core.documents import Document
from pathlib import Path

from src.researchit_ai.utils.config_loader import get_config

class DocumentLoader:
    def __init__(self, dir_path: str = None):
        self.dir_path = Path(dir_path)
        self.document_loader = self._build_loader()

    def _build_loader(self):

        if not self.dir_path:
            raise ValueError("No directory path entered")

        if not self.dir_path.is_dir():
            raise ValueError(f"Not a valid directory: {self.dir_path}")

        pdf_loader = DirectoryLoader(
            path=self.dir_path,
            glob="**/*.pdf",
            loader_cls=PyMuPDFLoader,
            recursive=True,
            use_multithreading=True
        )    


