import os
import chromadb
from pathlib import Path
from typing import List, Dict
import hashlib
from tqdm import tqdm
import PyPDF2
import re

class FederalLawIngestion:
    def __init__(self, chroma_db_path="./chroma_legal_db"):
        """Initialize ChromaDB for federal laws"""
        print("Initializing ChromaDB for federal laws...")
        
        self.client = chromadb.PersistentClient(path=chroma_db_path)
        
        # Separate collection for federal laws
        self.collection = self.client.get_or_create_collection(
            name="federal_laws",
            metadata={"description": "Federal law documents"}
        )
        
        print(f"Federal laws collection initialized")
        print(f"Current documents: {self.collection.count()}\n")
    
    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """Extract text from PDF file"""
        try:
            text = ""
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text += f"\n[Page {page_num + 1}]\n{page_text}"
                    except Exception as e:
                        print(f"Error extracting page {page_num + 1}: {e}")
                        continue
            
            # Clean text
            text = re.sub(r'\s+', ' ', text)
            text = re.sub(r'\n+', '\n', text)
            
            return text.strip()
            
        except Exception as e:
            print(f"Error reading PDF {pdf_path}: {e}")
            return None
    
    def chunk_text(self, text: str, chunk_size: int = 1500, overlap: int = 300) -> List[str]:
        """Split text into overlapping chunks"""
        if not text or len(text) == 0:
            return []
        
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = start + chunk_size
            
            # Try to break at sentence boundary
            if end < text_length:
                for punct in ['. ', '.\n', '! ', '? ']:
                    punct_pos = text.rfind(punct, end - 200, end + 200)
                    if punct_pos != -1:
                        end = punct_pos + len(punct)
                        break
            
            chunk = text[start:end].strip()
            
            if chunk and len(chunk) > 100:
                chunks.append(chunk)
            
            start = end - overlap if end < text_length else text_length
        
        return chunks
    
    def process_pdf_folder(self, pdf_folder: Path, batch_size: int = 50):
        """Process all PDFs in a folder"""
        
        if not pdf_folder.exists():
            print(f"Error: Folder {pdf_folder} not found!")
            return
        
        # Find all PDFs
        pdf_files = list(pdf_folder.glob('**/*.pdf'))
        
        if not pdf_files:
            print(f"No PDF files found in {pdf_folder}")
            return
        
        print(f"\nFound {len(pdf_files)} PDF files")
        print(f"Processing...\n")
        
        documents = []
        metadatas = []
        ids = []
        
        total_chunks = 0
        files_processed = 0
        files_skipped = 0
        
        for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
            # Extract text
            text = self.extract_text_from_pdf(pdf_file)
            
            if not text or len(text) < 100:
                files_skipped += 1
                continue
            
            # Chunk text
            chunks = self.chunk_text(text, chunk_size=1500, overlap=300)
            
            if not chunks:
                files_skipped += 1
                continue
            
            files_processed += 1
            
            # Get law title from filename
            law_title = pdf_file.stem.replace('_', ' ').replace('-', ' ')
            
            # Add chunks to batch
            for chunk_idx, chunk in enumerate(chunks):
                chunk_id = hashlib.md5(
                    f"{pdf_file.name}_{chunk_idx}".encode()
                ).hexdigest()
                
                documents.append(chunk)
                metadatas.append({
                    'source': 'federal_law',
                    'law_title': law_title,
                    'file_name': pdf_file.name,
                    'file_path': str(pdf_file),
                    'chunk_index': chunk_idx,
                    'total_chunks': len(chunks)
                })
                ids.append(chunk_id)
                total_chunks += 1
                
                # Insert batch
                if len(documents) >= batch_size:
                    try:
                        self.collection.add(
                            documents=documents,
                            metadatas=metadatas,
                            ids=ids
                        )
                        documents, metadatas, ids = [], [], []
                    except Exception as e:
                        print(f"\nError inserting batch: {e}")
                        documents, metadatas, ids = [], [], []
        
        # Insert remaining
        if documents:
            try:
                self.collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
            except Exception as e:
                print(f"\nError inserting final batch: {e}")
        
        print(f"\n{'='*70}")
        print(f"FEDERAL LAW INGESTION COMPLETE")
        print(f"{'='*70}")
        print(f"Files processed:  {files_processed}")
        print(f"Files skipped:    {files_skipped}")
        print(f"Total chunks:     {total_chunks}")
        print(f"In database:      {self.collection.count()}")
        print(f"{'='*70}\n")
        
        return {
            'files_processed': files_processed,
            'total_chunks': total_chunks
        }


def main():
    ingestion = FederalLawIngestion(chroma_db_path="./chroma_legal_db")
    
    # Path to your federal law PDFs
    pdf_folder = Path("C:/Users/hardi/Downloads/Federal law")
    
    ingestion.process_pdf_folder(pdf_folder, batch_size=50)


if __name__ == "__main__":
    main()