import os
import chromadb
from chromadb.config import Settings
from bs4 import BeautifulSoup
from pathlib import Path
from typing import List, Dict
import hashlib
from tqdm import tqdm

class LegalCaseVectorDB:
    def __init__(self, chroma_db_path="./legal_cases_chroma_db"):
        """
        Initialize Chroma database for legal cases
        """
        print("Initializing Chroma database...")
        
        # Create persistent Chroma client
        self.client = chromadb.PersistentClient(path=chroma_db_path)
        
        # Create or get collection for legal cases
        self.collection = self.client.get_or_create_collection(
            name="legal_cases",
            metadata={"description": "Legal cases from all US states"}
        )
        
        print(f"✓ Chroma database initialized at: {chroma_db_path}")
        print(f"✓ Current documents in collection: {self.collection.count()}")
    
    def extract_text_from_html(self, html_path: Path) -> Dict:
        """
        Extract text content from HTML file
        """
        try:
            with open(html_path, 'r', encoding='utf-8', errors='ignore') as f:
                html_content = f.read()
            
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style elements
            for element in soup(["script", "style", "meta", "link"]):
                element.decompose()
            
            # Get text content
            text = soup.get_text(separator='\n', strip=True)
            
            # Clean up extra whitespace
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            text = '\n'.join(lines)
            
            # Try to extract case title
            title = soup.find('title')
            if title:
                case_title = title.get_text(strip=True)
            else:
                # Use filename as fallback
                case_title = html_path.stem
            
            return {
                'text': text,
                'title': case_title,
                'file_path': str(html_path),
                'file_name': html_path.name
            }
            
        except Exception as e:
            print(f"    ✗ Error reading {html_path.name}: {e}")
            return None
    
    def chunk_text(self, text: str, chunk_size=1000, overlap=200) -> List[str]:
        """
        Split text into overlapping chunks
        
        Args:
            text: The text to chunk
            chunk_size: Target size of each chunk in characters
            overlap: Number of characters to overlap between chunks
        """
        if not text or len(text) == 0:
            return []
        
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = start + chunk_size
            
            # If we're not at the end, try to break at a sentence
            if end < text_length:
                # Look for sentence endings (. ! ?) in the next 150 characters
                for punct in ['. ', '! ', '? ', '\n\n']:
                    punct_pos = text.rfind(punct, end, min(end + 150, text_length))
                    if punct_pos != -1:
                        end = punct_pos + len(punct)
                        break
            
            chunk = text[start:end].strip()
            
            # Only add non-empty chunks with substantial content
            if chunk and len(chunk) > 50:
                chunks.append(chunk)
            
            # Move start position with overlap
            start = end - overlap if end < text_length else text_length
        
        return chunks
    
    def generate_chunk_id(self, state: str, file_name: str, chunk_index: int) -> str:
        """
        Generate unique ID for each chunk
        """
        unique_string = f"{state}_{file_name}_{chunk_index}"
        return hashlib.md5(unique_string.encode()).hexdigest()
    
    def process_state_folder(self, state_path: Path, batch_size=50):
        """
        Process all HTML files in a state folder and add to Chroma
        
        Args:
            state_path: Path to the state folder
            batch_size: Number of chunks to batch before inserting
        """
        state_name = state_path.name.lower()
        print(f"\n{'='*70}")
        print(f"Processing State: {state_name.upper()}")
        print(f"{'='*70}")
        
        # Find all HTML files in the state folder
        html_files = list(state_path.glob('*.html')) + list(state_path.glob('*.htm'))
        
        if not html_files:
            print(f"  ⚠ No HTML files found in {state_name}")
            return
        
        print(f"  Found {len(html_files)} HTML case files")
        
        # Prepare batches
        documents = []
        metadatas = []
        ids = []
        
        total_chunks = 0
        files_processed = 0
        files_skipped = 0
        
        # Process each HTML file
        for html_file in tqdm(html_files, desc=f"  Processing {state_name}"):
            # Extract text from HTML
            extracted = self.extract_text_from_html(html_file)
            
            if not extracted or not extracted['text']:
                files_skipped += 1
                continue
            
            # Chunk the text
            chunks = self.chunk_text(extracted['text'], chunk_size=1000, overlap=200)
            
            if not chunks:
                files_skipped += 1
                continue
            
            files_processed += 1
            
            # Add each chunk to batch
            for chunk_idx, chunk in enumerate(chunks):
                chunk_id = self.generate_chunk_id(state_name, html_file.name, chunk_idx)
                
                documents.append(chunk)
                metadatas.append({
                    'state': state_name,
                    'case_title': extracted['title'],
                    'file_name': extracted['file_name'],
                    'file_path': extracted['file_path'],
                    'chunk_index': chunk_idx,
                    'total_chunks': len(chunks)
                })
                ids.append(chunk_id)
                total_chunks += 1
                
                # Insert batch when it reaches batch_size
                if len(documents) >= batch_size:
                    try:
                        self.collection.add(
                            documents=documents,
                            metadatas=metadatas,
                            ids=ids
                        )
                        documents, metadatas, ids = [], [], []
                    except Exception as e:
                        print(f"\n    ✗ Error inserting batch: {e}")
                        documents, metadatas, ids = [], [], []
        
        # Insert remaining documents
        if documents:
            try:
                self.collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
            except Exception as e:
                print(f"\n    ✗ Error inserting final batch: {e}")
        
        print(f"\n  ✓ State Complete:")
        print(f"    - Files processed: {files_processed}")
        print(f"    - Files skipped: {files_skipped}")
        print(f"    - Chunks created: {total_chunks}")
        print(f"    - Total in DB: {self.collection.count()}")
    
    def process_all_states(self, root_dir='.'):
        """
        Process all state folders in the root directory
        """
        root_path = Path(root_dir)
        
        # Get all subdirectories (state folders)
        state_folders = [f for f in root_path.iterdir() 
                        if f.is_dir() 
                        and not f.name.startswith('.') 
                        and not f.name.startswith('_')
                        and f.name != 'legal_cases_chroma_db']
        
        if not state_folders:
            print("❌ No state folders found!")
            return
        
        print(f"\n{'#'*70}")
        print(f"# Found {len(state_folders)} state folders to process")
        print(f"# Starting vectorization process...")
        print(f"{'#'*70}")
        
        start_count = self.collection.count()
        
        # Process each state folder
        for state_folder in sorted(state_folders):
            try:
                self.process_state_folder(state_folder)
            except Exception as e:
                print(f"\n❌ Error processing {state_folder.name}: {e}")
                continue
        
        end_count = self.collection.count()
        
        print(f"\n{'#'*70}")
        print(f"# VECTORIZATION COMPLETE!")
        print(f"{'#'*70}")
        print(f"  Total documents in database: {end_count}")
        print(f"  New documents added: {end_count - start_count}")
        print(f"  Database location: ./legal_cases_chroma_db")
        print(f"{'#'*70}\n")
    
    def search_cases(self, query: str, n_results=5, state_filter=None):
        """
        Search for relevant legal cases
        
        Args:
            query: Search query
            n_results: Number of results to return
            state_filter: Optional state name to filter results
        """
        # Build filter if state specified
        where_filter = None
        if state_filter:
            where_filter = {"state": state_filter.lower()}
        
        # Perform search
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_filter
        )
        
        return results
    
    def print_search_results(self, query: str, n_results=5, state_filter=None):
        """
        Search and pretty print results
        """
        print(f"\n{'='*70}")
        print(f"Search Query: '{query}'")
        if state_filter:
            print(f"Filtered by state: {state_filter}")
        print(f"{'='*70}\n")
        
        results = self.search_cases(query, n_results, state_filter)
        
        if not results['documents'][0]:
            print("No results found.")
            return
        
        for i, (doc, metadata, distance) in enumerate(zip(
            results['documents'][0], 
            results['metadatas'][0],
            results['distances'][0]
        )):
            print(f"Result #{i+1} (Similarity: {1-distance:.3f})")
            print(f"  State: {metadata['state'].upper()}")
            print(f"  Case: {metadata['case_title']}")
            print(f"  File: {metadata['file_name']}")
            print(f"  Chunk: {metadata['chunk_index']+1}/{metadata['total_chunks']}")
            print(f"  Preview: {doc[:250]}...")
            print()


def main():
    """
    Main execution function
    """
    print("\n" + "="*70)
    print("LEGAL CASE VECTORIZATION SYSTEM")
    print("="*70 + "\n")
    
    # Initialize the vector database
    vector_db = LegalCaseVectorDB(chroma_db_path="./legal_cases_chroma_db")
    
    # Process all state folders
    vector_db.process_all_states(root_dir='.')
    
    # Test search functionality
    print("\n" + "="*70)
    print("TESTING SEARCH FUNCTIONALITY")
    print("="*70)
    
    # Example searches
    test_queries = [
        "employment wrongful termination",
        "tenant eviction rights",
        "contract breach damages"
    ]
    
    for query in test_queries:
        vector_db.print_search_results(query, n_results=3)
        print("\n" + "-"*70 + "\n")


if __name__ == "__main__":
    main()