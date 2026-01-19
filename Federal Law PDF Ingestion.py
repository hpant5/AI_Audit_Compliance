import os
import chromadb
from chromadb.utils import embedding_functions
from bs4 import BeautifulSoup
from pathlib import Path
from typing import Dict
import hashlib
from tqdm import tqdm
import time
import re

class LegalCaseVectorDB:
    def __init__(self, chroma_db_path="./chroma_legal_db", use_azure_for_classification=True):
        """
        Initialize Chroma database with Legal-BERT embeddings
        
        Args:
            chroma_db_path: Path to ChromaDB storage
            use_azure_for_classification: Use Azure OpenAI for case type classification
        """
        print("Initializing Chroma database with Legal-BERT...")
        
        # Create persistent Chroma client
        self.client = chromadb.PersistentClient(path=chroma_db_path)
        
        # Use Legal-BERT for embeddings
        legal_bert_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="nlpaueb/legal-bert-base-uncased"
        )
        
        # Create or get collection for legal cases
        self.collection = self.client.get_or_create_collection(
            name="legal_cases_bert",
            embedding_function=legal_bert_ef,
            metadata={"description": "Legal cases with Legal-BERT embeddings"}
        )
        
        # Initialize Azure OpenAI for case classification if enabled
        self.use_llm_classification = use_azure_for_classification
        if use_azure_for_classification:
            try:
                from langchain_openai import AzureChatOpenAI
                AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
                AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
                AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
                AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
                
                self.llm = AzureChatOpenAI(
                    azure_endpoint=AZURE_OPENAI_ENDPOINT,
                    api_key=AZURE_OPENAI_API_KEY,
                    azure_deployment=AZURE_OPENAI_DEPLOYMENT,
                    api_version=AZURE_OPENAI_API_VERSION,
                    temperature=0.1,
                )
                print("✓ Azure OpenAI initialized for case type classification")
            except Exception as e:
                print(f"⚠ Could not initialize Azure OpenAI: {e}")
                print("  Will use keyword-based classification as fallback")
                self.llm = None
                self.use_llm_classification = False
        
        print(f"✓ Chroma database initialized at: {chroma_db_path}")
        print(f"✓ Using embedding model: nlpaueb/legal-bert-base-uncased")
        print(f"✓ Current documents in collection: {self.collection.count()}")
    
    def detect_case_type(self, text: str, use_llm: bool = True) -> str:
        """
        Automatically detect case type using Azure OpenAI or keyword fallback
        """
        if use_llm and hasattr(self, 'llm'):
            try:
                # Use LLM for classification
                prompt = f"""Classify this legal case into ONE of these categories:
- Criminal - DUI
- Criminal
- Family Law
- Employment
- Property/Real Estate
- Corporate/Business
- Tort/Personal Injury
- Contract
- Administrative
- Tax
- Bankruptcy
- Civil
- General

Case text (first 1000 chars):
{text[:1000]}

Respond with ONLY the category name, nothing else."""
                
                from langchain_core.messages import HumanMessage
                response = self.llm.invoke([HumanMessage(content=prompt)])
                case_type = response.content.strip()
                
                # Validate response
                valid_types = [
                    "Criminal - DUI", "Criminal", "Family Law", "Employment",
                    "Property/Real Estate", "Corporate/Business", "Tort/Personal Injury",
                    "Contract", "Administrative", "Tax", "Bankruptcy", "Civil", "General"
                ]
                
                if case_type in valid_types:
                    return case_type
                    
            except Exception as e:
                print(f"LLM classification failed, using keyword fallback: {e}")
        
        # Keyword fallback
        text_lower = text.lower()
        
        if any(k in text_lower for k in ["dui", "d.u.i", "driving under the influence"]):
            return "Criminal - DUI"
        if any(k in text_lower for k in ["criminal", "defendant", "prosecution"]):
            return "Criminal"
        if any(k in text_lower for k in ["divorce", "custody", "child support"]):
            return "Family Law"
        if any(k in text_lower for k in ["employment", "wrongful termination"]):
            return "Employment"
        if any(k in text_lower for k in ["real property", "easement", "landlord"]):
            return "Property/Real Estate"
        if any(k in text_lower for k in ["shareholder", "corporate", "fiduciary duty"]):
            return "Corporate/Business"
        if any(k in text_lower for k in ["negligence", "personal injury", "malpractice"]):
            return "Tort/Personal Injury"
        if any(k in text_lower for k in ["breach of contract", "contractual"]):
            return "Contract"
        if any(k in text_lower for k in ["plaintiff", "damages"]):
            return "Civil"
        
        return "General"
    
    def parse_html_case(self, html_path: Path, state: str = "Unknown") -> Dict:
        """
        Extract full case content and metadata - ONE chunk per case
        
        Args:
            html_path: Path to HTML file
            state: State name (for court inference)
        """
        try:
            with open(html_path, 'r', encoding='utf-8', errors='ignore') as f:
                html_content = f.read()
            
            soup = BeautifulSoup(html_content, 'lxml')
            
            # Extract metadata
            metadata = {}
            
            # Case name (parties)
            parties = soup.find('h4', class_='parties')
            if parties:
                case_name = parties.get_text(strip=True)
                case_name = re.sub(r'[â€ ]', '', case_name)
                case_name = case_name.split('\n')[0].split(',')[0:2]
                metadata['case_name'] = ', '.join(case_name) if len(case_name) > 1 else case_name[0]
            else:
                title = soup.find('title')
                metadata['case_name'] = title.get_text(strip=True) if title else html_path.stem
            
            # Decision date - use regex to extract date from text
            decision_date_elem = soup.find('p', class_='decisiondate')
            if decision_date_elem:
                date_text = decision_date_elem.get_text(strip=True)
                # Extract date using regex pattern
                date_pattern = r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\b'
                dates = re.findall(date_pattern, date_text)
                metadata['decision_date'] = dates[0] if dates else date_text
            else:
                # Fallback: search in full text
                date_pattern = r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\b'
                full_text_preview = soup.get_text()[:1000]
                dates = re.findall(date_pattern, full_text_preview)
                metadata['decision_date'] = dates[0] if dates else 'Unknown'
            
            # Court - For Arizona, extract from docket number area or infer from state
            court = soup.find('p', class_='court')
            if court:
                metadata['court'] = court.get_text(strip=True)
            else:
                # Arizona doesn't have court element, infer from docket or use default
                docket = soup.find('p', class_='docketnumber')
                if docket:
                    docket_text = docket.get_text(strip=True).lower()
                    if 'civil' in docket_text:
                        metadata['court'] = f"{state} Superior Court"
                    elif 'criminal' in docket_text:
                        metadata['court'] = f"{state} Superior Court - Criminal"
                    else:
                        metadata['court'] = f"{state} Court"
                else:
                    # Last resort: check body text for court mentions
                    body_preview = soup.get_text()[:500].lower()
                    if 'supreme court' in body_preview:
                        metadata['court'] = f"{state} Supreme Court"
                    elif 'court of appeals' in body_preview or 'appellate' in body_preview:
                        metadata['court'] = f"{state} Court of Appeals"
                    else:
                        metadata['court'] = f"{state} Court"
            
            # Citation
            citation = soup.find('p', class_='citation')
            metadata['citation'] = citation.get_text(strip=True) if citation else 'Unknown'
            
            # Extract ALL text content from the case
            # Remove script, style, and navigation elements
            for element in soup(['script', 'style', 'nav', 'header', 'footer']):
                element.decompose()
            
            # Get all text from the case body
            case_section = soup.find('section', class_='casebody')
            if case_section:
                full_text = case_section.get_text(separator=' ', strip=True)
            else:
                # Fallback to entire body
                full_text = soup.get_text(separator=' ', strip=True)
            
            # Clean the text
            full_text = re.sub(r'\s+', ' ', full_text)
            full_text = re.sub(r'[â€ ]+', '', full_text)
            full_text = re.sub(r'\u200b', '', full_text)  # Zero-width space
            full_text = re.sub(r'\xa0', ' ', full_text)   # Non-breaking space
            full_text = full_text.strip()
            
            # Limit text length (ChromaDB has limits, typically 8191 tokens for BERT)
            # ~4000 words = ~5000 tokens (safe limit)
            words = full_text.split()
            if len(words) > 4000:
                full_text = ' '.join(words[:4000]) + '...'
            
            if len(full_text) < 100:
                return None
            
            # Detect case type from content (using Azure OpenAI if available)
            case_type = self.detect_case_type(full_text, use_llm=self.use_llm_classification)
            
            return {
                'case_name': metadata['case_name'],
                'decision_date': metadata['decision_date'],
                'court': metadata['court'],
                'citation': metadata['citation'],
                'case_type': case_type,  # Auto-detected
                'content': full_text,
                'file_path': str(html_path),
                'file_name': html_path.name
            }
            
        except Exception as e:
            print(f"Error parsing {html_path}: {e}")
            return None
    
    def generate_case_id(self, file_path: str) -> str:
        """Generate unique ID for each case"""
        return hashlib.md5(file_path.encode()).hexdigest()
    
    def process_single_folder(self, folder_path: Path, batch_size=50):
        """
        Process a single state/case type folder
        ONE CHUNK PER CASE
        
        Args:
            folder_path: Path to specific folder (e.g., Laws/samoa)
            batch_size: Number of cases to batch before inserting
        """
        if not folder_path.exists():
            print(f"ERROR: Folder '{folder_path}' not found!")
            return None
        
        print(f"\n{'='*70}")
        print(f"PROCESSING: {folder_path.name}")
        print(f"{'='*70}\n")
        
        # Find all HTML files in this specific folder
        html_files = list(folder_path.glob('**/*.html')) + list(folder_path.glob('**/*.htm'))
        
        if not html_files:
            print(f"No HTML files found in {folder_path}")
            return None
        
        print(f"Found {len(html_files):,} HTML files")
        
        # Extract state from path - FIXED: use -1 instead of -2
        path_parts = folder_path.parts
        state = path_parts[-1] if len(path_parts) >= 1 else 'Unknown'
        
        print(f"State: {state}\n")
        print("Note: Case types will be auto-detected from document content\n")
        
        # Start timing
        start_time = time.time()
        
        # Prepare batches
        documents = []
        metadatas = []
        ids = []
        
        cases_processed = 0
        cases_skipped = 0
        
        print("Starting processing...\n")
        
        # Process each HTML file - ONE CHUNK PER CASE
        for html_file in tqdm(html_files, desc="Processing cases"):
            # Parse entire case (pass state for court inference)
            case_data = self.parse_html_case(html_file, state=state)
            
            if not case_data:
                cases_skipped += 1
                continue
            
            cases_processed += 1
            
            # Generate unique ID for this case
            case_id = self.generate_case_id(str(html_file))
            
            # Add case to batch
            documents.append(case_data['content'])
            metadatas.append({
                'state': state,
                'case_name': case_data['case_name'],
                'case_type': case_data['case_type'],  # Now auto-detected
                'decision_date': case_data['decision_date'],
                'court': case_data['court'],  # Added court field
                'citation': case_data['citation'],  # Added citation field
                'file_name': case_data['file_name'],
                'file_path': case_data['file_path']
            })
            ids.append(case_id)
            
            # Insert batch when it reaches batch_size
            if len(documents) >= batch_size:
                try:
                    print(f"  Inserting batch of {len(documents)} cases...")
                    self.collection.add(
                        documents=documents,
                        metadatas=metadatas,
                        ids=ids
                    )
                    documents, metadatas, ids = [], [], []
                except Exception as e:
                    print(f"\nError inserting batch: {e}")
                    documents, metadatas, ids = [], [], []
        
        # Insert remaining documents
        if documents:
            try:
                print(f"  Inserting final batch of {len(documents)} cases...")
                self.collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
            except Exception as e:
                print(f"\nError inserting final batch: {e}")
        
        # End timing
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Print results
        print(f"\n{'='*70}")
        print(f"PROCESSING COMPLETE")
        print(f"{'='*70}\n")
        
        print(f"STATISTICS:")
        print(f"  Cases processed:        {cases_processed:,}")
        print(f"  Cases skipped:          {cases_skipped:,}")
        print(f"  Total in DB:            {self.collection.count():,}")
        print(f"  Chunks per case:        1 (full case per chunk)\n")
        
        print(f"TIMING:")
        print(f"  Total time:             {processing_time:.2f} seconds ({processing_time/60:.2f} minutes)")
        print(f"  Time per case:          {processing_time/cases_processed if cases_processed > 0 else 0:.3f} seconds")
        print(f"  Cases per minute:       {cases_processed/(processing_time/60) if processing_time > 0 else 0:.1f}\n")
        
        return {
            'cases_processed': cases_processed,
            'processing_time': processing_time
        }
    
    def search(self, query: str, n_results: int = 5, state_filter: str = None):
        """
        Search for relevant legal cases
        
        Args:
            query: Search query
            n_results: Number of results to return
            state_filter: Optional state filter
        """
        where_filter = {"state": state_filter} if state_filter else None
        
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_filter
        )
        
        return results
    
    def test_search(self):
        """
        Test search functionality with sample queries
        """
        print(f"\n{'='*70}")
        print(f"TESTING SEARCH FUNCTIONALITY")
        print(f"{'='*70}\n")
        
        test_queries = [
            "breach of fiduciary duty",
            "contract dispute",
            "employment termination"
        ]
        
        for query in test_queries:
            print(f"Query: '{query}'")
            print(f"  {'-'*66}")
            
            results = self.search(query, n_results=3)
            
            if results['documents'][0]:
                for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
                    print(f"\n  Result #{i+1}:")
                    print(f"    Case: {metadata['case_name']}")
                    print(f"    State: {metadata['state']}")
                    print(f"    Case Type: {metadata['case_type']}")
                    print(f"    Date: {metadata['decision_date']}")
                    print(f"    Preview: {doc[:300]}...")
            else:
                print("  No results found.")
            
            print("\n")


def main():
    """
    Main execution - Process single folder only
    """
    print("\n" + "="*70)
    print("LEGAL CASE VECTORIZATION - Legal-BERT")
    print("One Chunk Per Case with Auto Case Type Detection")
    print("="*70 + "\n")
    
    # Initialize vector database with Legal-BERT
    vector_db = LegalCaseVectorDB(chroma_db_path="./chroma_legal_db")
    
    # SPECIFY YOUR FOLDER HERE
    target_folder = Path("C:/Users/hardi/Downloads/testing/New Mexico")
    
    # Process the single folder
    stats = vector_db.process_single_folder(target_folder, batch_size=50)
    
    # Test search if data was processed
    if stats and stats['cases_processed'] > 0:
        vector_db.test_search()
    
    print("\n" + "="*70)
    print("COMPLETE!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
