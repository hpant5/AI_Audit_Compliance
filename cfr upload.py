import os
import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path
import xml.etree.ElementTree as ET
import hashlib
from tqdm import tqdm
import re

class CFRIngestion:
    def __init__(self, chroma_db_path="./chroma_legal_db"):
        """Initialize ChromaDB for CFR (Code of Federal Regulations)"""
        print("Initializing ChromaDB for CFR...")
        
        self.client = chromadb.PersistentClient(path=chroma_db_path)
        
        # Use Legal-BERT
        legal_bert_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="nlpaueb/legal-bert-base-uncased"
        )
        
        # Get or create federal_laws_bert collection
        self.collection = self.client.get_or_create_collection(
            name="federal_laws_bert",
            embedding_function=legal_bert_ef,
            metadata={"description": "Federal laws and regulations with Legal-BERT"}
        )
        
        print(f"✓ Federal laws collection initialized")
        print(f"✓ Current documents: {self.collection.count()}\n")
    
    def extract_text_from_xml(self, xml_path: Path) -> dict:
        """Extract text content from CFR XML file"""
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # Extract title and subject
            title_num = root.find('.//TITLENUM')
            title = title_num.text if title_num is not None else "Unknown Title"
            
            subject = root.find('.//SUBJECT')
            subject_text = subject.text if subject is not None else ""
            
            # Extract revision date
            revised = root.find('.//REVISED')
            revision_date = revised.text if revised is not None else "Unknown"
            
            # Extract all text content from the document
            all_text = []
            
            # Get all text elements (paragraphs, sections, etc.)
            for elem in root.iter():
                if elem.text and elem.text.strip():
                    text = elem.text.strip()
                    # Skip very short text and metadata
                    if len(text) > 20 and elem.tag not in ['CODE', 'PRTPAGE', 'GID']:
                        all_text.append(text)
            
            # Combine all text
            full_text = ' '.join(all_text)
            
            # Clean text
            full_text = re.sub(r'\s+', ' ', full_text)
            full_text = full_text.strip()
            
            if len(full_text) < 100:
                return None
            
            # Limit to 4000 words for BERT
            words = full_text.split()
            if len(words) > 4000:
                full_text = ' '.join(words[:4000]) + '...'
            
            return {
                'title': title,
                'subject': subject_text,
                'revision_date': revision_date,
                'content': full_text,
                'file_path': str(xml_path),
                'file_name': xml_path.name
            }
            
        except Exception as e:
            print(f"Error parsing {xml_path.name}: {e}")
            return None
    
    def process_cfr_folder(self, cfr_base_path: Path, batch_size=50):
        """
        Process all CFR XML files
        
        Structure: CFR-2025/title-1/file.xml, title-2/file.xml, ...
        """
        if not cfr_base_path.exists():
            print(f"Error: Folder {cfr_base_path} not found!")
            return
        
        print(f"Processing CFR folder: {cfr_base_path.name}\n")
        
        # Find all title folders
        title_folders = [f for f in cfr_base_path.iterdir() if f.is_dir() and f.name.startswith('title-')]
        
        if not title_folders:
            print("No title folders found!")
            return
        
        print(f"Found {len(title_folders)} title folders\n")
        
        total_processed = 0
        total_skipped = 0
        
        documents = []
        metadatas = []
        ids = []
        
        # Process each title folder
        for title_folder in sorted(title_folders):
            title_num = title_folder.name.replace('title-', '')
            print(f"\nProcessing Title {title_num}...")
            
            # Find all XML files in this title
            xml_files = list(title_folder.glob('*.xml'))
            
            if not xml_files:
                print(f"  No XML files in {title_folder.name}")
                continue
            
            print(f"  Found {len(xml_files)} XML files")
            
            # Process each XML file
            for xml_file in tqdm(xml_files, desc=f"  Title {title_num}"):
                cfr_data = self.extract_text_from_xml(xml_file)
                
                if not cfr_data:
                    total_skipped += 1
                    continue
                
                total_processed += 1
                
                # Generate unique ID
                doc_id = hashlib.md5(str(xml_file).encode()).hexdigest()
                
                # Add to batch
                documents.append(cfr_data['content'])
                metadatas.append({
                    'source': 'CFR',
                    'title': cfr_data['title'],
                    'subject': cfr_data['subject'],
                    'revision_date': cfr_data['revision_date'],
                    'file_name': cfr_data['file_name'],
                    'file_path': cfr_data['file_path'],
                    'cfr_title_number': title_num
                })
                ids.append(doc_id)
                
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
                        print(f"\n  Error inserting batch: {e}")
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
        print(f"CFR INGESTION COMPLETE")
        print(f"{'='*70}")
        print(f"XML files processed:  {total_processed}")
        print(f"XML files skipped:    {total_skipped}")
        print(f"Total in database:    {self.collection.count()}")
        print(f"{'='*70}\n")
        
        return {
            'files_processed': total_processed
        }


def main():
    print("\n" + "="*70)
    print("CFR (CODE OF FEDERAL REGULATIONS) INGESTION")
    print("="*70 + "\n")
    
    ingestion = CFRIngestion(chroma_db_path="./chroma_legal_db")
    
    # Path to your CFR folder
    cfr_folder = Path("C:/Users/hardi/Downloads/CFR-2025")
    
    if not cfr_folder.exists():
        print(f"Error: CFR folder not found at {cfr_folder}")
        print("Please update the path in the script")
        return
    
    # Process all CFR XML files
    ingestion.process_cfr_folder(cfr_folder, batch_size=50)


if __name__ == "__main__":
    main()