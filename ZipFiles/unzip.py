import os
import zipfile
import shutil
from pathlib import Path

def unzip_and_extract_html(root_dir='.'):
    """
    Unzips all volumes in each state folder and extracts HTML files to the state's main directory.
    
    Args:
        root_dir: Root directory containing state folders (default is current directory)
    """
    root_path = Path(root_dir)
    
    # Get all subdirectories (state folders)
    state_folders = [f for f in root_path.iterdir() if f.is_dir()]
    
    print(f"Found {len(state_folders)} state folders")
    
    for state_folder in state_folders:
        state_name = state_folder.name
        print(f"\n{'='*60}")
        print(f"Processing: {state_name}")
        print(f"{'='*60}")
        
        # Find all zip files in the state folder
        zip_files = sorted(state_folder.glob('*.zip'), key=lambda x: int(x.stem) if x.stem.isdigit() else x.stem)
        
        if not zip_files:
            print(f"  No zip files found in {state_name}")
            continue
        
        print(f"  Found {len(zip_files)} zip volumes")
        
        # Process each zip file
        for zip_file in zip_files:
            volume_name = zip_file.stem
            print(f"\n  Processing volume: {volume_name}")
            
            try:
                # Create a temporary extraction directory
                temp_extract_dir = state_folder / f"_temp_extract_{volume_name}"
                temp_extract_dir.mkdir(exist_ok=True)
                
                # Unzip the file
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    zip_ref.extractall(temp_extract_dir)
                    print(f"    ✓ Unzipped {zip_file.name}")
                
                # Find the html folder in the extracted content
                html_folder = temp_extract_dir / 'html'
                
                if html_folder.exists() and html_folder.is_dir():
                    # Get all HTML files
                    html_files = list(html_folder.glob('**/*.html')) + list(html_folder.glob('**/*.htm'))
                    print(f"    Found {len(html_files)} HTML files")
                    
                    # Copy HTML files to the main state folder
                    copied_count = 0
                    for html_file in html_files:
                        # Preserve the relative path structure if there are subdirectories
                        relative_path = html_file.relative_to(html_folder)
                        destination = state_folder / relative_path
                        
                        # Create parent directories if needed
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        
                        # Copy the file
                        shutil.copy2(html_file, destination)
                        copied_count += 1
                    
                    print(f"    ✓ Copied {copied_count} HTML files to {state_name}/")
                else:
                    print(f"    ⚠ Warning: 'html' folder not found in {zip_file.name}")
                
                # Clean up temporary extraction directory
                shutil.rmtree(temp_extract_dir)
                print(f"    ✓ Cleaned up temporary files")
                
            except zipfile.BadZipFile:
                print(f"    ✗ Error: {zip_file.name} is not a valid zip file")
            except Exception as e:
                print(f"    ✗ Error processing {zip_file.name}: {str(e)}")
                # Try to clean up if temp directory exists
                if temp_extract_dir.exists():
                    shutil.rmtree(temp_extract_dir)
        
        print(f"\n✓ Completed processing {state_name}")
    
    print(f"\n{'='*60}")
    print("All states processed!")
    print(f"{'='*60}")

if __name__ == "__main__":
    # Run the extraction
    print("Starting extraction process...")
    unzip_and_extract_html()
    print("\nExtraction complete!")