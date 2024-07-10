import os
import shutil
from cltkreaders.lat import LatinLibraryCorpusReader

# Create an instance of the LatinLibraryCorpusReader
lib = LatinLibraryCorpusReader()

# Destination directory
dest_dir = os.path.expanduser("~/Rogostyle/texts/latinlib")

# Create the destination directory if it doesn't exist
os.makedirs(dest_dir, exist_ok=True)

# Recursively move all .txt files to the destination directory
def move_txt_files(directory):
    try:
        # Get the list of file IDs in the directory
        file_ids = lib.fileids(directory)
        
        for file_id in file_ids:
            file_path = f"{directory}/{file_id}"
            
            if file_id.endswith(".txt"):
                # Move the text file to the destination directory
                shutil.copy(lib.open(file_path).name, dest_dir)
                print(f"Moved {file_path} to {dest_dir}")
            else:
                # Recursively process subdirectories
                move_txt_files(file_path)
    except FileNotFoundError:
        print(f"Directory not found: {directory}")

# Start moving files from the root directory
move_txt_files("")