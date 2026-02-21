import os

def consolidate_repo(root_dir, output_file):
    # Extensions to include
    extensions = {'.cpp', '.h', '.hh', '.py', '.pxd', '.pyx', '.toml'}
    
    # Files to ignore (common venv or git directories)
    ignore_dirs = {'.git', '__pycache__', 'build', 'dist', 'deps', 'venv', 'env', '.idea', '.vscode'}

    with open(output_file, 'w', encoding='utf-8') as outfile:
        # Walk through the directory tree
        for dirpath, dirnames, filenames in os.walk(root_dir):
            # Modify dirnames in-place to skip ignored directories
            dirnames[:] = [d for d in dirnames if d not in ignore_dirs]
            
            for filename in filenames:
                if filename == "wrapper.cpp":
                    continue
                _, ext = os.path.splitext(filename)
                
                if ext in extensions:
                    file_path = os.path.join(dirpath, filename)
                    # Create a relative path for cleaner metadata
                    rel_path = os.path.relpath(file_path, root_dir)
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='replace') as infile:
                            content = infile.read()
                            
                            # Write metadata header
                            outfile.write(f"\n{'='*80}\n")
                            outfile.write(f"FILE: {rel_path}\n")
                            outfile.write(f"{'='*80}\n\n")
                            
                            # Write file content
                            outfile.write(content)
                            outfile.write("\n") # Ensure separation
                            
                            print(f"Processed: {rel_path}")
                            
                    except Exception as e:
                        print(f"Error reading {rel_path}: {e}")

if __name__ == "__main__":
    # You can change these paths as needed
    source_directory = "."  # Current directory
    output_filename = "codebase.txt"
    
    print(f"Scanning directory: {os.path.abspath(source_directory)}")
    consolidate_repo(source_directory, output_filename)
    print(f"\nDone! All files consolidated into: {output_filename}")