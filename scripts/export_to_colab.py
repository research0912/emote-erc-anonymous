import os
import zipfile
from pathlib import Path

def zip_project(output_filename="erc_colab_export.zip"):
    project_root = Path(__file__).resolve().parent.parent
    output_path = project_root / output_filename
    
    # Exclude patterns
    exclude_dirs = {
        ".git",
        "__pycache__",
        "venv",
        ".venv",
        ".ipynb_checkpoints",
        ".gemini",
        "outputs", # Exclude past large output runs
    }
    
    exclude_extensions = {
        ".zip",
        ".tar",
        ".gz",
        ".pt", # Exclude large local checkpoints (they should be reloaded or are too big)
        ".pth"
    }

    print(f"Zipping project at: {project_root}")
    print(f"Excluding directories: {exclude_dirs}")
    print(f"Excluding extensions: {exclude_extensions}")

    count = 0
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for root, dirs, files in os.walk(project_root):
            # Prune directories in-place to avoid traversing excluded paths
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            
            for file in files:
                file_path = Path(root) / file
                
                # Skip the export zip itself and files with excluded extensions
                if file_path.suffix in exclude_extensions or file_path.name == output_filename:
                    continue
                    
                # Compute relative path for zip matching
                rel_path = file_path.relative_to(project_root)
                zip_file.write(file_path, rel_path)
                count += 1

    print(f"Successfully packaged {count} files into: {output_path}")
    print(f"Zip File size: {output_path.stat().st_size / (1024 * 1024):.2f} MB")
    print("You can now download this zip file and upload it directly to your Google Colab / Google Drive session!")

if __name__ == "__main__":
    zip_project()
