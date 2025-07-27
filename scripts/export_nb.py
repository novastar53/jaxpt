import nbformat
from nbconvert import PythonExporter
import sys
import os

def convert_notebook_to_script(ipynb_path, output_path=None):
    # Read notebook
    with open(ipynb_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    # Convert to Python
    exporter = PythonExporter()
    script, _ = exporter.from_notebook_node(nb)

    # Determine output path
    if output_path is None:
        output_path = os.path.splitext(ipynb_path)[0] + ".py"

    # Write .py file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(script)

    print(f"Converted: {ipynb_path} → {output_path}")

# --- Usage example ---
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_ipynb.py notebook.ipynb")
        sys.exit(1)

    ipynb_file = sys.argv[1]
    convert_notebook_to_script(ipynb_file)