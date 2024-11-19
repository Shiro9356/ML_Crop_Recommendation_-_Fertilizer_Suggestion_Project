
from nbconvert import ScriptExporter

# Path to the notebook
input_notebook = "fertilizer_suggestion_code.ipynb"
output_script = "fertilizer_suggestion_code.py"

# Convert the notebook to a script
exporter = ScriptExporter()
script, _ = exporter.from_filename(input_notebook)

# Save the script
with open(output_script, "w") as f:
    f.write(script)

print("Notebook converted to Python script successfully!")
