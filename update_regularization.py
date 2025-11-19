
import json

def modify_notebook_regularization():
    notebook_path = 'seir_dengue_workflow.ipynb'
    
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
    except Exception as e:
        print(f"Error reading notebook: {e}")
        return

    # We want to update "l2_regularization": 0.01 to 0.1 in Cell 2 (Model Configuration)
    config_cell_found = False
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if 'config = {' in source and '"l2_regularization": 0.01' in source:
                print("Found config cell.")
                new_source = source.replace('"l2_regularization": 0.01', '"l2_regularization": 0.1')
                cell['source'] = [line for line in new_source.splitlines(keepends=True)]
                config_cell_found = True
                break
    
    if config_cell_found:
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print("Notebook updated: Increased L2 Regularization to 0.1")
    else:
        print("Config cell not found or already updated.")

if __name__ == "__main__":
    modify_notebook_regularization()

