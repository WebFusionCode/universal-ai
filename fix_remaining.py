import re

with open("backend/main.py", "r") as f:
    code = f.read()

# Fix IMAGE_IMAGE_MODEL_PATH glitch
code = code.replace("IMAGE_IMAGE_MODEL_PATH", "IMAGE_MODEL_PATH")

# Replace MODEL_PATH in endpoints with a dynamic lookup or TABULAR_MODEL_PATH fallback
code = code.replace('    if not os.path.exists(MODEL_PATH):',
                    '    matched_path = TABULAR_MODEL_PATH if os.path.exists(TABULAR_MODEL_PATH) else TIME_SERIES_MODEL_PATH\n'
                    '    if not os.path.exists(matched_path):')

code = code.replace('    model_package = joblib.load(MODEL_PATH)',
                    '    model_package = joblib.load(matched_path)')

# Endpoints where it says if os.path.exists(MODEL_PATH)
code = code.replace('    if os.path.exists(MODEL_PATH):',
                    '    matched_path = TABULAR_MODEL_PATH if os.path.exists(TABULAR_MODEL_PATH) else TIME_SERIES_MODEL_PATH\n'
                    '    if os.path.exists(matched_path):')
code = code.replace('        return FileResponse(MODEL_PATH, filename="best_model.pkl")',
                    '        return FileResponse(matched_path, filename=os.path.basename(matched_path))')

# download_project_files 
code = code.replace('                zipf.write(MODEL_PATH, arcname=os.path.basename(MODEL_PATH))',
                    '                zipf.write(matched_path, arcname=os.path.basename(matched_path))')

# remaining usages like model_version=os.path.basename(MODEL_PATH) - we can just default to TABULAR or "unknown"
code = code.replace('os.path.basename(MODEL_PATH)', '"model.pkl"')

with open("backend/main.py", "w") as f:
    f.write(code)

print("Remaining MODEL_PATHs cleaned up.")
