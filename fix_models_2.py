import re

with open("backend/main.py", "r") as f:
    code = f.read()

# Time series replacements (around line 2049)
ts_start = code.find('elif detected_type == "time_series":')
tab_start = code.find('elif detected_type == "tabular":')

ts_block = code[ts_start:tab_start]
ts_block = ts_block.replace('MODEL_PATH', 'TIME_SERIES_MODEL_PATH')
code = code[:ts_start] + ts_block + code[tab_start:]

# Tabular replacements (from tab_start to the end of auto_train)
tab_start = code.find('elif detected_type == "tabular":')
risk_start = code.find('def risk_analysis(prob_array):')

tab_block = code[tab_start:risk_start]
tab_block = tab_block.replace('MODEL_PATH', 'TABULAR_MODEL_PATH')
code = code[:tab_start] + tab_block + code[risk_start:]

# Image block (from image block to time series block)
img_start = code.find('if detected_type == "image":')
ts_start = code.find('elif detected_type == "time_series":')

img_block = code[img_start:ts_start]
img_block = img_block.replace('MODEL_PATH', 'IMAGE_MODEL_PATH')
code = code[:img_start] + img_block + code[ts_start:]

with open("backend/main.py", "w") as f:
    f.write(code)

print("Split out model paths successfully in auto_train.")
