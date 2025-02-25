import os
from pathlib import Path
import logging

# Seting up logging for the project
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s')

project_name = "BreastCancerClassifier"

list_of_files = [
    ".github/workflows/.gitkeep",
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configuration.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/model/"
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/constants/__init__.py",
    "config/config.yaml",
    "dvc.yaml",
    "params.yaml",
    ".env"
    "setup.py",
    "tests/test.ipynb", # Notebook for testing code during devopment
    "requirements.txt", # Requirements file created beforehand 
    "frontend/index.html"
]

# Creating a directory for the project
for filePath in list_of_files:
    filePath = Path(filePath) 
    fileDir, fileName = os.path.split(filePath)

    # fileDir is not empty - case: creating a directory and a file
    if fileDir !="": 
        os.makedirs(fileDir, exist_ok=True)
        logging.info(f"Creating directory: {fileDir} for the file: {fileName}")

    # fileDir is empty - case: creating a file
    if (not os.path.exists(filePath)) or (os.path.getsize(filePath) == 0):
        with open(filePath, "w") as f:
            pass
            logging.info(f"Creating empty file: {filePath}")

    else:
        logging.info(f"{fileName} is already exists")
