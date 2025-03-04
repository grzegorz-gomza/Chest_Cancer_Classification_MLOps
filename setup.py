import setuptools

with open("README.md", "r", encoding="utf-8") as file:
    long_description = file.read()

__version__ = "0.0.0"

REPO_NAME = "Chest_Cancer_Classification_MLOps" # Be sure that it matches the repo name
AUTHOR_USER_NAME = "grzegorz-gomza" # Be sure that it matches the github username
SRC_REPO = "ChestCancerClassifier" # Be sure that it matches the name in src/ folder
AUTHOR_EMAIL = "gomza.datascience@gmail.com"

setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="The Chest Cancer Classification MLOps project",
    long_description=long_description,
    long_description_content="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src") # finds packages in src folder and includes them
)