# Chest Cancer Assistant

A machine learning application built on Convolutional Neural Networks capable of detecting and classifying three types of lung cancer:
* Adenocarcinoma
* Large cell carcinoma
* Squamous cell carcinoma

## Description

The Chest Cancer Assistant is a sophisticated medical diagnostic tool that leverages artificial intelligence to analyze CT scans of lungs for cancer detection. Built using modern MLOps practices and deep learning techniques, this system employs a fine-tuned CNN architecture to accurately identify and classify three primary types of lung cancer from medical imaging data. This technology aims to assist healthcare professionals by providing an efficient preliminary screening tool for lung cancer diagnosis.

## Deployed Application

The application is publicly accessible at: [LINK]

## Dataset

This project utilizes a chest CT scan image dataset available on Kaggle:
[Chest CT-Scan Images Dataset](https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images)

## Model Architecture

The application is powered by a modified VGG16 convolutional neural network, which was originally trained to classify 1,000 image categories. More information about the base model can be found on [Wikipedia: VGGNet](https://en.wikipedia.org/wiki/VGGNet).

### Augmentation Layers
To enhance model robustness and generalization, the following image augmentation techniques were implemented:
* Random Flip
* Random Rotation
* Random Zoom
* Random Contrast
* Random Brightness

### Model Customization
For this specific application, the original classification layers of VGG16 were replaced with a custom architecture:
* Flatten layer
* Dense layer (2048 neurons, ReLU activation)
* Dropout layer (0.5)
* Dense layer (1024 neurons, ReLU activation)
* Dropout layer (0.5)
* Dense layer (512 neurons, ReLU activation)
* Dropout layer (0.5)
* Dense layer (258 neurons, ReLU activation)
* Output layer for cancer type classification

This custom architecture was fine-tuned on the chest CT scan dataset to optimize performance for lung cancer detection.

## Technology Stack

* **Python**: Core programming language
* **TensorFlow & Keras**: Deep learning framework
* **Flask**: Web application framework
* **DVC**: Data version control
* **DagsHub**: MLOps platform
* **MLFlow**: ML lifecycle management
* **Docker**: Containerization
* **AWS**: Cloud deployment

## Getting Started

### Prerequisites
* Python 3.8+
* Git
* Docker (for containerized deployment)

### Installation

1. Clone the repository:

   ```bash
    git clone https://github.com/grzegorz-gomza/Chest_Cancer_Classification_MLOps.git
    cd Chest_Cancer_Classification_MLOps
   ```

2. Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Running Locally

0. Secrets needd to be stored in ```.env``` file:
    ```
    KAGGLE_USERNAME = ''
    KAGGLE_KEY = ''

    MLFLOW_TRACKING_URI = ''
    MLFLOW_TRACKING_USERNAME = ''
    MLFLOW_TRACKING_PASSWORD = ''
    ```
1. Training parameters are stored in ```params.yaml``` file
2. Neural Network Layers are defined in ```src/ChestCancerClassifier/components/prepare_base_model.py```
3. Activate the training pipeline
3.1. without dvc

    ```bash
    python main.py
    ```

    3.2. with dvc

    ```bash
    dvc init
    dvc repro
    ```

    to check the pipeline:

    ```bash
    dvc dag
    ```
4. Check the MLFlow Experiments

5. Start the Flask application:
    ```bash
    python app.py
    ```

6. Access the application at `http://localhost:5000`

### Docker Deployment
This step is not required in order to deploy the app on AWS.

1. Build the Docker image:
    ```bash
    docker build -t chest-cancer-assistant .
    ```

2. Run the container:
    ```bash
    docker run -p 1988:1988 chest-cancer-assistant
    ```

## AWS Deployment with GitHub Actions

This project uses GitHub Actions for CI/CD deployment to AWS. Follow these steps to replicate the deployment:

### 1. AWS Setup

1. **Create IAM User** with the following policies:
   - AmazonEC2ContainerRegistryFullAccess
   - AmazonEC2FullAccess

2. **Create ECR Repository** to store the Docker image
   - Note the URI: `[account-id].dkr.ecr.[region].amazonaws.com/[repo-name]`

3. **Launch EC2 Instance** (Ubuntu)
   - Ensure security group allows inbound traffic on port 5000

4. **Install Docker on EC2**:
   ```bash
   sudo apt-get update -y
   sudo apt-get upgrade
   curl -fsSL https://get.docker.com -o get-docker.sh
   sudo sh get-docker.sh
   sudo usermod -aG docker ubuntu
   newgrp docker
   ```

### 2. GitHub Setup

1. **Configure EC2 as Self-hosted Runner**:
   - Go to repository settings → Actions → Runners
   - Add new self-hosted runner
   - Follow the instructions to set up on your EC2 instance

2. **Set GitHub Secrets**:
   - `AWS_ACCESS_KEY_ID`: Your IAM user access key
   - `AWS_SECRET_ACCESS_KEY`: Your IAM user secret key
   - `AWS_REGION`: e.g., us-east-1
   - `AWS_ECR_LOGIN_URI`: Your ECR URI without the repository name
   - `ECR_REPOSITORY_NAME`: Your ECR repository name

The GitHub Actions workflow will automatically build the Docker image, push it to ECR, and deploy it to your EC2 instance whenever changes are pushed to the main branch.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

* ```entbappy``` for providing the tutorial