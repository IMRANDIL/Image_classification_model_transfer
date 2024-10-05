# Image Classification Model Training

![Figure_001](https://github.com/user-attachments/assets/7857f003-656b-41d4-ba76-30172133ae62)


This project focuses on building an image classification model using a custom dataset. The repository is structured to manage configurations, training logs, and model storage, with Python scripts for training and utility functions.

## Project Structure

```bash
IMAGE_CLASSIFICATION_PROJECT/
├── config/         # Configuration files
├── data/           # Training and testing datasets
├── logs/           # Logs generated during model training
├── models/         # Saved models after training
├── src/            # Source code folder
│   ├── main.py     # Main script to run the model training
│   ├── train.py    # Contains the function to train the model
│   └── utils.py    # Utility functions used across the project
├── venv/           # Python virtual environment (optional)
├── .gitignore      # Ignored files and folders
├── README.md       # Project documentation (this file)
└── requirements.txt # Python dependencies
```

## Getting Started

### Prerequisites

To run the project, ensure you have Python installed. Optionally, it's recommended to use a virtual environment.

1. **Clone the repository**:
   ```bash
   git clone https://github.com/IMRANDIL/Image_classification_model_transfer.git
   cd image-classification-project
   ```

2. **Create and activate a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # For Linux/Mac
   .\venv\Scripts\activate   # For Windows
   ```

3. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Project

To train the image classification model, execute the `main.py` file located in the `src/` directory.

```bash
cd src
python main.py --config ..\config\config.yaml
```

This will start the training process by calling the `train_the_model()` function from `train.py`.

### Example: main.py
```python
from train import train_the_model

def main():
    train_the_model()

if __name__ == "__main__":
    main()
```

## Configuration

You can adjust various hyperparameters and model configurations through the files in the `config/` directory. Ensure that any necessary paths (e.g., dataset paths) are correctly specified in these configuration files before running the model.

## Logging

Training logs and metrics will be stored in the `logs/` directory. These logs can be used for monitoring and evaluating the model’s performance over time.

## Saving Models

Trained models will be saved in the `models/` directory, which can be used for later evaluation or deployment.

## Dataset

Place your training and testing datasets in the `data/` directory. Ensure your dataset follows the required structure for image classification (e.g., class-based folders or appropriate labeling).

Example structure:
```
data/
├── train/
│   ├── class_1/
│   ├── class_2/
├── test/
│   ├── class_1/
│   ├── class_2/
```

## Utilities

The `utils.py` file contains utility functions that assist in tasks such as data preprocessing, image augmentation, and other helper operations needed during the model training process.

## Contributing

Contributions are welcome! If you'd like to contribute, please fork the repository and submit a pull request with your changes. Ensure your code adheres to the project's coding standards.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

