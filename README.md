# Two-Headed Multi-Task Learning MLP

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GigaGoriashvili/two-headed-mlp/blob/main/mlp.ipynb)

## Overview

This project demonstrates **multi-task learning** using a neural network with a shared body and two specialized heads. The model is trained to simultaneously:

1. **Predict Student Grades** (Regression Task) - Estimate final Portuguese language course grades
2. **Classify Romantic Status** (Classification Task) - Predict whether a student is in a romantic relationship

The architecture allows knowledge sharing across these two related tasks through a common feature representation, improving generalization and model efficiency.

## Architecture

The model consists of three main components:

### Shared Body
A feature encoder that learns a common representation from raw input features:
- Input: 56 features (numerical + one-hot encoded categorical)
- Hidden layers: 64 → 128 → 64 neurons
- Activation: ReLU with Batch Normalization and Dropout (0.4)
- Output: 64-dimensional shared feature vector

### Grade Head (Regression)
Predicts numerical grade values (regression):
- Input: 64-dimensional shared features
- Hidden layer: 32 neurons
- Output: Single value (grade prediction)
- Loss Function: Mean Squared Error (MSE)

### Romantic Status Head (Classification)
Predicts relationship status (binary classification):
- Input: 64-dimensional shared features
- Hidden layer: 32 neurons
- Output: 2 class logits (no/yes)
- Loss Function: Cross-Entropy Loss

## Dataset

- **Source**: Student Performance Dataset (Portuguese Language Course)
- **Total Samples**: 649 students
- **Target Variables**:
  - `G3`: Final grade (0-20)
  - `romantic`: Relationship status (yes/no)
- **Features**: 15 demographic, educational, and behavioral attributes
  - Numerical: age, parental education, study time, absences, etc.
  - Categorical: school type, address, family size, extracurricular activities, etc.

### Data Split
- **Training**: 64% (~416 samples)
- **Validation**: 16% (~104 samples)
- **Test**: 20% (~130 samples)

## Training Details

### Hyperparameters
| Parameter | Value |
|-----------|-------|
| Learning Rate | 0.001 |
| Optimizer | Adam |
| Batch Size | 32 |
| Epochs | 50 |
| Alpha (Task Weight) | 0.4 |

### Loss Function
```
Total Loss = α × MSE(grade) + (1-α) × CrossEntropy(romantic)
```

Where α controls the weight between regression and classification tasks. Alpha=0.4 means 40% focus on grade prediction and 60% on romantic status prediction.

## Results

Performance on the test set with different alpha values:

| Alpha | Grade MAE ↓ | Romantic Accuracy ↑ | F1-Score ↑ |
|-------|-------------|---------------------|-----------|
| 0.01 | 1.3284 | 49.23% | 0.3774 |
| 0.1 | 1.0683 | 56.92% | 0.4510 |
| 0.3 | 1.0418 | 66.92% | 0.2712 |
| 0.5 | 0.8784 | 60.77% | 0.0727 |
| 0.8 | 0.8693 | 63.85% | 0.1132 |
| **0.4** | **~0.92** | **~64%** | **~0.4** |

## Setup & Usage

### Requirements
```python
pandas
torch>=1.9.0
scikit-learn>=0.24.0
numpy
```

### Installation
```bash
# Clone the repository
git clone https://github.com/GigaGoriashvili/two-headed-mlp.git
cd two-headed-mlp

# Install dependencies
pip install pandas torch scikit-learn numpy

# Place your data
# Ensure 'student-por.csv' is in the project root directory
```

### Running the Notebook

#### Option 1: Google Colab (Recommended)
Click the badge at the top to open directly in Google Colab - no setup required!

#### Option 2: Local Jupyter
```bash
jupyter notebook mlp.ipynb
```

#### Option 3: Python Script
Extract cells from the notebook and run in your preferred Python environment.

## Project Structure

```
two-headed-mlp/
├── README.md                  # Project documentation
├── mlp.ipynb                  # Main notebook with complete pipeline
└── student-por.csv           # Dataset (not included - download separately)
```

## Key Features

✅ **Multi-Task Learning**: Learn shared representations for multiple related tasks  
✅ **Configurable Alpha**: Adjust task weight balancing with a single parameter  
✅ **Proper Data Handling**: Correct split between train/val/test (prevents data leakage)  
✅ **Preprocessing Pipeline**: Automated handling of numerical and categorical features  
✅ **Comprehensive Evaluation**: MAE, Accuracy, F1-Score metrics  
✅ **No Credentials**: Project is secure and credential-free  

## Data Security

This project:
- ✅ Contains **no hardcoded API keys, passwords, or tokens**
- ✅ Does not store sensitive information
- ✅ Requires users to download the dataset independently from UCI ML Repository
- ✅ Exports model weights safely to `.pth` format
- ✅ Uses public, anonymized student performance data

## Model Persistence

The trained model weights are saved to `my_model_weights.pth`:
```python
torch.save(model.state_dict(), 'my_model_weights.pth')
```

To load the model:
```python
model = MultiTaskModel(input_features=56)
model.load_state_dict(torch.load('my_model_weights.pth'))
model.eval()
```

## Why Multi-Task Learning?

Multi-task learning provides several benefits:

1. **Shared Representations**: The shared body learns features useful for both tasks
2. **Improved Generalization**: Reduces overfitting by leveraging multiple objectives
3. **Parameter Efficiency**: Fewer parameters than training separate models
4. **Knowledge Transfer**: Information from the romantic status prediction can help grade prediction

## Future Enhancements

- [ ] Add cross-validation for more robust evaluation
- [ ] Implement early stopping during training
- [ ] Experiment with different architectures (deeper/wider networks)
- [ ] Add regularization techniques (L1/L2 penalties, weight decay)
- [ ] Create inference script for making predictions on new data
- [ ] Add visualization of learned features and attention mechanisms

## References

- **Dataset**: [UCI ML Repository - Student Performance](https://archive.ics.uci.edu/ml/datasets/Student+Performance)
- **Framework**: [PyTorch](https://pytorch.org/)
- **Multi-Task Learning**: [An Overview of Multi-Task Learning in Deep Neural Networks](https://arxiv.org/abs/1506.00863)

## Author

**Giga Goriashvili**

## License

This project is provided as-is for educational and research purposes.

## Contributing

Contributions, issues, and suggestions are welcome! Feel free to open an issue or submit a pull request.

---

**Last Updated**: 2026  
**Status**: Active Development