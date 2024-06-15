# Cat and Dog Image Classification Using CLIP Embeddings

## Project Overview

This project leverages the power of CLIP (Contrastive Language-Image Pre-Training) embeddings for image classification. The objective is to build a system that can accurately differentiate between images of cats and dogs using CLIP embeddings as features and the Fisher algorithm for classification.

## Dataset and CLIP Embedding Extraction

### Steps:
1. **Data Splitting:**
   - Split the dataset into training and testing sets with 20% of the data reserved for testing.

2. **CLIP Installation and Setup:**
   - Install the CLIP repository from GitHub as a Python package along with any additional dependencies.
   - Load the CLIP model using a deep learning framework like PyTorch (use `clip.load()` method).

3. **Embedding Extraction:**
   - Use the CLIP model to extract embeddings for each image in the dataset (both train and test sets).
   - Pass each image through the CLIP model to retrieve its embedding vector using the `model.encode_image(image: Tensor)` method.
   - Organize the extracted CLIP embeddings into a feature matrix.

## Fisher Algorithm

### Implementation:
1. **Training:**
   - Implement the Fisher algorithm using the training set’s CLIP embeddings as features and associated labels.
   - Use the following equation for Fisher’s Linear Discriminant: \( w = C \cdot S^{-1} \cdot (m_2 - m_1) \), where \( C=0.1 \).

2. **Testing and Visualization:**
   - Test the model using the testing dataset.
   - Visualize the classification results using a confusion matrix to gain insights into the model’s behavior.

## Model Evaluation

### Metrics:
- Calculate accuracy, precision, recall, and F1-score to assess the model’s performance.
- Visualize the classification results using a confusion matrix.

## Bonus Task

### Experimentation:
- Alter Fisher’s Linear Discriminant equation and try three different C values.
- Determine the best C value for the model.

## Deliverables

1. **Python Code:**
   - Well-commented and organized Python code implementing the entire workflow.

2. **Report:**
   - Explanation of each step, method, results, and model performance insights, including visualizations (e.g., Confusion Matrix).
