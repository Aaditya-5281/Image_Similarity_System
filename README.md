# Image Similarity System with Transformers

This project implements an image similarity system using transformers from the Transformers library. It allows you to find visually similar images from a dataset given a query image.

## Features

- Employs a pre-trained vision transformer model (`nateraw/vit-base-beans`) for efficient image feature extraction.
- Leverages the FAISS library for fast and scalable nearest neighbor search.
- Provides functions for:
  - Extracting image embeddings
  - Retrieving top-k similar images from the dataset
  - Evaluating the accuracy of the system

## Installation

1. Clone this repository:

    ```bash
    git clone https://github.com/your-username/image-similarity-system.git
    cd image-similarity-system
    ```

2. Create a virtual environment (recommended):

    ```bash
    python -m venv venv
    ```

3. Activate the virtual environment:

    - On Windows:
  
      ```bash
      venv\Scripts\activate
      ```
  
    - On macOS/Linux:
  
      ```bash
      source venv/bin/activate
      ```

4. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Data Preparation:

- Ensure you have access to an image dataset with labels (e.g., the `beans` dataset used in this example).
- This code assumes the dataset is loaded using the `datasets` library. Adapt the loading logic if your dataset format differs.

## Dependencies

- `pyarrow==14.0.1`
- `cudf-cu12` (if using GPU)
- `ibis-framework`
- `transformers`
- `datasets`
- `faiss-cpu` (CPU version) or `faiss-gpu` (GPU version)
- `Pillow` (for image manipulation)

## Important Notes

- This code uses a pre-trained model for image classification (e.g., the "beans" dataset). For your specific use case, you might need to fine-tune or replace the model with a suitable one.
- The chosen model's performance might vary depending on the dataset and similarity criteria.
- Consider incorporating error handling for potential issues during data loading or image processing.
