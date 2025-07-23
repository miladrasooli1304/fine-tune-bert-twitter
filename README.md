# Twitter Sentiment Analysis with BERT

This project fine-tunes a BERT model (`bert-base-uncased`) for multi-class sentiment analysis on a Twitter dataset. The dataset contains tweets labeled with sentiments, and the model is trained to classify tweets into one of six sentiment categories.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Details](#model-details)
- [Results](#results)
- [Visualization](#visualization)
- [Inference](#inference)
- [Saving and Loading the Model](#saving-and-loading-the-model)
- [Contributing](#contributing)
- [License](#license)

## Overview
This project demonstrates how to fine-tune a BERT model for multi-class sentiment classification using the Hugging Face Transformers library. The pipeline includes:
- Data preprocessing and tokenization
- Visualizing data distributions
- Fine-tuning the BERT model
- Evaluating model performance
- Making predictions on new text inputs

The code uses Python libraries such as `transformers`, `datasets`, `scikit-learn`, `evaluate`, `matplotlib`, `seaborn`, and `bertviz`.

## Dataset
The dataset used is `twitter_multi_class_sentiment.csv`, which contains tweets with corresponding sentiment labels. The dataset includes:
- **Columns**: `text` (tweet content), `label` (numerical sentiment label), `label_name` (sentiment category), and `words_per_tweet` (word count per tweet).
- **Sentiment Labels**: Six unique sentiment categories (mapped to numerical labels 0–5).
- **Data Split**: The dataset is split into training (70%), validation (10%), and test (20%) sets, stratified by `label_name`.

### Data Insights
- **Label Distribution**: Visualized using pie and bar charts to show the proportion of each sentiment category.
- **Words per Tweet**: Analyzed with a boxplot to show word count distribution across sentiment categories.
- **Maximum Tweet Length**: The longest tweet contains the highest number of words (index found via `argmax`).

## Installation
To run this project, install the required dependencies:

```bash
pip install transformers datasets accelerate evaluate bertviz umap-learn scikit-learn pandas matplotlib seaborn torch
```

Ensure you have a compatible Python version (3.7 or higher) and access to a GPU for faster training (optional).

## Usage
1. **Load the Dataset**:
   - The dataset is loaded from a CSV file (`twitter_multi_class_sentiment.csv`) using `pandas`.
   - The dataset is split into training, validation, and test sets using `train_test_split` from `scikit-learn`.

2. **Preprocess the Data**:
   - The `AutoTokenizer` from `transformers` is used to tokenize the tweets with padding and truncation.
   - The dataset is converted to a Hugging Face `DatasetDict` for efficient processing.

3. **Fine-Tune the Model**:
   - The `bert-base-uncased` model is fine-tuned using the `Trainer` API from `transformers`.
   - Training arguments include:
     - Epochs: 2
     - Learning Rate: 2e-5
     - Batch Size: 64
     - Evaluation Strategy: Per epoch
     - Output Directory: `bert_base_train_dir`

4. **Evaluate the Model**:
   - The model is evaluated on the test set using accuracy and weighted F1-score metrics.
   - A confusion matrix is generated to visualize classification performance.

5. **Visualize Results**:
   - Data distributions are visualized using pie charts, bar charts, and boxplots.
   - A confusion matrix heatmap is created using `seaborn`.

6. **Inference**:
   - New text inputs can be classified using the fine-tuned model or a `pipeline` for simplified inference.

## Model Details
- **Model**: `bert-base-uncased`
- **Task**: Multi-class text classification
- **Number of Labels**: 6
- **Device**: GPU (if available) or CPU
- **Tokenizer**: `AutoTokenizer` from `transformers`
- **Training Parameters**:
  - Epochs: 2
  - Learning Rate: 2e-5
  - Batch Size: 64
  - Weight Decay: 0.01

The model is fine-tuned with a custom configuration that maps numerical labels to sentiment categories using `label2id` and `id2label` dictionaries.

## Results
The model achieves the following performance on the test set:
- **Accuracy**: [Insert accuracy from `preds_out.metrics`]
- **F1-Score (Weighted)**: [Insert F1-score from `preds_out.metrics`]

A detailed classification report is generated, showing precision, recall, and F1-score for each class. A confusion matrix is also plotted to visualize misclassifications.

## Visualization
The project includes several visualizations:
- **Pie Chart**: Shows the percentage distribution of sentiment labels.
- **Bar Chart**: Displays the count of each sentiment category.
- **Boxplot**: Illustrates the distribution of words per tweet by sentiment label.
- **Confusion Matrix**: Visualizes the model's classification performance across all classes.

## Inference
To classify a new tweet, use the following code:

```python
exam_text = "Hi my name is Milad, Today I am incredibly happy because I read a book"
classifier = pipeline("text-classification", model="twitter_sentiment_bert")
result = classifier(exam_text)
print(result)
```

Alternatively, use the fine-tuned model directly:

```python
def get_pred(text):
    input_encoded = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outs = model(**input_encoded)
    logits = outs.logits
    pred = torch.argmax(logits, dim=1).item()
    return pred, id2label[pred]

pred, label = get_pred(exam_text)
print(f"Prediction: {label} (Label {pred})")
```

## Saving and Loading the Model
The fine-tuned model is saved to `/content/drive/MyDrive/fine-tune Beart/twitter_sentiment_bert` using:

```python
trainer.save_model("/content/drive/MyDrive/fine-tune Beart/twitter_sentiment_bert")
```

To load the model for inference:

```python
from transformers import pipeline
classifier = pipeline("text-classification", model="twitter_sentiment_bert")
```

## Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue on the [GitHub repository](https://github.com/your-username/your-repo-name).

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.