# RoBERTa-based Sentiment Analysis

This project implements sentiment analysis on Twitter data using the RoBERTa transformer model. It includes data preprocessing, model training, and prediction scripts, with a focus on classifying tweets as positive or negative.

## 📁 Project Structure

```
roBERTa/
├── my_model.keras                  # Trained RoBERTa Keras model
├── sentiment_analysis(roBERTa).ipynb  # Main Jupyter notebook for data processing and training
├── test.py                         # Script to test the trained model on new text
├── README.md                       # Project documentation (this file)
```

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/kaushalnandaniya/Twitter-Sentiment-Analysis-Using-roBERTa.git
cd roBERTa
```

### 2. Install Dependencies

Make sure you have Python 3.8+ and pip installed. Then run:

```bash
pip install -r ../requirements.txt
```

**Key dependencies:**
- transformers
- tensorflow
- pandas
- numpy
- scikit-learn

### 3. Download the Dataset

The notebook downloads the [Sentiment140 dataset](https://www.kaggle.com/kazanova/sentiment140) using Kaggle's API. Place your `kaggle.json` in the working directory and run the notebook to fetch and extract the data.

## 🏗️ Training the Model

Open and run `sentiment_analysis(roBERTa).ipynb` to:
- Preprocess and clean the Twitter data
- Tokenize using RoBERTa tokenizer
- Train a RoBERTa-based model using TensorFlow/Keras
- Save the trained model as `my_model.keras`

## 🔍 Making Predictions

Use `test.py` to run sentiment predictions on new text:

```bash
python test.py
```

- The script loads the trained model and tokenizer
- Modify the `text` variable in `test.py` to test your own sentences
- Output: Sentiment label (Positive/Negative) and prediction probabilities

## 🧠 Model Details

- **Architecture:** RoBERTa-base transformer (via Hugging Face Transformers)
- **Input:** Cleaned tweet text, tokenized to max length 128
- **Output:** Binary sentiment (Positive/Negative)
- **Training:** Fine-tuned on Sentiment140 with class balancing and preprocessing

## 📦 Requirements

See `../requirements.txt` for all dependencies.

## ✍️ Notes
- The model file (`my_model.keras`) is large (~476MB).
- For best results, use a GPU-enabled environment for training.
- The notebook contains all steps from data download to evaluation.

## 📄 License

This project is for educational purposes. Please check dataset and model licenses before commercial use.

## ⬇️ Download the Model

Before running predictions, download the trained model from Google Drive:

```bash
pip install gdown

gdown --id 1cAiiCXYCXl6RnQaSUbdxiSmOUHchzYDO -O my_model.keras
```

This will save the model as `my_model.keras` in your current directory. 
