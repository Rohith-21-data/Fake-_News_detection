# Fake News Detection

## Overview
Fake news is a growing issue in today's digital age, impacting public opinion and social trust. This project focuses on detecting fake news articles using machine learning and deep learning techniques. The system classifies news articles as real or fake based on their textual content, leveraging various models such as Logistic Regression, Naïve Bayes, Random Forest, LSTM, and CNN.

## Project Structure
```
Fake-News-Detection/
│-- final_fake__desertation_(2).ipynb  # Jupyter Notebook with code implementation
│-- FAke_news_detection_report.pdf      # Project report
│-- dataset/                            # Data used for training/testing (if applicable)
│-- models/                             # Trained machine learning models (if applicable)
│-- README.md                           # Project documentation
```

## Dataset
The dataset used for training and testing was obtained from Kaggle (ISOT Fake News Dataset). It includes two CSV files:
- `Fake.csv` - Contains fake news articles.
- `True.csv` - Contains real news articles.

Each article consists of the following attributes:
- **Title**: The headline of the article.
- **Text**: The body content of the news article.
- **Subject**: The category of news (e.g., politics, world news).
- **Date**: The publication date of the article.

## Methodology
### Data Preprocessing
- Tokenization
- Stopword Removal
- Lemmatization
- TF-IDF Vectorization
- Padding (for deep learning models)

### Machine Learning Models Used
1. **Logistic Regression**
2. **Multinomial Naïve Bayes**
3. **Random Forest** (Best performing classifier)
4. **Long Short-Term Memory (LSTM)**
5. **Convolutional Neural Networks (CNN)**

### Hyperparameter Tuning for LSTM & CNN
To enhance the model performance, hyperparameter tuning was performed for LSTM and CNN. The following parameters were optimized:

#### LSTM Hyperparameter Tuning:
- **Batch Size:** Tested values [32, 64, 128]
- **Embedding Dimension:** 100
- **LSTM Units:** 150
- **Dropout:** 0.4
- **Epochs:** 5
- **Optimizer:** Adam

#### CNN Hyperparameter Tuning:
- **Batch Size:** Tested values [32, 64, 128]
- **Embedding Dimension:** 150
- **Conv1D Filters:** 128
- **Kernel Size:** 5
- **Activation Function:** ReLU
- **Pooling Layer:** GlobalMaxPooling1D
- **Optimizer:** Adam

### Evaluation Metrics
The models were evaluated using:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**
- **ROC Curve & AUC Score**

## Results
- The **Random Forest classifier** achieved the highest accuracy of **99.84%**.
- **CNN performed better than LSTM**, achieving **98.98% accuracy**.
- **Larger batch sizes and optimized hyperparameters improved deep learning model performance.**

## Future Improvements
- Implementing **BERT** or **GPT-based transformers** for enhanced text classification.
- Incorporating **social media metadata** to improve detection accuracy.
- Utilizing **hybrid models** combining traditional ML and deep learning approaches.

## Installation & Usage
### Prerequisites
Ensure you have the following installed:
- Python 3.x
- Jupyter Notebook
- Required libraries (install using the command below):
  ```sh
  pip install pandas numpy sklearn tensorflow keras matplotlib seaborn nltk
  ```

### Running the Model
1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/Fake-News-Detection.git
   cd Fake-News-Detection
   ```
2. Open the Jupyter Notebook:
   ```sh
   jupyter notebook final_fake__desertation_(2).ipynb
   ```
3. Run the notebook cells to train and evaluate the models.

