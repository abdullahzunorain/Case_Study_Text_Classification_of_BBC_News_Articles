## BBC News Articles Text Classification

### Overview

This project focuses on classifying BBC News articles into different categories using various machine learning techniques. The goal is to accurately predict the category of a news article based on its content. The process involves data preprocessing, feature extraction, model training, evaluation, and comparison of multiple classification algorithms.

### Dataset

The dataset consists of BBC News articles, each labeled with a category. The dataset is split into training and testing sets to evaluate the performance of different models.

### Steps Involved

#### 1. Data Preprocessing

Data preprocessing is crucial for preparing the text data for model training. The following steps were performed:

- **Loading the Dataset:** The dataset was loaded and inspected for any issues.
- **Text Normalization:** Text normalization included converting text to lowercase, removing punctuation, and filtering out non-alphabetic characters.
- **Tokenization:** The text was split into individual words (tokens).
- **Stemming and Lemmatization:** Stemming reduced words to their root forms, while lemmatization reduced words to their base or dictionary form.
- **Stop Words Removal:** Commonly used words (stop words) that do not carry significant meaning were removed.

#### 2. Feature Extraction

Feature extraction transforms the textual data into numerical features suitable for machine learning algorithms. The following methods were used:

- **Bag of Words (BoW):** Represents text as a fixed-size vector, counting the frequency of words in each document.
- **TF-IDF:** Stands for Term Frequency-Inverse Document Frequency, which accounts for the importance of words in the dataset.

#### 3. Model Selection and Training

Multiple machine learning models were implemented and trained on the preprocessed data. The models used include:

- **Logistic Regression**
- **Naive Bayes**
- **Support Vector Machine (SVM)**
- **Random Forest**
- **Decision Tree**
- **K-Nearest Neighbors (KNN)**
- **Multilayer Perceptron (MLP)**

Each model was trained using both the BoW and TF-IDF feature representations.

#### 4. Model Evaluation

The models were evaluated using the testing set. The primary metric used for evaluation was accuracy. Additional metrics like precision, recall, and F1-score were also calculated to provide a more comprehensive assessment of the models' performance.

#### 5. Results and Analysis

The accuracy of each model, along with other evaluation metrics, was compared. The results were as follows:
Here's a table summarizing the performance metrics (accuracy) of each model using unigrams, bigrams, and trigrams:

| Model                | Unigrams Accuracy | Bigrams Accuracy | Trigrams Accuracy |
|----------------------|-------------------|------------------|-------------------|
| Naive Bayes          | 0.9832            | 0.9832           | 0.9866            |
| Logistic Regression  | 0.9664            | 0.9631           | 0.9631            |
| Support Vector Machine (SVM) | 0.9530    | 0.9497           | 0.9329            |
| Random Forest        | 0.9698            | 0.9430           | 0.9530            |
| K-Nearest Neighbors  | 0.6141            | 0.4664           | 0.3960            |
| Decision Tree        | 0.8591            | 0.8691           | 0.8523            |
| Perceptron           | 0.9698            | 0.9698           | 0.9698            |


### Analysis
The results from the predictions on new text data indicate that the models had varying success in accurately classifying the news articles. While some models like Naive Bayes and Perceptron managed to correctly classify multiple categories, others like Logistic Regression, SVM, Random Forest, KNN, and Decision Tree struggled, often predicting "sport" for a majority of the texts. This suggests that these models may have overfitted on certain categories or lacked sufficient differentiation capability.

### Conclusion
This evaluation highlights the challenges of text classification, particularly the need for robust models that can generalize well to unseen data. Further tuning and potentially integrating advanced methods like deep learning could improve accuracy and category differentiation. The results also underscore the importance of diverse training data to cover a broad range of topics for more reliable predictions.


### Future Work

Future work could involve:

- Exploring deep learning models, such as Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs), for text classification.
- Performing hyperparameter tuning to optimize model performance.
- Expanding the dataset to include more diverse categories and a larger number of articles.

### How to Use

1. **Setup:** Clone the repository and install the required packages.
2. **Data Preprocessing:** Use the provided scripts to preprocess the data.
3. **Model Training:** Train the models using the provided training scripts.
4. **Evaluation:** Evaluate the models and compare their performance.

### Dependencies

- Python 3.7+
- scikit-learn
- NLTK
- pandas
- NumPy

### Contact

For any questions or suggestions, feel free to contact me (abdullahzunorain2@gmail.com).
