# **AI-Powered-Insights-Sentiment-and-Topic-Analysis-of-Movie-Reviews**
Most Popular Topics in Positive and Negative Sentiments in Amazon Movies and TV Reviews Dataset

## **Author**  
Rishikesh Kakde  

## **Project Overview**  
This project analyzes sentiment and key topics in Amazon Movies and TV reviews from 2023. It combines **supervised sentiment classification** using a fine-tuned **BERT model** with **unsupervised topic modeling** using **Latent Dirichlet Allocation (LDA)** to uncover prevalent themes in positive and negative reviews.  

The dataset is derived from **17 million Amazon reviews**, with a sample of **1,500 reviews** for analysis and **2,000 reviews** for training. Sentiment labels were generated using the **EMPATH** library, and the trained **BERT model** was used to classify sentiments. **LDA topic modeling** was then applied to extract dominant themes from positive and negative reviews.

---

## **Folder Structure**

### **1. Data Folder**  
Contains datasets used for training and analysis.  
- **`analysis_dataset.csv`** – Dataset used for sentiment analysis.  
- **`training_dataset_labeled.csv`** – Labeled dataset used to train the BERT model, annotated with EMPATH.  
- **`analysis_dataset_with_sentiments.csv`** – Output of the BERT model applied to `analysis_dataset.csv`.  

### **2. Code Folder**  
Jupyter Notebooks used for data processing, model training, and topic modeling.  
- **`Paper3_DataExtraction.ipynb`** – Extracts data from the original JSON file (17M reviews) and creates `analysis_dataset.csv` and `training_dataset_labeled.csv`.  
- **`Training_BERT.ipynb`** – Loads and fine-tunes the **BERT model** on the training dataset.  
- **`Evaluate_BertModel.ipynb`** – Evaluates the BERT model's performance using **accuracy, precision, recall, and F1-score**, with EMPATH labels as the ground truth.  
- **`Topic_Modeling.ipynb`** – Implements **LDA topic modeling** on positive and negative reviews to identify key themes.  

### **3. Visualizations**  
Exports from the **Topic Modeling Notebook** for interactive exploration.  
- **`positive_reviews_topics.html`** – Interactive LDA visualization for positive reviews.  
- **`negative_reviews_topics.html`** – Interactive LDA visualization for negative reviews.  

---

## **Methodology**  

### **Data Preprocessing & Sentiment Analysis**  
1. **Data Extraction** – Extracted **1,500 reviews** for analysis and **2,000 reviews** for BERT training.  
2. **Sentiment Labeling** – Used **EMPATH** to generate initial sentiment labels (positive, negative, neutral).  
3. **Model Training** – Fine-tuned a **pretrained BERT model** on the labeled dataset.  
4. **Evaluation** – Achieved **~80% alignment** with EMPATH sentiment tags.  
5. **Classification** – Applied the trained BERT model to classify sentiments in the analysis dataset.  

### **Topic Modeling with LDA**  
1. **Text Preprocessing** – Tokenization, stopword removal, lemmatization, and **TF-IDF vectorization**.  
2. **Optimal Topic Selection** – Used **coherence score** to determine the best number of topics.  
3. **LDA Modeling** – Applied **Latent Dirichlet Allocation (LDA)** to extract dominant topics.  
4. **Visualization** – Generated **interactive PyLDAvis visualizations** for deeper topic exploration.  

---

## **Key Findings**  

### **Positive Sentiments (37%)**  
**Dominant themes:**  
- **Engaging Storylines** – Emotional depth and strong character development.  
- **Cinematic Quality** – Praise for production, direction, and cinematography.  
- **Entertainment Value** – Viewers enjoyed fun and well-made productions.  

**Top Keywords:** *great, movie, love, character, story*  

### **Negative Sentiments (16%)**  
**Dominant themes:**  
- **Lack of Originality** – Predictable and repetitive plots.  
- **Poor Execution** – Frustration with pacing, acting, and production quality.  
- **Customer Frustration** – Disappointment with content that lacked engagement.  

**Top Keywords:** *quot, boring, show, watch, time*  

---

## **Limitations & Future Work**  
- **Limited Dataset** – Analysis was based on **1,500 reviews**, which may not capture the full dataset's diversity.  
- **Model Performance** – The **BERT model**'s accuracy was limited by **EMPATH label quality** and **computational constraints** (batch size, epochs).  
- **Computational Resources** – A **larger dataset and better hardware** could improve model training and classification performance.  
- **Future Work** – Expand the dataset, apply **advanced deep learning models**, and explore **multilingual reviews** for broader insights.  

---

## **References**  
- Yasen, M., & Tedmori, S. (2019). *Movies Reviews Sentiment Analysis and Classification*. IEEE JEEIT.  
- McAuley, J., & Leskovec, J. (2023). *Amazon Reviews Dataset*. [Dataset Source](https://amazon-reviews-2023.github.io)  
