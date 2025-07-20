# Mushroom Classification Using Machine Learning

This project is a comprehensive machine learning pipeline to classify mushrooms as **edible or poisonous** based on their physical characteristics. It covers everything from data preprocessing and exploratory data analysis to model training, evaluation, and deployment-ready prediction on unseen data.

---

## Dataset

- **File Used:** `secondary_data.csv`
- **Source:** [UCI Mushroom Dataset](https://archive.ics.uci.edu/ml/datasets/Mushroom)
- **Description:** The dataset contains various categorical features describing mushroom properties like cap shape, surface, color, gill type, stalk dimensions, and more. Each mushroom is labeled as either edible (`e`) or poisonous (`p`).

---

## 🧪 Features Used

After applying feature selection using **Random Forest** and **SelectKBest**, the following features were selected for the final model:

- `cap-diameter`, `stem-width`, `stem-height`, `stem-color_w`, `gill-spacing_d`,  
  `does-bruise-or-bleed_t`, `gill-color_w`, `gill-attachment_p`, `has-ring_t`,  
  `cap-shape_x`, `gill-attachment_x`, `gill-attachment_d`, `ring-type_z`,  
  `cap-surface_t`, `cap-surface_s`, `cap-surface_k`, `cap-color_e`, `cap-color_r`,  
  `gill-attachment_e`, `gill-color_n`, `stem-color_w`, `habitat_g`, `season_w`

---

##  Tools & Libraries

- Python 3.x
- pandas, numpy, matplotlib, seaborn
- scikit-learn
- joblib (for saving model)

---

## Machine Learning Models

We tested and evaluated the following classifiers:

- Logistic Regression
- Support Vector Machine (SVM)
- Decision Tree
- Random Forest
- K-Nearest Neighbors (KNN)
- Naive Bayes
- MLP Classifier
- Gradient Boosting
- AdaBoost

Each model was evaluated using:

- **Accuracy**
- **Precision, Recall, F1-Score**
- **Confusion Matrix**
- **ROC-AUC Curve**

---

## Project Workflow

1. **Data Collection & Loading**
2. **Data Cleaning**
   - Handled missing values
   - Removed columns with >60% missing data
3. **Exploratory Data Analysis (EDA)**
   - Histograms, Boxplots, KDEs, Correlation Heatmaps
4. **Feature Engineering**
   - Label Encoding, One-Hot Encoding
5. **Feature Selection**
   - Random Forest Feature Importance
   - SelectKBest (Chi-squared)
6. **Data Splitting & Scaling**
7. **Model Building & Comparison**
8. **Hyperparameter Tuning**
9. **Model Saving (with LabelEncoder)**
10. **Testing on Unseen Data**

---

##  Results

- **Best Model:** Random Forest / Gradient Boost
- **Accuracy Achieved:** ~99%
- **Important Features:** cap-diameter, stem-width, gill-color, gill-spacing, ring-type, etc.

---

## Unseen Data Prediction

You can test the saved model using unseen mushroom features with the `predict_unseen.py` script. Ensure you load the same `LabelEncoder` for decoding predictions.

---

## Files in This Repo

- `secondary_data.csv` — original dataset
- `mushroom_classification.ipynb` — full notebook
- `best_classification_model.pkl` — saved trained model
- `README.md` — this file

---

## Future Work

- Add deep learning models (e.g. neural networks)
- Build a web interface or API for predictions
- Automate model retraining with new data

---

##  Author

**Sheji Adhil**  


## 📝 License

This project is open-source and available under the [MIT License](LICENSE).

