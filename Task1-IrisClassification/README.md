# 🌸 Iris Flower Classification with Machine Learning

**AICTE OASIS INFOBYTE - Data Science Internship Task 1**

## 📝 Project Description

This project builds a machine learning model to classify iris flowers into three species (Setosa, Versicolor, Virginica) based on their measurements. The model achieves **93.33% accuracy** using Support Vector Machine (SVM).

## 📊 Dataset Information

- **Total Samples**: 150 iris flowers
- **Features**: 4 (Sepal Length, Sepal Width, Petal Length, Petal Width)
- **Classes**: 3 (Setosa, Versicolor, Virginica)
- **Training/Testing Split**: 70-30
- **Source**: Scikit-learn built-in Iris dataset

## 🔬 Machine Learning Models Tested

| Model | Accuracy | Performance |
|-------|----------|-------------|
| Random Forest | 88.89% | Good ensemble approach |
| Logistic Regression | 91.11% | Linear classification |
| **SVM (RBF Kernel)** | **93.33%** | **Best Performance ✓** |

## 📈 Model Performance

### Confusion Matrix (SVM)

```
Predicted    Setosa  Versicolor  Virginica
Actual
Setosa          15          0          0
Versicolor       0         14          1
Virginica        0          2         13
```

- **Setosa Classification**: 100% (15/15 correct)
- **Versicolor Classification**: 93.3% (14/15 correct)
- **Virginica Classification**: 86.7% (13/15 correct)
- **Overall Test Accuracy**: 93.33%

## 🛠️ Technologies Used

- **Python 3** - Programming language
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Scikit-learn** - Machine learning library
- **Matplotlib & Seaborn** - Data visualization
- **Google Colab** - Development environment

## 📂 Project Structure

```
Iris-Flower-Classification/
├── Task_1_internship.ipynb  # Complete Jupyter Notebook
├── README.md                 # Project documentation
└── requirements.txt          # Python dependencies
```

## 📋 Project Workflow

1. **Data Loading** - Import iris dataset from scikit-learn
2. **Data Exploration** - Analyze distributions and correlations
3. **Visualization** - Create histograms for each feature by species
4. **Preprocessing** - Split data (70-30), apply StandardScaler normalization
5. **Model Training** - Train Random Forest, Logistic Regression, and SVM models
6. **Evaluation** - Compare accuracies using classification metrics
7. **Results** - SVM provides best classification performance

## 🚀 How to Use

### Option 1: Google Colab (Recommended)

1. Open the `.ipynb` file in Google Colab
2. Run all cells sequentially (Runtime → Run all)
3. View visualizations and model performance
4. Modify parameters to experiment with different settings

### Option 2: Local Environment

```bash
# Install dependencies
pip install -r requirements.txt

# Run in Jupyter Notebook
jupyter notebook Task_1_internship.ipynb
```

## 🔑 Key Insights

1. **Species Separation**: Setosa is easily distinguishable from Versicolor and Virginica
2. **Feature Importance**: Petal measurements are more discriminative than sepal measurements
3. **Model Selection**: SVM with RBF kernel outperforms other models for this multiclass problem
4. **Data Preprocessing**: Proper feature scaling significantly improves model performance
5. **Accuracy Trade-off**: 93.33% accuracy is excellent for real-world iris classification

## 💡 Learning Outcomes

✅ Implemented end-to-end ML pipeline
✅ Worked with scikit-learn for multiple algorithms
✅ Created professional data visualizations
✅ Practiced model evaluation and comparison
✅ Documented code and results clearly

## 📈 Potential Improvements

- Cross-validation for more robust accuracy estimates
- Hyperparameter tuning using GridSearchCV
- Ensemble methods combining multiple models
- Real-time flower classification API
- Deployment on web/mobile platform

## 📚 Code Highlights

### Model Training Example

```python
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Train SVM model
svm_model = SVC(kernel='rbf', random_state=42)
svm_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = svm_model.predict(X_test_scaled)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"SVM Accuracy: {accuracy:.4f}")
# Output: 0.9333
```

## 📄 License

This project is open-source and available for educational purposes.

## 👤 Author

Built as part of **AICTE OASIS INFOBYTE Data Science Internship - Task 1**

---

**GitHub Repository**: [Iris-Flower-Classification](https://github.com/Shaikjanibasha3450/Iris-Flower-Classification)

**Last Updated**: November 29, 2025
