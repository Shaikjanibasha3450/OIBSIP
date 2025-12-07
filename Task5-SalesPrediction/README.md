# Task 5: Sales Prediction Using Linear Regression

## Overview
This task implements a **Linear Regression model** to predict sales based on advertising expenditure across different media channels (TV, Radio, and Newspaper).

## Objective
Build a machine learning model that:
- Analyzes the relationship between advertising spending and sales
- Predicts future sales based on advertising budgets
- Identifies which advertising channels have the most impact on sales
- Evaluates model performance using regression metrics

## Dataset
- **Source**: Advertising Dataset (200 samples)
- **Features**:
  - TV advertising spend
  - Radio advertising spend
  - Newspaper advertising spend
- **Target**: Sales (in $1000s)

## Model Details

### Algorithm: Linear Regression
- **Type**: Supervised Learning - Regression
- **Technique**: Ordinary Least Squares (OLS)
- **Libraries**: scikit-learn

### Key Features:
1. **Data Loading & Preprocessing**
   - Load dataset from URL or create synthetic data
   - Handle missing values
   - Data normalization (if needed)

2. **Exploratory Data Analysis (EDA)**
   - Correlation analysis between features and target
   - Visualization of relationships using scatter plots
   - Correlation heatmap

3. **Model Training**
   - 80-20 train-test split
   - LinearRegression model fitting
   - Coefficient extraction and interpretation

4. **Model Evaluation**
   - R² Score (Coefficient of Determination)
   - RMSE (Root Mean Squared Error)
   - MAE (Mean Absolute Error)
   - Performance on training and testing sets

5. **Visualization**
   - Actual vs Predicted plots for training and testing
   - Residuals analysis plots
   - Feature importance visualization

## Results Summary

### Model Performance:
- **R² Score**: Varies based on dataset (typically 0.8-0.95)
  - Explains variance in sales predictions
- **RMSE**: Very low error margin
- **Training vs Testing**: Consistent performance (no overfitting)

### Feature Importance (Coefficients):
1. **Radio Advertising**: Highest impact on sales
2. **TV Advertising**: Significant positive impact
3. **Newspaper Advertising**: Minimal impact

### Key Findings:
- Radio advertising generates the best ROI
- TV advertising maintains steady sales growth
- Newspaper advertising contributes minimally to sales
- Model fits well with low residuals

## Practical Recommendations

Based on the model analysis:
1. **Allocate more budget to Radio advertising** - highest ROI
2. **Maintain TV advertising** - stable sales contribution
3. **Minimize or eliminate newspaper advertising** - minimal impact
4. **Use model for budget planning** - data-driven decision making

## Files Included

- `sales_prediction.py` - Complete implementation with all analysis
- `README.md` - This documentation file

## Code Structure

The implementation includes:
1. Data loading and exploration
2. EDA with visualizations
3. Data preprocessing and train-test split
4. Model creation and training
5. Performance evaluation
6. Prediction visualization
7. Residuals analysis
8. Sample predictions on new data
9. Conclusions and recommendations

## Technologies Used
- **Python 3.x**
- **pandas** - Data manipulation
- **NumPy** - Numerical computations
- **scikit-learn** - Machine learning
- **Matplotlib & Seaborn** - Visualization

## How to Use

1. Run the Python script: `python sales_prediction.py`
2. View the EDA plots and model evaluation metrics
3. Check the predictions on sample advertising budgets
4. Use the model coefficients for budget allocation decisions

## Model Reliability

✅ **Strengths:**
- Low prediction errors (MAE < 1k typically)
- Consistent training and testing performance
- Interpretable coefficients
- Good fit (high R² score)

⚠️ **Limitations:**
- Linear assumptions may not capture complex relationships
- Limited to the variables in the model
- Assumes historical patterns continue

## Learning Outcomes

Through this task, you will learn:
- How to build and train a linear regression model
- Regression evaluation metrics and interpretation
- Feature importance analysis
- Data visualization for ML
- Model interpretation and practical application

---

**Task Status**: ✅ Completed  
**AICTE OASIS INFOBYTE SIP Program**
