# Foam_level_classification
Classification of Foaming Levels in a Carbonated Beverage Production Line

The foaming level in a carbonated beverage bottling line is a crucial factor that can impact the final product's quality and customer satisfaction. Excessive foaming can lead to significant economic costs, including production delays, loss of various raw materials, and a reduction in the lifespan of packaging, labels, and more. In this project, a dataset will be used to predict the foaming level based on various variables, such as the beverage temperature, carbonator tank temperature, dissolved oxygen level in the beverage, and other related factors. The goal is to develop an accurate predictive model that can help optimize the bottling process and minimize foaming levels.

Conclusion

Economic implications

The models developed for prediction and classification in this project have significant economic implications, especially in terms of production optimization and cost reduction. Here are some key points highlighted:
Production Optimization: Classification and regression models can accurately predict filling speed and foaming level. This allows production parameters to be adjusted in real time, ensuring more efficient operation and minimizing waste.
Cost Reduction: The ability to predict and adjust production can reduce operating costs. For example, better control of the foaming level can prevent product loss, which directly affects production cost and net profit.
Increased Net Benefit: The analysis shows that the Decision Tree model provides the highest net benefit (111.39), followed by KNN (108.4), SVM (105.4) and Naive Bayes (93.4). This suggests that an implementation based on the Decision Tree model could be more economically advantageous for the company.

Lessons learned

Importance of data quality: The accuracy of classification models depends largely on the quality and quantity of data available. Accurate and well-labeled data allows you to train more efficient and reliable models.
Model choice: Different models have different advantages and disadvantages. Although the Decision Tree initially showed high performance, tuning parameters and preventing overtraining made its performance and net benefit even higher.
Parameter Optimization: Using GridSearchCV and other hyperparameter optimization techniques is crucial to find the optimal configuration for each model, thus improving its performance and applicability.
Influence of key variables: We identified that beverage temperature, CO2 volume and dissolved air are critical variables to control foaming. Correct management of these variables can significantly improve the efficiency of the bottling process.

Areas of improvement

Continuous data improvement: Maintain and improve data quality by implementing more accurate sensors and constantly recording process variables.
Quality control automation: Implement automated systems to monitor and adjust critical process variables, such as temperature and CO2 volume, in real time to keep them within optimal ranges.
Staff training: Provide ongoing training to staff so they understand the importance of key variables and how to manage them appropriately.
Real-time analysis: Develop real-time analysis systems that allow an immediate response to any deviations in critical variables, thus minimizing the risk of excessive foaming and other quality problems.
This project has demonstrated the potential of sorting techniques to optimize the soft drink bottling process, with important economic and operational implications. Implementing appropriate classification models and continuous improvement of processes and data can lead to greater efficiency, cost reduction, and improvement in the quality of the final product.

Technologies Used in the Project

This project leverages advanced Python libraries and tools for data analysis, visualization, and modeling. The main technologies used are:

1. Data Manipulation and Analysis:
NumPy: For numerical operations and handling multidimensional arrays.
Pandas: For data manipulation and analysis using DataFrame structures.

2. Data Visualization:
Matplotlib: For creating static plots, color maps, and custom annotations.
Plotly: For building interactive and customizable graphs, including subplots.

3. Data Preprocessing:
StandardScaler (Scikit-learn): For normalizing features by removing the mean and scaling to unit variance.

4. Data Modeling:
Classification Models:
Decision Trees (DecisionTreeClassifier)
K-Nearest Neighbors (KNeighborsClassifier)
Support Vector Machines (SVC)
Gaussian Naive Bayes (GaussianNB)

5. Model Evaluation and Validation:
Scikit-learn:
Metrics: accuracy, confusion matrix, mean squared error (MSE), coefficient of determination (R²), mean absolute error (MAE).
Validation techniques: cross-validation, hyperparameter tuning (GridSearchCV, StratifiedKFold, etc.).

6. Metric and Model Visualization:
plot_tree (Scikit-learn): For visualizing decision tree structures.
ConfusionMatrixDisplay: For graphical representation of the confusion matrix.

Classification Models Used
Decision Trees (DecisionTreeClassifier)
K-Nearest Neighbors (KNeighborsClassifier)
Support Vector Machines (SVC)
Gaussian Naive Bayes (GaussianNB)
