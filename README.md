Alphabet Soup Neural Network Model Analysis

Overview of the Analysis

This project aimed to create a deep learning model to help Alphabet Soup predict whether nonprofit funding applicants would be successful. We used data about past applicants, cleaned it, trained a neural network, and tested how well it could make predictions. The goal was to reach an accuracy of 75% or higher.

Results
Data Preprocessing
Target Variable (What We’re Predicting):

The column IS_SUCCESSFUL was our target. It indicates if an applicant was successful (1) or not (0).
Feature Variables (What the Model Uses to Predict):

All other columns, except for EIN and NAME, were used as features.
Removed Variables:

EIN and NAME were removed because they don’t help predict success. EIN is just an ID; NAME is textual data that doesn’t add useful information.
Compiling, Training, and Evaluating the Model
Model Architecture:

Neurons:

The first hidden layer had 80 neurons.
The second hidden layer had 30 neurons.
The output layer had one neuron (to predict success or failure).

Layers:

Input layer: Took in the features.

Hidden layers: Two layers with ReLU activation to handle non-linear patterns.
Output layer: Used sigmoid activation to predict probabilities (0 to 1).

Activation Functions:

ReLU was chosen for the hidden layers because it works well with non-linear data.

Sigmoid was used in the output layer for binary classification (yes or no).

Model Performance Comparison:

Google Colab Results:
Test Loss: 0.6909
Test Accuracy: 53.41%

Local Neural Network Results (deep_learning_challenge):
Test Loss: 0.7314
Test Accuracy: 64.10%

Neither environment achieved the target accuracy of 75%. However, the local model performed better, with an accuracy of 64.10%, compared to 53.41% in Google Colab.
Optimization Steps Taken:

Data Binning:

Combined rare categories in APPLICATION_TYPE and CLASSIFICATION into a group called “Other” to reduce noise in the data.

Feature Scaling:

Used StandardScaler to ensure all features were on the same scale so no single feature dominated the learning process.

Architectural Adjustments:

I tried adding more neurons to the layers.
Tested additional hidden layers to improve learning.
Adjusted training parameters, like the number of epochs and batch sizes.

Summary

I found That the neural network model could not achieve the target accuracy of 75% in either Google Colab or the local environment. The regional model performed slightly better, reaching 64.10% accuracy, compared to 53.41% in Google Colab.

Recommendations for Improvement

Try a Different Model:

Random forest classification could work better because it handles categorical data well and finds patterns in structured data well.
Gradient Boosting (e.g., XGBoost): Known for its high accuracy with tabular datasets and ability to capture complex relationships.
Feature Engineering:

Create new features by combining or transforming existing ones. For example, grouping similar classifications or adding indicators for key factors might improve the model's accuracy.
Hyperparameter Tuning:

Use tools like Keras Tuner or GridSearchCV to find the best settings for the neural network, like the optimal number of neurons, layers, learning rates, or batch sizes.
Increase Data Quality:

Consider collecting more data or removing noise from the dataset to improve model performance.
By exploring these strategies, Alphabet Soup can achieve better predictive results and make smarter funding decisions.

FURTHER TESTING:
Key Steps in the Process
1. Setting Up the Environment
•	Google Colab: A cloud-based environment configured dynamically with Spark and Java installations. Dependencies Ire fetched and installed at runtime, demonstrating the flexibility of Colab for rapid prototyping and experimentation.
•	Custom Container (Neural_Net): A pre-configured containerized environment using Docker and Conda. This setup ensured reproducibility, stability, and fine-grained control over the Spark runtime configuration. Local and cloud resources Ire integrated seamlessly, enabling predictable and scalable performance.
The containerized setup required upfront effort in building and configuring the environment. HoIver, this was offset by its consistency and the ability to handle larger datasets without the limitations of transient cloud environments.
________________________________________
2. Data Ingestion and Preparation
•	The dataset was loaded into a Spark DataFrame, ensuring schema inference for automated type detection. This step was critical for efficiently running Spark SQL queries later.
•	A temporary SQL table was created from the DataFrame, enabling us to use SQL queries for data analysis. This approach was chosen for its familiarity and readability, mainly when translating complex filtering and aggregation logic.
________________________________________
3. Executing Queries
Using the temporary table, I implemented several SQL queries to extract insights from the data:
•	Home prices Ire calculated for specific property features (e.g., bedrooms, bathrooms, square footage) over the years.
•	Advanced filters, such as combining multiple property attributes, Ire applied to ensure meaningful segmentations of the dataset.
These queries demonstrated PySpark's ability to handle complex operations efficiently. Each query was tested in both environments, showcasing the differences in execution speed and resource utilization.
________________________________________
4. Optimizing Performance
I leveraged caching and partitioning to optimize query execution:
•	Caching: Frequently accessed data was cached in memory, reducing runtime for repeated queries. This technique highlighted the differences in how each environment manages memory and resource allocation.
•	Partitioning: The dataset was partitioned by the date_built field, improving performance by enabling Spark to scan only relevant partitions. The partitioned data was stored in Parquet format, a columnar storage format that further boosted query performance.
________________________________________
5. Comparing Environments
The inclusion of a custom containerized setup added depth to the challenge, allowing us to compare its performance against Google Colab:
•	Colab: Suitable for small to medium datasets and rapid experimentation, but its transient nature and limited resource allocation pose challenges for more demanding tasks.
•	Custom Container: Designed for scalability and robustness, this environment excelled in handling large datasets and provided fine-tuned control over Spark configurations.
By analyzing the same dataset in both environments, I evaluated factors such as setup time, query execution speed, and resource management.
________________________________________
FURTHER TESTING
This project aimed to develop a machine learning model to predict the success of charitable donations with an accuracy exceeding 75%. I explored multiple approaches, including artificial neural networks, decision tree-based models, feature engineering, and data balancing techniques.
I first built a neural network model with multiple hidden layers and tuned hyperparameters such as the number of neurons, activation functions, and dropout rates. The initial neural network achieved an accuracy of 72.45%, which indicated that further optimization was required. I then tested different improvements, including adding more layers, using batch normalization, and adjusting the learning rate. HoIver, these adjustments led to decreased performance, with accuracy dropping to 68.57%, suggesting that the neural network was not the best approach for this dataset.
I then tested two decision tree-based models: Random Forest and XGBoost. The Random Forest model achieved an accuracy of 72.52%, while the XGBoost model outperformed it with an accuracy of 72.71%. This suggested that XGBoost was a more suitable model for this structured dataset. To further refine XGBoost, I applied hyperparameter tuning, adjusting parameters such as the learning rate, tree depth, and the number of trees. This fine-tuning improved accuracy slightly, bringing it to 72.80%.
Since class imbalance can impact model performance, I applied SMOTE (Synthetic Minority Oversampling Technique) to balance the dataset. After using SMOTE, XGBoost achieved an improved accuracy of 73.68%, indicating that balancing the dataset had a positive effect. To push accuracy further, I conducted another round of hyperparameter tuning, increasing the number of estimators, adjusting the learning rate, and increasing tree depth. This final tuning raised XGBoost's accuracy to 73.83%, making it the best-performing model in this study.
Finally, I tested an ensemble approach, combining XGBoost and Random Forest into a hybrid model. This method aimed to leverage both models' strengths, but the hybrid approach's final accuracy was 73.60%, which was slightly loIr than XGBoost alone. Given all the results, the fine-tuned XGBoost model with SMOTE and optimized hyperparameters was the most effective, achieving a final accuracy of 73.83%.
After finalizing the model, I saved it using XGBoost’s built-in save_model() function in JSON format for future use. While the project did not reach the 75% threshold, the structured approach to testing different machine learning models, tuning hyperparameters, and improving data balance provided valuable insights into the best techniques for this dataset.
REFERENCES
https://pandas.pydata.org/pandasdocs/stable/reference/api/pandas.DataFrame.nunique.html
https://pandas.pydata.org/pandasdocs/stable/reference/api/pandas.DataFrame.replace.html
https://pandas.pydata.org/pandasdocs/stable/reference/api/pandas.Series.value_counts.htmlhttps://pandas.pydata.org/pandasdocs/stable/reference/api/pandas.get_dummies.hml
https://keras.io/api/models/sequential/
https://keras.io/api/layers/core_layers/input/
https://keras.io/api/metrics/accuracy_metrics/#binaryaccuracyclassbinaryaccuracy_keras_metrics
https://numpy.org/doc/stable/reference/generated/numpy.asarray.html
https://pandas.pydata.org/pandasdocs/stable/reference/api/pandas.DataFrame.fillna.html
https://keras.io/api/models/model_training_apis/
https://keras.io/api/models/model_training_apis/#model_evaluate_function
https://scikitlearn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html
https://seaborn.pydata.org/generated/seaborn.heatmap.html
https://scikitlearn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html
https://keras.io/api/models/sequential/
https://keras.io/api/models/model_training_apis/#model_compile_function
ChatGPT

