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
