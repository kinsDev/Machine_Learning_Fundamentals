# Report: Predict Bike Sharing Demand with AutoGluon Solution
#### KINSLEY KAIMENYI GITONGA

## Initial Training
### What did you realize when you tried to submit your predictions? What changes were needed to the output of the predictor to submit your results?
I realized that I had to drop the "casual" and "registered" columns that prevented submission. Dropping the rwo columns allowed generating predictions.

### What was the top ranked model that performed?
Respectively in the initial training and testing phase the top ranked model was the **WeightedEnsemble_L3** with a score of **1.34469** 

## Exploratory data analysis and feature creation
### What did the exploratory analysis find and how did you add additional features?
I added additional features such as:
1. Datetime parsing into year, month, day, hour, minute and second
2. categorized morning, lunch, and evening rush hour, 7-9am, 11am-1pm, and 5-6pm.
3. created a temperature_category column where I filled it with temperature description(hot/cold/mild) and added if-statements to specify what temperature value mateched to a specific temperature_category
4. created a windspeed column that was used to specify whether it was very windy or mild based on the if-statements that I used to assign the range of windspeed
5. created a humidity column that I used to specify whether it was humid or not humid based on the conditions that I had put in place on the humidity values

### How much better did your model preform after adding additional features and why do you think that is?
The model's performance notably improved after the addition of new features, resulting in a score improvement from 1.34469 to 0.64893. These added features provided the model with richer information, enabling it to capture more nuanced patterns in the data and thus achieve higher accuracy.

## Hyper parameter tuning
### How much better did your model preform after trying different hyper parameters?
After experimenting with different hyperparameters, the model's performance further improved, evidenced by a decrease in score from 0.64893 to 0.49212. This reduction suggests that optimizing hyperparameters allowed the model to better capture underlying data patterns, leading to more accurate predictions.

### If you were given more time with this dataset, where do you think you would spend more time?
1. Feature Engineering: I would explore additional ways to extract meaningful insights from the dataset's features, potentially uncovering new variables that could enhance model performance.
2. Hyperparameter Optimization: I would delve deeper into optimizing the model's hyperparameters to achieve even better performance, leveraging techniques such as Bayesian optimization or grid search.

### Create a table with the models you ran, the hyperparameters modified, and the kaggle score.
|model|hpo1|hpo2|hpo3|score|
|--|--|--|--|--|
|initial|?|?|?|?|
|add_features|?|?|?|?|
|hpo|?|?|?|?|

### Create a line plot showing the top model score for the three (or more) training runs during the project.

![model_train_score.png](img/model_train_score.png)

### Create a line plot showing the top kaggle score for the three (or more) prediction submissions during the project.
![model_test_score.png](img/model_test_score.png)

## Summary
This project involved predicting bike sharing demand using AutoGluon. Through exploratory data analysis, feature engineering, and hyperparameter tuning, I was able to significantly improve model performance. Further optimization and feature refinement could lead to even better results.