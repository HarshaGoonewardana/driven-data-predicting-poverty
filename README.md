# Predicting Poverty 
This is my submission for the 'Predicting Poverty' data science challenge hosted by the World Bank on DrivenData (https://www.drivendata.org/competitions/50/worldbank-poverty-prediction/). The goal of the contest is to predict if a household is poor or not. The input data came from household-level and individual-level World Bank surveys for three countries. The column names and categorical column values were anonymized to keep the countries anonymous. The evaluation metric used for the leaderboard was mean log loss. I finished the competition with a mean log loss score of 0.1625 (the top score was 0.1480) and ranked 126 out of 2500+ contestants.

## Summary of my approach
I used a regularized logistic regression model to classify if the household is poor and predict the probability. I chose the parameters of the regression model and lasso vs ridge regression using k-fold cross validation (with k = 5). I decided to start with logistic regression since it is easy to implement and good for predicting probabilities. I compared the performance against the random forest model (from the h20 library) and found that regularized logistic regression performed better on my features for this dataset.

Some of the other techniques that helped improve the performance of the classifier are removing columns with high missing data, imputing missing data and dropping features that have low variation. Please refer to the IPython notebook for the code and more details on the approach. 




