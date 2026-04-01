#!/usr/bin/env python
# coding: utf-8

# # "THE PRICE IS RIGHT" Capstone Project
#
# This week - build a model that predicts how much something costs from a description, based on a scrape of Amazon data
#
# # Order of play
#
# DAY 1: Data Curation
# DAY 2: Data Pre-processing
# DAY 3: Evaluation, Baselines, Traditional ML
# DAY 4: Deep Learning and LLMs
# DAY 5: Fine-tuning a Frontier Model
#
# ## DAY 3: Evaluation, Baselines, Traditional ML
#
# Today we'll write some simple models to predict the price of a product
#
# We'll use an approach to evaluate the performance of the model
#
# And we'll test some Baseline Models using Traditional machine learning

# %%


import random

import numpy as np
import pandas as pd
import xgboost as xgb
from pricer.evaluator import evaluate
from pricer.items import Item
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# %%


LITE_MODE = False


# %%


username = "ed-donner"
dataset = f"{username}/items_lite" if LITE_MODE else f"{username}/items_full"

train, val, test = Item.from_hub(dataset)

print(
    f"Loaded {len(train):,} training items, {len(val):,} validation items, {len(test):,} test items"
)


# %%


def random_pricer(item):
    return random.randrange(1, 1000)


# %%


random.seed(42)
evaluate(random_pricer, test)


# %%


# That was fun!
# We can do better - here's another rather trivial model

training_prices = [item.price for item in train]
training_average = sum(training_prices) / len(training_prices)
print(training_average)


def constant_pricer(item):
    return training_average


# %%


evaluate(constant_pricer, test)


# %%


def get_features(item):
    return {
        "weight": item.weight,
        "weight_unknown": 1 if item.weight == 0 else 0,
        "text_length": len(item.summary),
    }


# %%


def list_to_dataframe(items):
    features = [get_features(item) for item in items]
    df = pd.DataFrame(features)
    df["price"] = [item.price for item in items]
    return df


train_df = list_to_dataframe(train)
test_df = list_to_dataframe(test)


# %%


# Traditional Linear Regression!

np.random.seed(42)

# Separate features and target
feature_columns = ["weight", "weight_unknown", "text_length"]

X_train = train_df[feature_columns]
y_train = train_df["price"]
X_test = test_df[feature_columns]
y_test = test_df["price"]

# Train a Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

for feature, coef in zip(feature_columns, model.coef_):
    print(f"{feature}: {coef}")
print(f"Intercept: {model.intercept_}")

# Predict the test set and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared Score: {r2}")


# %%


def linear_regression_pricer(item):
    features = get_features(item)
    features_df = pd.DataFrame([features])
    return model.predict(features_df)[0]


# %%


evaluate(linear_regression_pricer, test)


# %%


prices = np.array([float(item.price) for item in train])
documents = [item.summary for item in train]


# %%


np.random.seed(42)
vectorizer = CountVectorizer(max_features=2000, stop_words="english")
X = vectorizer.fit_transform(documents)


# %%


# Here are the 1,000 most common words that it picked, not including "stop words":

selected_words = vectorizer.get_feature_names_out()
print(f"Number of selected words: {len(selected_words)}")
print("Selected words:", selected_words[1000:1020])


# %%


regressor = LinearRegression()
regressor.fit(X, prices)


# %%


def natural_language_linear_regression_pricer(item):
    x = vectorizer.transform([item.summary])
    return max(regressor.predict(x)[0], 0)


# %%


evaluate(natural_language_linear_regression_pricer, test)


# %%


subset = 15_000
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=4)
rf_model.fit(X[:subset], prices[:subset])


# ## Random Forest model
#
# The Random Forest is a type of "**ensemble**" algorithm, meaning that it combines many smaller algorithms to make better predictions.
#
# It uses a very simple kind of machine learning algorithm called a **decision tree**. A decision tree makes predictions by examining the values of features in the input. Like a flow chart with IF statements. Decision trees are very quick and simple, but they tend to overfit.
#
# In our case, the "features" are the elements of the Vector - in other words, it's the number of times that a particular word appears in the product description.
#
# So you can think of it something like this:
#
# **Decision Tree**
# \- IF the word "TV" appears more than 3 times THEN
# -- IF the word "LED" appears more than 2 times THEN
# --- IF the word "HD" appears at least once THEN
# ---- Price = $500
#
#
# With Random Forest, multiple decision trees are created. Each one is trained with a different random subset of the data, and a different random subset of the features. You can see above that we specify 100 trees, which is the default.
#
# Then the Random Forest model simply takes the average of all its trees to product the final result.

# %%


def random_forest(item):
    x = vectorizer.transform([item.summary])
    return max(0, rf_model.predict(x)[0])


# %%


evaluate(random_forest, test)


# %%


# This is how to save the model if you want to, particularly if you run this on a larger dataset

# import joblib
# joblib.dump(rf_model, "random_forest.joblib")


# ## Introducing XGBoost
#
# Like Random Forest, XGBoost is also an ensemble model that combines multiple decision trees.
#
# But unlike Random Forest, XGBoost builds one tree after another, with each next tree correcting for errors in the prior trees, using 'gradient descent'.
#
# It's much faster than Random Forest, so we can run it for the full dataset, and it's typically better at generalizing.
#
# **If this import doesn't work, please skip this! It's not required. On a Mac, you might need to do `brew install libomp` in the terminal.**

# %%


# %%


np.random.seed(42)

xgb_model = xgb.XGBRegressor(
    n_estimators=1000, random_state=42, n_jobs=4, learning_rate=0.1
)
xgb_model.fit(X, prices)


# %%


def xg_boost(item):
    x = vectorizer.transform([item.summary])
    return max(0, xgb_model.predict(x)[0])


# %%


evaluate(xg_boost, test)


# <table style="margin: 0; text-align: left;">
#     <tr>
#         <td style="width: 150px; height: 150px; vertical-align: middle;">
#             <img src="../assets/business.jpg" width="150" height="150" style="display: block;" />
#         </td>
#         <td>
#             <h2 style="color:#181;">Business applications</h2>
#             <span style="color:#181;">Traditional ML isn't just useful for learning the history; it's still heavily used in industry today, particularly for tasks where there are clearly identifiable features. It's worth spending time exploring the algorithms and experimenting here. See if you can beat my numbers with Traditional ML! I ran the Random Forest for the entire 800,000 training dataset. It took about 15 hours to run, and it ended up getting a low error of $56.40. Traditional ML can do well - try it for yourself.</span>
#         </td>
#     </tr>
# </table>

#
