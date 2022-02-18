# ML Model - Based on Decision Tree
# all necessary imports
import warnings
from decimal import Decimal
from statistics import mean

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier

# ignore warnings generated due to usage of old version of tensorflow
warnings.simplefilter("ignore")


def predictor(symptoms):
    # Load Dataset scraped from NHP (https://www.nhp.gov.in/disease-a-z) & Wikipedia
    # Scrapping and creation of dataset csv is done in a separate program
    # df_comb = pd.read_csv(
    #     "/Users/prachiragrawal/Web Development/HealthApp/mysite/myapi/disease_symptom_dataset_combinations.csv")
    # df_norm = pd.read_csv(
    #     "/Users/prachiragrawal/Web Development/HealthApp/mysite/myapi/disease_symptom_dataset_normal.csv")

    df_comb = pd.read_csv(
        "./myapi/disease_symptom_dataset_combinations.csv")
    df_norm = pd.read_csv(
        "./myapi/disease_symptom_dataset_normal.csv")

    # Real Time Predictions
    X = df_comb.iloc[:, 1:]
    Y = df_comb.iloc[:, 0:1]

    # List of symptoms
    dataset_symptoms = list(X.columns)

    # User symptoms
    final_symp = symptoms

    sample_x = [0 for x in range(0, len(dataset_symptoms))]

    for val in final_symp:
        sample_x[dataset_symptoms.index(val)] = 1

    # Disease Prediction

    dt = DecisionTreeClassifier()
    dt = dt.fit(X, Y)
    scores_dt = cross_val_score(dt, X, Y, cv=5)
    prediction = dt.predict_proba([sample_x])

    k = 10
    diseases = list(set(Y['label_dis']))
    diseases.sort()
    topk = prediction[0].argsort()[-k:][::-1]

    print(f"\nTop {k} diseases predicted based on symptoms")
    topk_dict = {}
    # Show top 10 highly probable disease to the user.
    for idx, t in enumerate(topk):
        match_sym = set()
        row = df_norm.loc[df_norm['label_dis'] == diseases[t]].values.tolist()
        row[0].pop(0)

        for idx, val in enumerate(row[0]):
            if val != 0:
                match_sym.add(dataset_symptoms[idx])
        prob = (len(match_sym.intersection(set(final_symp)))+1) / \
            (len(set(final_symp))+1)
        prob *= mean(scores_dt)
        topk_dict[t] = prob
    j = 0
    topk_index_mapping = {}
    topk_sorted = dict(
        sorted(topk_dict.items(), key=lambda kv: kv[1], reverse=True))

    data = {'diseases': []}
    for key in topk_sorted:
        prob = topk_sorted[key]*100
        print(str(j) + " Disease name:",
              diseases[key], "\tProbability:", str(round(prob, 2))+"%")

        data['diseases'].append([diseases[key], str(round(prob, 2))])
        topk_index_mapping[j] = key
        j += 1

    return data
