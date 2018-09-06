import pandas as pd
import numpy as np

import config

from sklearn.model_selection import cross_val_predict, GroupKFold
from sklearn.pipeline import FeatureUnion
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack


def post_process_train(preds, train, cvlist):
    preds = preds.copy()
    for tr_index, val_index in cvlist:
        tr = train.iloc[tr_index]
        vl = train.iloc[val_index]
        y_preds_val = preds[val_index]
        domain_dict = tr.groupby("Domain")["Tag"].apply(lambda x: x.unique().tolist()).to_dict()
        val_domains = list(vl.Domain.unique())
        val_corr_domains = set(val_domains) & set(list(domain_dict.keys()))
        for domain in val_corr_domains:
            if domain in domain_dict.keys():
                dm_idx = np.where(vl.Domain.values == domain)[0]
                for tag in domain_dict[domain]:
                    col_idx = config.TAG_DICT[tag]
                    y_preds_val[dm_idx, col_idx] = 0
            else:
                continue
        preds[val_index] = y_preds_val
    return preds


def post_process_test(test_preds, train, test):
    test_preds = test_preds.copy()
    domain_dict = train.groupby("Domain")["Tag"].apply(lambda x: x.unique().tolist()).to_dict()
    test_domains = list(test.Domain.unique())
    test_corr_domains = set(test_domains) & set(list(domain_dict.keys()))
    for domain in test_corr_domains:
        if domain in domain_dict.keys():
            dm_idx = np.where(test.Domain.values == domain)[0]
            for tag in domain_dict[domain]:
                col_idx = config.TAG_DICT[tag]
                test_preds[dm_idx, col_idx] = 0
        else:
            continue
    return test_preds


def tokenize_url(df):
    df["Url"] = df["Url"].str.replace("\/", " ")
    return df


if __name__ == "__main__":
    train = pd.read_csv(config.CLEAN_TRAIN_DATA)
    test = pd.read_csv(config.CLEAN_TEST_DATA)

    # Get numerical target
    train['target'] = train.Tag.map(config.TAG_DICT)
    y = train["target"].values

    # Replicate train/test split strategy for cross validation
    train["target_str"] = train["Domain"].astype(str) + train["Tag"].astype(str)
    train["target_str"] = train["target_str"].astype("category")
    cvlist = list(GroupKFold(5).split(train, groups=train["target_str"]))

    # Word and character TFIDF on URLs
    vec1 = TfidfVectorizer(analyzer='char', ngram_range=(1, 5), min_df=500, sublinear_tf=True)
    vec2 = TfidfVectorizer(analyzer='word', ngram_range=(1, 1), min_df=400, sublinear_tf=True)
    vec = FeatureUnion([("char", vec1), ("word", vec2)])

    train = tokenize_url(train)
    test = tokenize_url(test)
    all_url = pd.concat([train["Url"], test["Url"]])

    vec.fit(all_url)
    X_url_train = vec.transform(train["Url"])
    X_url_test = vec.transform(test["Url"])

    # Predictions from webpage text only model
    train_text = pd.read_csv(config.TRAIN_TEXT_MODEL_PATH)
    test_text = pd.read_csv(config.TEST_TEXT_MODEL_PATH)

    X_text_train = train_text[config.LABEL_COLS]
    X_text_test = test_text[config.LABEL_COLS]

    X_train = hstack((X_url_train, X_text_train)).tocsr()
    X_test = hstack((X_url_test, X_text_test)).tocsr()
    print("Shape of train and test after concatenating features are ",
          X_train.shape, X_test.shape)

    # Logistic regression
    model = LogisticRegression(C=0.1, solver="liblinear", class_weight="balanced",
                               max_iter=300, dual=True, random_state=123, verbose=1)
    y_preds = cross_val_predict(model, X_train, y, cv=cvlist, method='predict_proba',
                                n_jobs=-1)
    model.fit(X_train, y)
    y_test_preds = model.predict_proba(X_test)

    label_preds = np.argmax(y_preds, axis=1)
    score = f1_score(y, label_preds, average="weighted")
    print("Validation F1 score", score)

    # Post process predictions to use data split property
    y_preds_corr = post_process_train(y_preds, train, cvlist)
    y_test_preds_corr = post_process_test(y_test_preds, train, test)
    label_preds_corr = np.argmax(y_preds_corr, axis=1)
    score_pp = f1_score(y, label_preds_corr, average="weighted")
    print("Validation score after post processing ", score_pp)

    # Make submission
    inv_tag_dict = {v: k for k, v in config.TAG_DICT.items()}
    sub = test[["Webpage_id"]]
    sub["Tag"] = np.argmax(y_test_preds_corr, axis=1)
    sub["Tag"] = sub["Tag"].map(inv_tag_dict)
    sub.to_csv(config.SUBMISSION_PATH, index=False)


