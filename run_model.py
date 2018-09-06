import pandas as pd
import numpy as np

import config

from sklearn.model_selection import cross_val_predict, GroupKFold
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack


def post_process_train(y_preds, train, cvlist):
    y_preds = y_preds.copy()
    for tr_index, val_index in cvlist:
        tr = train.iloc[tr_index]
        vl = train.iloc[val_index]
        y_preds_val = y_preds[val_index]
        domain_dict = tr.groupby("Domain")["Tag"].apply(lambda x: x.unique().tolist()).to_dict()

        val_domains = list(vl.Domain.unique())
        val_corr_domains = set(val_domains) & set(list(domain_dict.keys()))
        print(len(val_corr_domains))
        for domain in val_corr_domains:
            if domain in domain_dict.keys():
                # print("here")
                dm_idx = np.where(vl.Domain.values == domain)[0]
                # print(dm_idx)
                for tag in domain_dict[domain]:
                    col_idx = config.TAG_DICT[tag]
                    # print(tag, col_idx)
                    y_preds_val[dm_idx, col_idx] = 0
            else:
                continue
        y_preds[val_index] = y_preds_val
    return y_preds


def post_process_test(y_test_preds, train, test):
    y_test_preds = y_test_preds.copy()

    domain_dict = train.groupby("Domain")["Tag"].apply(lambda x: x.unique().tolist()).to_dict()

    test_domains = list(test.Domain.unique())
    test_corr_domains = set(test_domains) & set(list(domain_dict.keys()))
    print(len(test_corr_domains))
    for domain in test_corr_domains:
        if domain in domain_dict.keys():
            # print("here")
            dm_idx = np.where(test.Domain.values == domain)[0]
            # print(dm_idx)
            for tag in domain_dict[domain]:
                col_idx = config.TAG_DICT[tag]
                # print(tag, col_idx)
                y_test_preds[dm_idx, col_idx] = 0
        else:
            continue
    return y_test_preds


if __name__ == "__main__":
    train = pd.read_csv(config.CLEAN_TRAIN_DATA)
    test = pd.read_csv(config.CLEAN_TEST_DATA)

    # Word and character TFIDF on URLs
    vec1 = TfidfVectorizer(analyzer='char', ngram_range=(1, 5), min_df=500, sublinear_tf=True)
    vec2 = TfidfVectorizer(analyzer='word', ngram_range=(1, 1), min_df=400, sublinear_tf=True)

    vec1.fit(pd.concat([train["Url"].str.replace("\/", " "), test["Url"].str.replace("\/", " ")]))
    vec2.fit(pd.concat([train["Url"].str.replace("\/", " "), test["Url"].str.replace("\/", " ")]))

    X_url1 = vec1.transform(train.Url.str.replace("\/", " "))
    X_url2 = vec2.transform(train.Url.str.replace("\/", " "))

    X_test_url1 = vec1.transform(test.Url.str.replace("\/", " "))
    X_test_url2 = vec2.transform(test.Url.str.replace("\/", " "))

    # Predictions from webpafe text only model
    oof_text2 = pd.read_csv(config.TRAIN_TEXT_MODEL_PATH)
    test_text2 = pd.read_csv(config.TEST_TEXT_MODEL_PATH)

    oof_cols = ['others', 'news', 'publication', 'profile', 'conferences',
                'forum', 'clinicalTrials', 'thesis', 'guidelines']
    X = hstack((X_url1, X_url2, oof_text2[oof_cols])).tocsr()
    X_test = hstack((X_test_url1, X_test_url2, test_text2[oof_cols])).tocsr()
    print("Shape of train and test after concatenating features ", X.shape, X_test.shape)

    # Logistic regression
    lr = LogisticRegression(C=0.1, solver="liblinear", class_weight="balanced", max_iter=300, dual=True,
                            random_state=123, verbose=1)
    cvlist = list(GroupKFold(5).split(train, groups=train["target_str"].astype('category')))
    preds = cross_val_predict(lr, X, y, cv=cvlist, n_jobs=-1, verbose=10, method='predict_proba')
    score = f1_score(y, np.argmax(preds, axis=1), average="weighted")
    print("Validation F1 score", score)

    # Post process predictions to use data split property
    y_preds_corr = post_process_train(preds, train, cvlist)
    score_pp = f1_score(y, np.argmax(y_preds_corr, axis=1), average="weighted")
    print("Validation score after post processing ", score_pp)
    y_test_preds_corr = post_process_test(y_test_preds, train, test)

    # Make submission
    inv_tag_dict = {v: k for k, v in config.TAG_DICT.items()}
    sub = test[["Webpage_id"]]
    sub["Tag"] = np.argmax(y_test_preds_corr, axis=1)
    sub["Tag"] = sub["Tag"].map(inv_tag_dict)
    sub.to_csv(config.SUBMISSION_PATH, index=False)


