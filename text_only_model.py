import pandas as pd
import numpy as np

import config

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

from sklearn.model_selection import cross_val_predict, GroupKFold

from sklearn.metrics import f1_score


def main():
    train = pd.read_csv(config.CLEAN_TRAIN_DATA)
    test = pd.read_csv(config.CLEAN_TEST_DATA)
    print("Shape of train and test", train.shape, test.shape)
    print("Columns in train are", list(train.columns))

    # Map labels to integers
    train['target'] = train.Tag.map(config.TAG_DICT)

    # Replicate train/test split strategy for cross validation
    train["target_str"] = train["Domain"].astype(str) + train["Tag"].astype(str)
    cvlist = list(GroupKFold(5).split(train, groups=train["target_str"].astype('category')))

    # TFIDF for words and characters
    vec1 = TfidfVectorizer(ngram_range=(1, 4), analyzer="char",
                           min_df=1000, max_df=1.0, strip_accents='unicode', use_idf=1,
                           smooth_idf=1, sublinear_tf=1, max_features=20000)
    vec2 = TfidfVectorizer(ngram_range=(1, 1), analyzer="word",
                           min_df=1000, max_df=1.0, strip_accents='unicode', use_idf=1,
                           smooth_idf=1, sublinear_tf=1, max_features=20000)

    vec = FeatureUnion([("char", vec1), ("word", vec2)])
    vec.fit(pd.concat([train["text"], test["text"]]))
    X = vec.transform(train["text"])
    X_test = vec.transform(test["text"])
    print("Shape of train and test after TFIDF transformation".format(X.shape, X_test.shape))

    # Model
    y = train["target"].values
    model = LogisticRegression(C=1, class_weight='balanced', dual=True, solver='liblinear')
    y_preds = cross_val_predict(model, X, y, cv=cvlist, method='predict_proba')
    model.fit(X, y)
    y_test_preds = model.predict_proba(X_test)
    print("Overall f1 score for text only model", f1_score(y, np.argmax(y_preds, axis=1), average="weighted"))

    # Save out-of-fold and test predictions
    oof_preds: pd.DataFrame = train[['Webpage_id']]
    for i, col in enumerate(config.LABEL_COLS):
        oof_preds.loc[:, col] = y_preds[:, i]

    test_preds: pd.DataFrame = test[['Webpage_id']]
    for i, col in enumerate(config.LABEL_COLS):
        test_preds.loc[:, col] = y_test_preds[:, i]

    oof_preds.to_csv(config.TRAIN_TEXT_MODEL_PATH, index=False)
    test_preds.to_csv(config.TEST_TEXT_MODEL_PATH, index=False)


if __name__ == "__main__":
    main()

