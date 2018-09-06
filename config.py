# Please change this appropriately
TRAIN_DATA = "../input/train.csv"
TEST_DATA = "../input/test.csv"
HTML_DATA = "../input/html_data.csv"

CLEAN_TRAIN_DATA = "../input2/train_v2.csv"
CLEAN_TEST_DATA = "../input2/test_v2.csv"

TRAIN_TEXT_MODEL_PATH = "../input2/train_text_preds.csv"
TEST_TEXT_MODEL_PATH = "../input2/test_text_preds.csv"

SUBMISSION_PATH = "../input2/submission.csv"

# Preferably keep them as is
LABEL_COLS = ["others", "news", "publication", "profile",
              "conferences", "forum", "clinicalTrials",
              "thesis", "guidelines"]

TAG_DICT = {"others":0, "news": 1, "publication":2, "profile": 3,
            "conferences": 4, "forum": 5, "clinicalTrials": 6,
            "thesis": 7, "guidelines": 8}

