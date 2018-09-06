import pandas as pd
import config

from bs4 import BeautifulSoup
from bs4.element import Comment

from tqdm import tqdm
tqdm.pandas(tqdm)


def tag_visible(element):
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True


def text_from_html(body):
    soup = BeautifulSoup(body, 'html.parser')
    texts = soup.findAll(text=True)
    visible_texts = filter(tag_visible, texts)
    return u" ".join(t.strip() for t in visible_texts)


def main():
    train = pd.read_csv(config.TRAIN_DATA)
    test = pd.read_csv(config.TEST_DATA)

    html_data = pd.read_csv(config.HTML_DATA)
    html_data["text"] = html_data.Html.progress_apply(text_from_html)
    html_data["text2"] = html_data.text.progress_apply(lambda x: x.encode("utf-8", errors="ignore"))

    train["text"] = train.Webpage_id.map(html_data.set_index("Webpage_id")["text2"])
    test["text"] = test.Webpage_id.map(html_data.set_index("Webpage_id")["text2"])

    print(train.shape, test.shape)

    train.to_csv(config.CLEAN_TRAIN_DATA, index=False)
    test.to_csv(config.CLEAN_TEST_DATA, index=False)


if __name__ == "__main__":
    main()

