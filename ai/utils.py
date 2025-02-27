from nltk import pos_tag, download
from nltk.tokenize import word_tokenize

if not download('punkt_tab'):
    print("Download punkt_tab not done")

def test():
    text = "NLP is a fascinating field of study."
    return tokenize(text)

def tokenize(text):
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)
    print(pos_tags)
    return tokens