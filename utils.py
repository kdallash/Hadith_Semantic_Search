import re

def remove_tashkeel(text):
    tashkeel_pattern = re.compile(
        r'[\u0617-\u061A\u064B-\u0652]'
    )
    return re.sub(tashkeel_pattern, '', text)

def normalize_arabic(text):
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ؤ", "و", text)
    text = re.sub("ئ", "ي", text)
    text = re.sub("ة", "ه", text)
    return text

def preprocess_arabic(text):
    text = remove_tashkeel(text)
    text = normalize_arabic(text)
    return text

def bm25_tokenize(text):
    return preprocess_arabic(text).split()
def preprocess_query(query):
    query = preprocess_arabic(query)
    return bm25_tokenize(query)
