import re
def clean_text(text):
    text = re.sub(r'(\s\?)',' ',text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r"\b\?\b", "\'", text)
    text = re.sub(r"(,\?)",",", text)
    text = re.sub(r"\?+", "?", text)
    text = text.strip()
    return text