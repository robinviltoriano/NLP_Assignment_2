import re

def clean_text(text):
    # Remove non-ASCII characters
    text = ''.join([char for char in text if ord(char) < 128])

    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)

    # Remove question mark problems
    text = re.sub(r'(\s\?)',' ',text)
    text = re.sub(r"\b\?\b", "\'", text)
    text = re.sub(r"(,\?)",",", text)
    text = re.sub(r"\?+", "?", text)
    text = text.strip()

    return text