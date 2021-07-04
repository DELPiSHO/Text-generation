from tqdm import tqdm
import string
def delete_punctuation(book):
    normalized_text = []
    for sentence in tqdm(book):
        fixed = punctuation(sentence)
        normalized_text.append(fixed)
    return normalized_text
def punctuation(text):
    fixed = str(text)
    fixed = fixed.strip('“')
    fixed = fixed.lstrip('“')
    fixed = fixed.rstrip('”')
    fixed = fixed.strip('”')
    fixed = fixed.strip('_')
    fixed = fixed.strip('‘')
    fixed = fixed.strip('’')
    fixed = fixed.strip('\t')
    return fixed
def normalization(book):
    book = delete_punctuation(book)
    normalized = open("Wiedzmin-normalized.txt", "w", encoding="latin-1")
    book = "".join(book)
    normalized.write(book)
    normalized.close()
with open('Wiedzmin.txt', 'r', encoding="latin-1") as pre_normalized:
    lines = pre_normalized.readlines()
    chars = " ".join(lines)
normalization(chars)