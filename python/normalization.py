from tqdm import tqdm
import re,string,json
import pkg_resources
from symspellpy.symspellpy import SymSpell,Verbosity
import spacy
import contractions

def simplify_punctuation_and_whitespace(sentence_list):
    norm_sents = []
    print("Normalizing whitespaces and punctuation")
    for sentence in tqdm(sentence_list):
        sent = _simplify_punctuation(sentence)
        sent = _normalize_whitespace(sent)
        norm_sents.append(sent)
    return norm_sents

def _simplify_punctuation(text):
    corrected = str(text)
    corrected = re.sub(r'([!?,;])\1+', r'\1', corrected)
    corrected = re.sub('"','',corrected)
    corrected = re.sub(r'\.{2,}', r'...', corrected)
    return corrected

def _normalize_whitespace(text):
    corrected = str(text)
    corrected = re.sub('r\t','',corrected)
    return corrected


def normalize_contractions(sentence_list):
    contraction_list = json.loads(open('english_contractions.json', 'r').read())
    norm_sents = []
    print("Normalizing contractions")
    for sentence in tqdm(sentence_list):
        norm_sents.append(sentence)
    return norm_sents

def _reduce_exaggerations(text):
#    Przykład:
#    "Heeeeeejka" -> "Hejka"
#    "jaaaaaaak" -> "jak"
    correction = str(text)
    #TODO work on complexity reduction.
    return re.sub(r'([\w])\1+', r'\1', correction)


def normalization_pipeline(sentences):
    print("##############################")
    print("Starting Normalization Process")
    sentences = simplify_punctuation_and_whitespace(sentences)
    sentences = normalize_contractions(sentences)
    print("Normalization Process Finished")
    print("##############################")
    sentences = "".join(sentences)
    normalized = open("normalized_text.txt","w",encoding="latin-1")
    normalized.write(sentences)
    print(sentences)
    return sentences

PATH_TEXT = "Wiedzmin - Copy.txt"
with open(PATH_TEXT, 'r', encoding="latin-1") as f:
    lines = f.readlines()

    chars = " ".join(lines)

#call func
normalization_pipeline(chars)