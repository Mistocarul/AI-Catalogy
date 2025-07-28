import sys
import os
import nltk
import langcodes
import asyncio
import re
import string
import random
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag
from nltk.corpus import wordnet
from collections import Counter
from langdetect import detect
from googletrans import Translator
import spacy
import yake
from generare import generate_sentence_with_word

nltk.download('punkt') # Tokenizare propoziții si cuvinte
nltk.download('wordnet') # Dicționar de sinonime
nltk.download('omw-1.4') # Open Multilingual Wordnet
nltk.download('averaged_perceptron_tagger') # Etichetare POS (part-of-speech)

def read_text():
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                return f.read()
        else:
            print('File not found')
            sys.exit(1)
    else:
        return input('Enter text: ')

def detect_language(text):
    lang = detect(text)
    lang_name = langcodes.Language.make(language=lang).display_name('en')
    return lang_name

async def translate_to_english(text):
    translator = Translator()
    translation = await asyncio.to_thread(translator.translate, text, src='ro', dest='en')
    return translation.text

def stylometric_analysis(text):
    text = re.sub(r'([.,!?])', r' \1 ', text) # Adăugare spații înainte și după semnele de punctuație
    text = re.sub(r'\s{2,}', ' ', text) # Eliminare spații multiple

    words = word_tokenize(text, language='english')
    words = [word for word in words if word not in string.punctuation]
    print(words)
    word_count = len(words)
    char_count = len(text)
    freq_dist = Counter(words)

    print("\nStylometric Analysis:")
    print(f"Word count: {word_count}")
    print(f"Character count: {char_count}")
    print("Word frequency:")
    for word, freq in freq_dist.most_common(10):
        print(f"{word}: {freq}")




def get_synonyms(word, pos):
    synonyms = set()
    for syn in wn.synsets(word, pos=pos):
        for lemma in syn.lemmas():
            if lemma.name() != word and lemma.name() not in string.punctuation:
                synonyms.add(lemma.name())
    return list(synonyms)[:5]

def get_hypernyms(word, pos):
    hypernyms = set()
    for syn in wn.synsets(word, pos=pos):
        for hyper in syn.hypernyms():
            for lemma in hyper.lemmas():
                if lemma.name() != word and lemma.name() not in string.punctuation:
                    hypernyms.add(lemma.name())
    return list(hypernyms)[:5]

def get_antonyms(word, pos):
    antonyms = set()
    for syn in wn.synsets(word, pos=pos):
        for lemma in syn.lemmas():
            if lemma.antonyms() and lemma.antonyms()[0].name() not in string.punctuation:
                antonyms.add("not " + lemma.antonyms()[0].name())
    return list(antonyms)[:5]

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def generate_alternative_text(text):
    words = word_tokenize(text)
    words = [word for word in words if word not in string.punctuation]

    pos_tags = pos_tag(words)
    priority_order = {'R': 2, 'J': 1, 'V': 3, 'N': 4} # Adjective, Adverb, Verb, Noun
    words_sorted = sorted(pos_tags, key=lambda x: priority_order.get(x[1][0],5))

    num_to_replace = max(1, int(0.2 * len(words)))
    replaced_words = {}

    for word, tag in words_sorted:
        if len(replaced_words) >= num_to_replace:
            break
        wordnet_pos = get_wordnet_pos(tag)
        if not wordnet_pos:
            continue

        antonyms = get_antonyms(word, wordnet_pos)
        hypernyms = get_hypernyms(word, wordnet_pos)
        synonyms = get_synonyms(word, wordnet_pos)

        new_word = word
        if wordnet_pos == wordnet.NOUN:
            if synonyms:
                new_word = random.choice(synonyms)
            elif hypernyms:
                new_word = random.choice(hypernyms)
            elif antonyms:
                new_word = random.choice(antonyms)
        elif wordnet_pos == wordnet.ADJ or wordnet_pos == wordnet.ADV:
            if antonyms:
                new_word = random.choice(antonyms)
            elif synonyms:
                new_word = random.choice(synonyms)
            elif hypernyms:
                new_word = random.choice(hypernyms)
        elif wordnet_pos == wordnet.VERB:
            if antonyms:
                new_word = random.choice(antonyms)
            elif hypernyms:
                new_word = random.choice(hypernyms)
            elif synonyms:
                new_word = random.choice(synonyms)
        else:
            if antonyms:
                new_word = random.choice(antonyms)
            elif hypernyms:
                new_word = random.choice(hypernyms)
            elif synonyms:
                new_word = random.choice(synonyms)

        if new_word != word:
            replaced_words[word] = new_word
            words[words.index(word)] = new_word

    print("Replaced words:", replaced_words)
    return ' '.join(words)

def extract_keywords_and_sentences(text):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    filtered_words = [token.text for token in doc if token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV']]

    kw_extractor = yake.KeywordExtractor()
    filtered_text = ' '.join(filtered_words)
    keywords = kw_extractor.extract_keywords(filtered_text)

    sentences = [sent.text for sent in doc.sents]

    for keyword, score in keywords:
        if len(keyword.split()) == 1:
            print(f"Keyword: {keyword}, Score: {score}")
            print(generate_sentence_with_word(text, keyword))

if __name__ == '__main__':
    while True:
        text = read_text()
        language = detect_language(text)
        print(f'The language of the text is: {language}')

        text = asyncio.run(translate_to_english(text))
        print(f'Translated text: {text}')

        stylometric_analysis(text)
        alt_text = generate_alternative_text(text)
        print(f"\nAlternative text: {alt_text}")

        extract_keywords_and_sentences(text)