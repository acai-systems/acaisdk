import json
import pandas as pd
from nltk.tokenize import sent_tokenize
from nltk.tokenize import WordPunctTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from load_write_data import loadData
lemmatizer = WordNetLemmatizer()
import os.path


def load_data(fn):
    print("load_data is file? " + str(os.path.isfile(fn)))
    df = pd.read_json(open(fn, "r", encoding="utf8"))
    data = df.to_dict(orient='list')
    return data

def tokenize_sentence_in_context(data):
    '''
    Split sentences for all pasages, sentences are candidate answers for
    the questions from the same paragraph
    '''
    for article in data['data']:
        for context in article['paragraphs']:
            context['context_sent'] = sent_tokenize(context['context'])

def getlabel(context, lens, ansPos):
    lenSum = 0
    for i in range(len(lens)):
        while not context[lenSum].isalnum():  # either a num/lettere
            lenSum += 1
        lenSum += lens[i]
        if ansPos <= lenSum:
            return i
    return -1


def buildLabel(data):
    '''
    Create label of the context sentence for correct answers
    '''
    for article in data['data']:
        for context in article['paragraphs']:
            # get num of context sentences
            context['num_candidate_ans'] = len(context['context_sent'])
            # get context sentences length
            sentLens = [len(sent) for sent in context['context_sent']]
            # get label from context
            for qas in context['qas']:
                for ans in qas['answers']:
                    ans['ans_label'] = getlabel(context['context'],
                                                sentLens,
                                                ans['answer_start'])


def remove_non_alnum(word_list):
    return [word.lower() for word in word_list if word.isalnum()]

def tokenize_words(dataset):
    '''
    Word tokenization for questions and context with removal of non-alphanumeric tokens
    '''
    word_punct_tokenizer = WordPunctTokenizer()
    for article in dataset['data']:
        for qas_context in article['paragraphs']:
            # tokenize all context_sentences
            qas_context['context_sents_words'] = list()
            for sentence in qas_context['context_sent']:
                word_list = word_punct_tokenizer.tokenize(sentence)
                qas_context['context_sents_words'].append(remove_non_alnum(word_list))

            # tokenize questionstokenize_wordstokenize_words
            for qas in qas_context['qas']:
                question = qas['question']
                word_list = word_punct_tokenizer.tokenize(question)
                qas['question_words'] = remove_non_alnum(word_list)


def lemmatize_tokens(tokens):
    lemmas = [lemmatizer.lemmatize(token, pos='v') for token in tokens]
    return lemmas

def lemmatization(dataset):
    word_punct_tokenizer = WordPunctTokenizer()
    for article in dataset['data']:
        for qas_context in article['paragraphs']:
            # lemmatize all context_sentences_words
            qas_context['context_sents_lemmas'] = list()
            for word_list in qas_context['context_sents_words']:
                qas_context['context_sents_lemmas'].append(lemmatize_tokens(word_list))

            # lemmatize questions
            for qas in qas_context['qas']:
                word_list = qas['question_words']
                qas['question_lemmas'] = lemmatize_tokens(word_list)


def remove_stopwords(tokens):
    stopwords_set = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token not in set(stopwords_set)]
    return filtered_tokens


def stopwords_elimination(dataset):
    for article in dataset['data']:
        for qas_context in article['paragraphs']:
            # remove stopwords from context_sentences_lemmas
            qas_context['context_sents_lemmas_without_stopwords'] = list()
            for lemmas_list in qas_context['context_sents_lemmas']:
                qas_context['context_sents_lemmas_without_stopwords'].append(remove_stopwords(lemmas_list))

            # remove stopwords from questions lemmas
            for qas in qas_context['qas']:
                lemmas_list = qas['question_lemmas']
                qas['question_lemmas_without_stopwords'] = remove_stopwords(lemmas_list)


def pipeline(data):
    tokenize_sentence_in_context(data)
    buildLabel(data)
    tokenize_words(data)
    lemmatization(data)
    stopwords_elimination(data)

def saveData(data, fn):
    with open(fn, 'w') as f:
        json.dump(data, f)

def preprocess(path, newPrepocess=True):
    if newPrepocess:
        # Load dataset
        print("*** load data\t train ***")
        base_path = os.path.dirname(os.path.abspath(path+"train.json"))
        print("base file path is " + base_path)
        train = load_data(os.path.join(base_path, "train.json"))
        print("*** load data\t dev ***")
        dev = load_data(os.path.join(base_path, "dev.json"))

        # Preprocess dataset
        print("*** preprocess data\t train ***")
        pipeline(train)
        print("*** preprocess data\t dev ***")
        pipeline(dev)

        # save preprocessed data
        saveData(train, path + 'train-preprocessed.json')
        saveData(dev, path + 'dev-preprocessed.json')
        return train, dev
    else:
        # Load
        print("** load existing preprocessed data ***")
        train = loadData(path + 'train-preprocessed.json')
        dev = loadData(path + 'dev-preprocessed.json')
        return train, dev








