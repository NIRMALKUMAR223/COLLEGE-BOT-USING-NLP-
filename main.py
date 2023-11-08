import io
import random
import string
import warnings
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.stem import WordNetLemmatizer

nltk.download('popular', quiet=True)
nltk.download('punkt')
nltk.download('wordnet')

# Reading in the corpus
with open('college.txt', 'r', encoding='utf8', errors='ignore') as fin:
    raw = fin.read().lower()

# Tokenization
sent_tokens = nltk.sent_tokenize(raw)
word_tokens = nltk.word_tokenize(raw)

# Preprocessing
lemmer = WordNetLemmatizer()

def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

# Keyword Matching
GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey",)
GREETING_RESPONSES = ["hi", "hey", "Hello again!", "hi there", "hello", "I am glad! You are talking to me"]

def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

# Generating response
def response(user_response):
    robo_response = ''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if (req_tfidf == 0):
        robo_response = robo_response + "\nI am not entirely sure what you're referring to. Could you please provide more context or clarify your question?\nI would be happy to help once I understand your inquiry better."
        return robo_response
    else:
        robo_response = robo_response + sent_tokens[idx]
        return robo_response

flag = True
print("Alice: \nMy name is Alice. I'm here to assist you with information about our institution. If you want to exit, type 'Bye!'")
name_user = input("May I know your name: ")
print(f"Hi {name_user}, welcome to kit college.")
print("How can I assist you today?")
while flag:
    user_response = input("You: ")
    user_response = user_response.lower()
    if user_response != 'bye':
        if user_response == 'thanks' or user_response == 'thank you':
            flag = False
            print("Alice: \nYou're welcome! If you have more questions in the future, feel free to ask.")
        else:
            if greeting(user_response) is not None:
                print(f"Alice: \n{greeting(user_response).title()}")
                print("\n")
            else:
                print("Alice: \n", end="")
                print(response(user_response).title())
                sent_tokens.remove(user_response)
                print("\n")
    else:
        flag = False
        print("Alice: \nGoodbye! If you have more questions in the future, don't hesitate to return.")
