import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
import spacy

nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()
nlp = spacy.load('en_core_web_sm')

def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment
    vader_scores = sia.polarity_scores(text)
    return sentiment, vader_scores

from spacy.lang.en.stop_words import STOP_WORDS

def extract_keywords(text):
    doc = nlp(text)
    keywords = [ent.text for ent in doc.ents if ent.label_ in ['PERSON', 'ORG', 'GPE', 'EVENT', 'WORK_OF_ART', 'PRODUCT']]
    if not keywords:
        keywords = [token.text for token in doc if token.pos_ != 'PRON' and token.text.lower() not in STOP_WORDS]
    return keywords if keywords else ["the topic"]


import random

def generate_friendly_comment(keywords, sentiment):
    friendly_comments = [
        "Your positive perspective on {} is appreciated!",
        "You've highlighted an important aspect with {}!",
        "Thanks for sharing your thoughts about {} in such a positive light!",
        "I'm glad you brought up {} in such a positive manner!",
        "You've offered valuable insights about {}!",
        "Great job in noticing the importance of {}!",
        "Your perspective on {} is refreshing!",
        "Your optimism about {} is uplifting!",
        "Your insights on {} are quite insightful!",
        "I appreciate your positive outlook on {}!"
    ]

    negative_comments = [
        "While you mentioned {}, let's consider other viewpoints.",
        "Your thoughts on {} are appreciated, but let's explore more.",
        "It's interesting that you mentioned {}, but let's discuss further.",
        "I hear your concerns about {}, but let's consider the bright side!",
        "While you pointed out {}, let's explore other perspectives.",
        "You've raised an interesting perspective on {}, but let's look deeper.",
        "While you highlighted {}, let's focus on the good things!",
        "Your thoughts on {} are noted, but let's consider alternative viewpoints.",
        "Your insights on {} are valuable, but let's delve deeper.",
        "While you brought up {}, let's discuss the positive aspects!"
    ]

    if sentiment.polarity > 0:
        return random.choice(friendly_comments).format(random.choice(keywords))
    else:
        return random.choice(negative_comments).format(random.choice(keywords))

def generate_funny_comment(keywords):
    funny_comments = [
        "That's hilarious! You've got a great sense of humor mentioning {}!",
        "Thanks for bringing some humor with {}!",
        "You've brightened my day with {}!",
        "I'm laughing so hard at {}! You've nailed it!",
        "That joke about {} was gold! Thanks for sharing!",
        "I can't stop laughing at {}! You really know how to make it funny!",
        "Wow, you really know how to make {} funny!",
        "Thanks for the laugh with {}! You're hilarious!",
        "You've made my day with that joke about {}!",
        "I'm still chuckling at {}! Well done!"
    ]

    return random.choice(funny_comments).format(random.choice(keywords))


def generate_congratulating_comment(keywords, sentiment):
    positive_comments = [
        "Congrats on your unique perspective on {}!",
        "Well done on highlighting {}!",
        "Kudos for your thoughts on {}!",
        "You've offered valuable insights about {}!",
        "Great job in noticing the importance of {}!",
        "You've brought up a significant point regarding {}!",
        "Nice work on {}! You've done well!",
        "I'm impressed by your insights on {}!",
        "You've hit the nail on the head with {}!",
        "Excellent job in pointing out the significance of {}, keep it up!"
    ]
    negative_comments = [
        "Interesting take on {}, well done!",
        "Your thoughts on {} are quite insightful!",
        "Good catch on {}, keep it up!",
        "You've brought up a significant point regarding {}, keep it going!",
        "Nice try with {}, but let's consider other viewpoints.",
        "Interesting perspective on {}, let's discuss further!",
        "You've raised some good points about {}, but let's explore more.",
        "While you highlighted {}, let's look deeper into the issue.",
        "Interesting choice of {}, but there may be other angles to consider.",
        "You might have a different take on {}, but I see it differently."
    ]

    return random.choice(positive_comments).format(random.choice(keywords)) if sentiment.polarity > 0 else random.choice(negative_comments).format(random.choice(keywords))

def generate_questioning_comment(keywords):
        questioning_pronouns = ['You', 'Your']
        comments = [
            f"Can {random.choice(questioning_pronouns)} explain more about {random.choice(keywords)}?",
            f"What does {random.choice(questioning_pronouns)} mean by {random.choice(keywords)}?",
            f"Why is {random.choice(keywords)} significant from {random.choice(questioning_pronouns)} perspective?",
            f"Can {random.choice(questioning_pronouns)} provide further insights on {random.choice(keywords)}?",
            f"{random.choice(questioning_pronouns)} mentioned {random.choice(keywords)}, but what's the reasoning behind it?",
            f"What's the thought process behind {random.choice(keywords)}, {random.choice(questioning_pronouns)}?",
            f"Any particular reason behind {random.choice(keywords)}, {random.choice(questioning_pronouns)}?",
            f"Could {random.choice(questioning_pronouns)} elaborate more on {random.choice(keywords)}?",
            f"{random.choice(questioning_pronouns)} seem interested in {random.choice(keywords)}. What's the story behind it?",
            f"Curious to know more about {random.choice(keywords)}, {random.choice(questioning_pronouns)}!"
        ]
        return random.choice(comments)

def generate_disagreement_comment(keywords):
    disagreement_pronouns = ['You', 'Your']
    comments = [
        f"I'm not sure I agree with {random.choice(disagreement_pronouns)} view on {random.choice(keywords)}.",
        f"Do {random.choice(disagreement_pronouns)} really think {random.choice(keywords)} is correct?",
        f"I have a different opinion on {random.choice(keywords)}.",
        f"{random.choice(disagreement_pronouns)} might have a point, but I see it differently.",
        f"While {random.choice(disagreement_pronouns)} mentioned {random.choice(keywords)}, I disagree.",
        f"I see where {random.choice(disagreement_pronouns)} coming from with {random.choice(keywords)}, but I beg to differ.",
        f"{random.choice(disagreement_pronouns)}'re entitled to {random.choice(keywords)}, but I have a different perspective.",
        f"I respectfully disagree with {random.choice(disagreement_pronouns)} view on {random.choice(keywords)}.",
        f"I'm not convinced by {random.choice(disagreement_pronouns)} argument regarding {random.choice(keywords)}.",
        f"{random.choice(disagreement_pronouns)} might have a different take on {random.choice(keywords)}, but I see it differently."
    ]
    return random.choice(comments)

# Now, let's call these functions to generate comments for a given text
def generate_comments(text):
    sentiment, vader_scores = analyze_sentiment(text)
    keywords = extract_keywords(text)
    if not keywords:
        return {
            "Friendly": "Great read!",
            "Funny": "This made me smile!",
            "Congratulating": "Well done!",
            "Questioning": "Can you elaborate more?",
            "Disagreement": "I have a different opinion."
        }
    comments = {
        "Friendly": generate_friendly_comment(keywords, sentiment),
        "Funny": generate_funny_comment(keywords),
        "Congratulating": generate_congratulating_comment(keywords, sentiment),
        "Questioning": generate_questioning_comment(keywords),
        "Disagreement": generate_disagreement_comment(keywords)
    }
    return comments
