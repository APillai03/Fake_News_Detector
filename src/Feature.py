import spacy
import re
import json
import textstat
import pandas as pd
from textblob import TextBlob

class Features:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.clickbait_keywords = [
            # Emotionally Charged & Shocking
            "shocking", "unbelievable", "outrageous", "jaw-dropping", "mind-blowing",
            "scandalous", "terrifying", "disturbing", "heartbreaking", "insane",
            "horrifying", "crazy", "emotional", "brutal", "shocking truth",
            "you won’t believe", "this is what happened",
            # Curiosity & Mystery
            "what happened next", "you’ll never guess", "hidden secrets", "top secret",
            "exposed", "uncovered", "finally revealed", "the truth about",
            "little-known facts", "mystery solved", "no one talks about", "the real reason",
            "experts won’t tell you", "watch till the end", "only few know", "newly discovered",
            # Superlatives & Absolutes
            "best ever", "worst ever", "ultimate", "most epic", "most unbelievable",
            "#1 trick", "top 10", "only way", "can't miss", "once in a lifetime",
            "guaranteed", "life-changing", "game-changer", "never before seen", "the only one you need",
            # Knowledge & Hacks
            "genius", "weird trick", "secret hack", "proven method", "little-known trick",
            "experts use this", "science-backed", "one simple trick", "learn this now",
            "do this every day", "avoid this mistake", "try this at home",
            "step-by-step guide", "here's how", "you’ve been doing it wrong",
        
            # Money & Success
            "make money fast", "earn $X per day", "passive income secrets", "quit your job",
            "millionaire habits", "success formula", "how I made $X", "financial freedom",
            "rich people do this", "broke to millionaire", "money-saving hacks", "get rich quick",
            "ultimate side hustle", "financial secrets",
        
            # Urgency & Exclusivity
            "limited time", "act fast", "don’t miss out", "before it’s gone", "ends tonight",
            "too late?", "hurry!", "only today", "get in now", "members only",
            "exclusive access", "secret invite", "just released", "early access", "be the first to know"
            ]
    def extract_article_features(self, text):
        doc = self.nlp(text)

        # Text length metrics
        num_chars = len(text)
        words = [token.text for token in doc if token.is_alpha]
        num_words = len(words)
        sentences = list(doc.sents)
        num_sentences = len(sentences)
        avg_sentence_length = round(num_words / num_sentences, 2) if num_sentences else 0

        # Capitalized words
        capitalized_words = [token.text for token in doc if token.text.isupper() and len(token.text) > 1]
        num_caps = len(capitalized_words)

        # Special punctuation
        num_exclamations = text.count('!')
        num_questions = text.count('?')

        # Clickbait detection
        text_lower = text.lower()
        has_clickbait = any(word in text_lower for word in self.clickbait_keywords)

        # Readability score
        readability_score = textstat.flesch_reading_ease(text)

        # Sentiment
        sentiment = TextBlob(text).sentiment
        polarity = sentiment.polarity
        subjectivity = sentiment.subjectivity

        # POS Ratios
        total_tokens = len([token for token in doc if token.is_alpha])
        pos_counts = {}
        for token in doc:
            if token.is_alpha:
                pos = token.pos_
                pos_counts[pos] = pos_counts.get(pos, 0) + 1
        pos_ratios = {k: round(v / total_tokens, 3) for k, v in pos_counts.items()}
        return {
            "num_characters": num_chars,
            "num_words": num_words,
            "num_sentences": num_sentences,
            "avg_sentence_length": avg_sentence_length,
            "num_capitalized_words": num_caps,
            "num_exclamations": num_exclamations,
            "num_questions": num_questions,
            "has_clickbait_words": has_clickbait,
            "readability_score": readability_score,
            "sentiment_polarity": polarity,
            "sentiment_subjectivity": subjectivity,
            "PROPN": pos_ratios.get("PROPN", 0),
            "ADV": pos_ratios.get("ADV", 0),
            "VERB": pos_ratios.get("VERB", 0),
            "DET": pos_ratios.get("DET", 0),
            "CCONJ": pos_ratios.get("CCONJ", 0),
            "PRON": pos_ratios.get("PRON", 0),
            "ADP": pos_ratios.get("ADP", 0),
            "PART": pos_ratios.get("PART", 0),
            "NOUN": pos_ratios.get("NOUN", 0),
            "ADJ": pos_ratios.get("ADJ", 0),
            "NUM": pos_ratios.get("NUM", 0),
            "SCONJ": pos_ratios.get("SCONJ", 0),
            "AUX": pos_ratios.get("AUX", 0)
        }
