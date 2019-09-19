# -*- coding: utf-8 -*-

""" Use DeepMoji to score texts for emoji distribution.

The resulting emoji ids (0-63) correspond to the mapping
in emoji_overview.png file at the root of the DeepMoji repo.

"""
import json
import numpy as np
from simple_deepmoji.sentence_tokenizer import SentenceTokenizer
from simple_deepmoji.model_def import deepmoji_emojis
from simple_deepmoji.global_variables import PRETRAINED_PATH, VOCAB_PATH
from simple_deepmoji.emoji_maps import DEEPMOJI_MAP, EMOJI_TO_EMOTION, EMOJI_TO_EMOTICON, EMOJI_POLARITY, EMOJI_DATA


class Emoji:
    def __init__(self, emoji_id):
        self.emoji_id = emoji_id

    @property
    def name(self):
        return DEEPMOJI_MAP.get(self.emoji_id)

    @property
    def emoticon_data(self):
        return EMOJI_TO_EMOTICON.get(self.name)

    @property
    def data(self):
        return EMOJI_DATA.get(self.name)

    @property
    def emotion(self):
        return EMOJI_TO_EMOTION.get(self.emoji_id)

    @property
    def polarity(self):
        return EMOJI_POLARITY.get(self.name.replace(":", ""))

    def __str__(self):
        return self.name


class DeepMoji:
    def __init__(self, maxlen=30, vocab_path=VOCAB_PATH, model_path=PRETRAINED_PATH, verbose=False):
        self.verbose = verbose
        if self.verbose:
            print('Tokenizing using dictionary from {}'.format(vocab_path))
        with open(vocab_path, 'r') as f:
            self.vocabulary = json.load(f)
        self.st = SentenceTokenizer(self.vocabulary, maxlen)
        if self.verbose:
            print('Loading model from {}.'.format(model_path))
        self.model = deepmoji_emojis(maxlen, model_path)
        self.model_summary()

    def model_summary(self):
        self.model.summary()

    def tokenize(self, sentences):
        tokenized, _, _ = self.st.tokenize_sentences(sentences)
        return tokenized

    def encode(self, sentences):
        tokenized = self.tokenize(sentences)
        if self.verbose:
            print('Encoding texts..')
        encoding = self.model.predict(tokenized)
        return encoding

    def predict(self, sentences):
        tokenized = self.tokenize(sentences)
        if self.verbose:
            print('Running predictions.')
        prob = self.model.predict(tokenized)
        # Find top emojis for each sentence. Emoji ids (0-63)
        # correspond to the mapping in emoji_overview.png
        # at the root of the DeepMoji repo.
        scores = []
        # Text,Top5%,Emoji_1,Emoji_2,Emoji_3,Emoji_4,Emoji_5,Pct_1,Pct_2,Pct_3,Pct_4,Pct_5
        for i, t in enumerate(sentences):
            t_prob = prob[i]
            ind_top = self.top_elements(t_prob, 5)
            top5 = sum(t_prob[ind_top])
            emojis = ind_top
            probs = [t_prob[ind] for ind in ind_top]
            prediction = {
                "sentence": t,
                "emoji_codes": list(emojis),
                "emoji_names": [self.emoji_id_to_emoji_name(moji) for moji in emojis],
                "emoji_scores": probs
            }
            scores.append(prediction)
            if self.verbose:
                print(prediction)
        return scores

    @staticmethod
    def top_elements(array, k):
        ind = np.argpartition(array, -k)[-k:]
        return ind[np.argsort(array[ind])][::-1]

    @staticmethod
    def emoji_id_to_emoji_name(emoji_id):
        return DEEPMOJI_MAP.get(emoji_id) or emoji_id


if __name__ == "__main__":
    TEST_SENTENCES = [u'I love mom\'s cooking',
                      u'I love how you never reply back..',
                      u'I love cruising with my homies',
                      u'I love messing with yo mind!!',
                      u'I love you and now you\'re just gone..',
                      u'This is shit',
                      u'This is the shit']

    dm = DeepMoji()
    dm.predict(TEST_SENTENCES)
