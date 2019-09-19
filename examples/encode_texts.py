# -*- coding: utf-8 -*-

""" Use DeepMoji to encode texts into emotional feature vectors.
"""
from simple_deepmoji import DeepMoji

model_path = "/home/user/PycharmProjects/simple_deepmoji/examples/model/deepmoji_weights.hdf5"
vocab_path="/home/user/PycharmProjects/simple_deepmoji/examples/model/vocabulary.json"

TEST_SENTENCES = [u'I love mom\'s cooking',
                  u'I love how you never reply back..',
                  u'I love cruising with my homies',
                  u'I love messing with yo mind!!',
                  u'I love you and now you\'re just gone..',
                  u'This is shit',
                  u'This is the shit']

dm = DeepMoji(model_path=model_path, vocab_path=vocab_path)
encoding = dm.encode(TEST_SENTENCES)


print('First 5 dimensions for sentence: {}'.format(TEST_SENTENCES[0]))
print(encoding[0, :5])

# Now you could visualize the encodings to see differences,
# run a logistic regression classifier on top,
# or basically anything you'd like to do.
