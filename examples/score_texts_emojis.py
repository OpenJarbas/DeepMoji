from simple_deepmoji import DeepMoji

model_path = "/home/user/PycharmProjects/simple_deepmoji/examples/model/deepmoji_weights.hdf5"
vocab_path = "/home/user/PycharmProjects/simple_deepmoji/examples/model/vocabulary.json"

TEST_SENTENCES = [u'I love mom\'s cooking',
                  u'I love how you never reply back..',
                  u'I love cruising with my homies',
                  u'I love messing with yo mind!!',
                  u'I love you and now you\'re just gone..',
                  u'This is shit',
                  u'This is the shit']

dm = DeepMoji(model_path=model_path, vocab_path=vocab_path)
predictions = dm.predict(TEST_SENTENCES)
