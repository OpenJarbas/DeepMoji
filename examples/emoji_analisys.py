from simple_deepmoji import DeepMoji, Emoji
from pprint import pprint

model_path = "/home/user/PycharmProjects/simple_deepmoji/examples/model/deepmoji_weights.hdf5"
vocab_path = "/home/user/PycharmProjects/simple_deepmoji/examples/model/vocabulary.json"

TEST_SENTENCES = [u'I love mom\'s cooking']

dm = DeepMoji(model_path=model_path, vocab_path=vocab_path)
prediction = dm.predict(TEST_SENTENCES)[0]
pprint(prediction)

for emoji_id in prediction["emoji_codes"]:
    emoji = Emoji(emoji_id)
    print("Name", emoji.name)
    print("Afiin polarity", emoji.polarity)
    print("Data", emoji.data)
    print("Emotion (hand tagged)", emoji.emotion)
    print("Emoticon data", emoji.emoticon_data)
    print("____________")