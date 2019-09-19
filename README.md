# Simple Deepmoji

Deepmoji made easy, all credits to the [original deepmoji](https://github.com/bfelbo/DeepMoji) repo


## Install

    pip install simple_deepmoji
    
    
## Usage

Predictions

```python

from simple_deepmoji import DeepMoji

model_path = "model/deepmoji_weights.hdf5"
vocab_path = "model/vocabulary.json"

TEST_SENTENCES = [u'I love mom\'s cooking',
                  u'I love how you never reply back..',
                  u'I love cruising with my homies',
                  u'I love messing with yo mind!!',
                  u'I love you and now you\'re just gone..',
                  u'This is shit',
                  u'This is the shit']

dm = DeepMoji(model_path=model_path, vocab_path=vocab_path)
predictions = dm.predict(TEST_SENTENCES)

# {'emoji_codes': [36, 4, 8, 16, 47],
# 'emoji_names': [':stuck_out_tongue_closed_eyes:',
#                 ':heart_eyes:',
#                 ':heart:',
#                 ':blush:',
#                 ':yellow_heart:'],
# 'emoji_scores': [0.49053323,
#                  0.087657765,
#                  0.030925231,
#                  0.029632866,
#                  0.028943174],
# 'sentence': "I love mom's cooking"}

```

Use DeepMoji to encode texts into emotional feature vectors.
```python
from simple_deepmoji import DeepMoji

model_path = "model/deepmoji_weights.hdf5"
vocab_path = "model/vocabulary.json"

dm = DeepMoji(model_path=model_path, vocab_path=vocab_path)
encoding = dm.encode(SENTENCES)

print('First 5 dimensions for sentence: {}'.format(SENTENCES[0]))
print(encoding[0, :5])

# Now you could visualize the encodings to see differences,
# run a logistic regression classifier on top,
# or basically anything you'd like to do.
```

Easily get extra information regarding emoji results

```python
from simple_deepmoji import DeepMoji, Emoji

model_path = "model/deepmoji_weights.hdf5"
vocab_path = "model/vocabulary.json"

TEST_SENTENCES = [u'I love mom\'s cooking']

dm = DeepMoji(model_path=model_path, vocab_path=vocab_path)
prediction = dm.predict(TEST_SENTENCES)[0]

for emoji_id in prediction["emoji_codes"]:
    emoji = Emoji(emoji_id)
    print("Name", emoji.name)
    print("Afiin polarity", emoji.polarity)
    print("Data", emoji.data)
    print("Emotion (hand tagged)", emoji.emotion)
    print("Emoticon data", emoji.emoticon_data)
    print("____________")
    
# Name :heart:
# Afiin polarity 3
# Data {'emoji': ' ❤️ ', 'tags': ['love']}
# Emotion (hand tagged) Love
# Emoticon data {'emoji': ' ❤️ ', 'emoticons': [' `<3` '], 'tags': [' love ']}
```

