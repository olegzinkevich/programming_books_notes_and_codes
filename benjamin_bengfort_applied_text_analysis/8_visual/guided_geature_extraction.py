# we used the part-of-speech tags together with a grammar to perform
# keyphrase extraction. One of the challenges of this kind of feature engineering is that
# it can be very difficult to know a priori which grammar to use to find significant keyphrases.
# Generally, our strategy is to use heuristics and experimentation until we land
# on a regular expression that does a good job at capturing the high-signal keyphrases.
# This strategy actually works quite well with grammatical English text. But what if the
# text with which we are working is ungrammatical, or rife with spelling and punctuation
# errors? In these cases, our out-of-the-box part-of-speech tagger may do more
# harm than good.
#
# The Yellowbrick library offers a feature that enables the user to print out colorized
# text that illustrates different parts of speech. A PosTagVisualizer colorizes text to
# enable the user to visualize the proportions of nouns, verbs, etc., and to use this information
# to make decisions about part-of-speech tagging, text normalization (e.g.,
# stemming versus lemmatization), and vectorization.
# The transform method transforms the raw text input for the part-of-speech tagging
# visualization. Note that it requires that documents be in the form of (tag, token)
# tuples

from nltk import pos_tag, word_tokenize
from yellowbrick.text.postag import PosTagVisualizer

pie = """
In a small saucepan, combine sugar and eggs
until well blended. Cook over low heat, stirring
constantly, until mixture reaches 160° and coats
the back of a metal spoon. Remove from the heat.
Stir in chocolate and vanilla until smooth. Cool
to lukewarm (90°), stirring occasionally. In a small
bowl, cream butter until light and fluffy. Add cooled
chocolate mixture; beat on high speed for 5 minutes
or until light and fluffy. In another large bowl,
beat cream until it begins to thicken. Add
confectioners' sugar; beat until stiff peaks form.
Fold into chocolate mixture. Pour into crust. Chill
for at least 6 hours before serving. Garnish with
whipped cream and chocolate curls if desired.
"""
tokens = word_tokenize(pie)
tagged = pos_tag(tokens)

visualizer = PosTagVisualizer()
visualizer.transform(tagged)

print(' '.join((visualizer.colorize(token, color) for color, token in visualizer.tagged)))
print('\n')

# Most informative features

# Identifying the most informative (i.e., predictive) features from a dataset is a key part of the model selection triple.

# One method for visually exploring text is with frequency distributions. In the context
# of a text corpus, a frequency distribution tells us the prevalence of a vocabulary item
# or token.