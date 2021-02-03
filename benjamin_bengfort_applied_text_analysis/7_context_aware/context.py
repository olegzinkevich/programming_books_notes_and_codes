What we haven’t taken into account yet, however, is the context in which the words
appear, which we instinctively know plays a huge role in conveying meaning. Consider
the following phrases: “she liked the smell of roses” and “she smelled like roses.”
Using the text normalization techniques presented in previous chapters such as stopwords
removal and lemmatization, these two utterances would have identical bag-ofwords
vectors though they have completely different meanings.

This does not mean that bag-of-words models should be completely discounted, and
in fact, bag-of-words models are usually very useful initial models. Nonetheless,
lower performing models can often be significantly improved with the addition of
contextual feature extraction. One simple, yet effective approach is to augment models
with grammars to create templates that help us target specific types of phrases,
which capture more nuance than words alone.

In this chapter, we will begin by using a grammar to extract key phrases from our
documents. Next, we will explore n-grams and discover significant collocations we can
use to augment our bag-of-words models.