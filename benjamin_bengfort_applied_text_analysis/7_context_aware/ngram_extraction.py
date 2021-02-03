# Grammar-based feature extraction is also somewhat inflexible, because we must
# begin by defining a grammar. It is often very difficult to know in advance which
# grammar pattern will most effectively capture the high-signal terms and phrases
# within a text.
# We can address these challenges iteratively, by experimenting with many different
# grammars or by training our own custom part-of-speech tagger. However, in this section
# we will explore another option, backing off from grammar to n-grams, which
# will give us a more general way of identifying sequences of tokens.

# To identify all of the n-grams from our text, we simply slide a fixed-length window # over a list of words until the window reaches the end of the list. We can do this in pure Python as follows:
def ngrams(words, n=2):
    # This function ranges a start index from 0 to the position that is exactly one n-gram away from the end of the word list. It then slices the word list from the start index to n-gram length, returning an immutable tuple.
    for idx in range(len(words)-n+1):
        yield tuple(words[idx:idx+n])


words = [
"The", "reporters", "listened", "closely", "as", "the", "President",
"of", "the", "United", "States", "addressed", "the", "room", ".",
]

for ngram in ngrams(words, n=3):
    print(ngram)

# So how do we decide which n to choose? Consider an application where we are using
# n-grams to identify candidates for named entity recognition. If we consider a chunk
# size of n=2, our results include “The reporters,” “the President,” “the United,” and “the
# room.” While not perfect, this model successfully identifies three of the relevant entities
# as candidates in a lightweight fashion.

On the other hand, a model based on the small n-gram window of 2 would fail to
capture some of the nuance of the original text. For instance, if our sentence is from a
text that references multiple heads of state, “the President” could be somewhat ambiguous.
In order to capture the entirety of the phrase “the President of the United
States,” we would have to set n=6: Unfortunately, as we can see in the results above, if we build a model based on an ngram
order that is too high, it will be very unlikely that we’ll see any repeated entities.

Choosing n can also be considered as balancing the trade-off between bias and variance.
A small n leads to a simpler (weaker) model, therefore causing more error due
to bias. A larger n leads to a more complex model (a higher-order model), thus causing
more error due to variance. Just as with all supervised machine learning problems,
we have to strike the right balance between the sensitivity and the specificity of
our model.

Significant Collocations

Now that our corpus reader is aware of n-grams, we can incorporate these features
into our downstream models by vectorizing our text using n-grams as vector elements
instead of simply vocabulary. However, using raw n-grams will produce many,
many candidates, most of which will not be relevant.

In practice, this is too high a computational cost to be useful in most applications.
The solution is to compute conditional probability. For example, what is the likelihood
that the tokens ('the', 'fall') appear in the text given the token 'during'? We
can compute empirical likelihoods by calculating the frequency of the (n-1)-gram
conditioned by the first token of the n-gram. Using this technique we can value ngrams
that are more often used together such as ('corn', 'maze') over rarer compositions
that are less meaningful.

The idea of some n-grams having more value than others leads to another tool in the
text analysis toolkit: significant collocations. Collocation is an abstract synonym for ngram
(without the specificity of the window size) and simply means a sequence of
tokens whose likelihood of co-occurrence is caused by something other than random chance. Using conditional probability, we can test the hypothesis that a specified collocation
is meaningful

NLTK contains two tools to discover significant collocations: the Collocation
Finder, which finds and ranks n-gram collocations, and NgramAssocMeasures, which
contains a collection of metrics to score the significance of a collocation. Both utilities
are dependent on the size of n and the module contains bigram, trigram, and quadgram
ranking utilities.

look - collocations.py


