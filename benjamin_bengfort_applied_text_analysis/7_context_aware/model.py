#!/usr/bin/env python3

# работает, предугадывает следующее слово.

# Consider an application where a user will enter the first few words of a phrase, then
# suggest additional text based on the most likely next words (like a Google search). ngram
# models utilize the statistical frequency of n-grams to make decisions about text.
# To compute an n-gram language model that predicts the next word after a series of
# words, we would first count all n-grams in the text and then use those frequencies to
# predict the likelihood of the last token in the n-gram given the tokens that precede it.

# To build a language model that can generate text, our next step is to create a class that
# puts together the pieces we have stepped through in the above sections and implement
# one additional technique: conditional frequency.

# Frequency and Conditional Frequency
# We first explored the concept of token frequency in Figure 4-2, where we used frequency
# representations with our bag-of-words model with the assumption that word
# count could sufficiently approximate a document’s contents to differentiate it from
# others. Frequency is also a useful feature with n-gram modeling, where the frequency
# with which an n-gram occurs in the training corpus might reasonably lead us to
# expect to see that n-gram in new documents

# Imagine we are reading a book one word at a time and we want to compute the probability
# of the next word we’ll see. A naive choice would be to assign the highest probability
# to the words that appear most frequently in the text,
#
# However, we know that this basic use of frequency is not enough; if we’re starting a
# sentence some words have higher probability than other words and some words are
# much more likely given preceding words. For example, asking the question what is
# the probability of the word “chair” following “lawn” is very different than the probability
# of the word “chair” following “lava” (or “lamp”). These likelihoods are informed
# by conditional probabilities and are formulated as P(chair|lawn) (read as “the probability
# of chair given lawn”). To model these probabilities, we need to be able to compute
# the conditional frequencies of each of the possible n-gram windows.



import nltk

from math import log
from collections import Counter, defaultdict

from nltk.util import ngrams
from nltk.probability import ProbDistI, FreqDist, ConditionalFreqDist

from reader import PickledCorpusReader


# Now we can define a quick method (outside of our NgramCounter class definition)
# that instantiates the counter and computes the relevant frequencies. Our
# count_ngrams function takes as parameters the desired n-gram size, the vocabulary,
# and a list of sentences represented as comma-separated strings.
def count_ngrams(n, vocabulary, texts):
    counter = NgramCounter(n, vocabulary)
    counter.train_counts(texts)
    return counter

# We begin by defining an NgramCounter class that can keep track of conditional
# frequencies of all subgrams from unigrams up to n-grams using FreqDist and
# ConditionalFreqDist. Our class also implements the sentence padding we explored
# earlier in the chapter, and detects words that are not in the vocabulary of the original
# corpus.

# Our NgramCounter class gives us the ability to transform a corpus into a conditional
# frequency distribution of n-grams.

class NgramCounter(object):
    """
    The NgramCounter class counts ngrams given a vocabulary and ngram size.
    """

    def __init__(self, n, vocabulary, unknown="<UNK>"):
        """
        n is the size of the ngram
        """
        if n < 1:
            raise ValueError("ngram size must be greater than or equal to 1")

        self.n = n
        self.unknown = unknown
        self.padding = {
            "pad_left": True,
            "pad_right": True,
            "left_pad_symbol": "<s>",
            "right_pad_symbol": "</s>"
        }

        self.vocabulary = vocabulary
        self.allgrams = defaultdict(ConditionalFreqDist)
        self.ngrams = FreqDist()
        self.unigrams = FreqDist()

    # Next, we will create a method for the NgramCounter class that enables us to systematically
# compute the frequency distribution and conditional frequency distribution for
# the requested n-gram window.
    def train_counts(self, training_text):
        for sent in training_text:
            checked_sent = (self.check_against_vocab(word) for word in sent)
            sent_start = True
            for ngram in self.to_ngrams(checked_sent):
                self.ngrams[ngram] += 1
                context, word = tuple(ngram[:-1]), ngram[-1]
                if sent_start:
                    for context_word in context:
                        self.unigrams[context_word] += 1
                    sent_start = False

                for window, ngram_order in enumerate(range(self.n, 1, -1)):
                    context = context[window:]
                    self.allgrams[ngram_order][context][word] += 1
                self.unigrams[word] += 1

    def check_against_vocab(self, word):
        if word in self.vocabulary:
            return word
        return self.unknown

    def to_ngrams(self, sequence):
        """
        Wrapper for NLTK ngrams method
        """
        return ngrams(sequence, self.n, **self.padding)

# In the context of our hypothetical next word prediction
# application, we need a mechanism for scoring the possible candidates for next
# words after an n-gram so we can provide the most likely. In other words, we need a
# model that computes the probability of a token, t, given a preceding sequence, s.

# One straightforward way to estimate the probability of the n-gram (s,t) is by computing
# its relative frequency. This is the number of times we see t appear as the next
# word after s in the corpus, divided by the total number of times we observe s in the
# corpus. The resulting ratio gives us a maximum likelihood estimate for the n-gram
# (s,t).

# We will start by creating a class, BaseNgramModel, that will take as input an Ngram
# Counter object and produce a language model. We will initialize the BaseNgramModel
# model with attributes to keep track of the highest order n-grams from the trained
# NgramCounter, as well as the conditional frequency distributions of the n-grams, the
# n-grams themselves, and the vocabulary.

class BaseNgramModel(object):
    """
    The BaseNgramModel creates an n-gram language model.
    This base model is equivalent to a Maximum Likelihood Estimation.
    """

    def __init__(self, ngram_counter):
        """
        BaseNgramModel is initialized with an NgramCounter.
        """
        self.n = ngram_counter.n
        self.ngram_counter = ngram_counter
        self.ngrams = ngram_counter.ngrams
        self._check_against_vocab = self.ngram_counter.check_against_vocab

    def check_context(self, context):
        """
        Ensures that the context is not longer than or equal to the model's
        n-gram order.

        Returns the context as a tuple.
        """
        if len(context) >= self.n:
            raise ValueError("Context too long for this n-gram")

        return tuple(context)

    # Next, inside our BaseNgramModel class, we create a score method to compute the relative
# frequency for the word given the context, checking first to make sure that the
# context is always shorter than the highest order n-grams from the trained Ngram
# Counter. Since the ngrams attribute of the BaseNgramModel is an NLTK Conditional
# FreqDist, we can retrieve the FreqDist for any given context, and get its relative
# frequency with freq:

    def score(self, word, context):
        """
        For a given string representation of a word, and a string word context,
        returns the maximum likelihood score that the word will follow the
        context.
        """
        context = self.check_context(context)

        return self.ngrams[context].freq(word)

#     # In practice, n-gram probabilities tend to be pretty small, so they are often represented
# as log probabilities instead. For this reason, we’ll create a logscore method that
# transforms the result of our score method into log format, unless the score is less
# than or equal to zero, in which case we’ll return negative infinity:
    def logscore(self, word, context):
        """
        For a given string representation of a word, and a word context,
        computes the log probability of this word in this context.
        """
        score = self.score(word, context)
        if score == 0.0:
            return float("-inf")

        return log(score, 2)

    # Now that we have methods for scoring instances of particular n-grams, we want a
# method to score the language model as a whole, which we will do with entropy. We can create an entropy method for our BaseNgramModel by taking the average log probability of every n-gram from our NgramCounter.
    def entropy(self, text):
        """
        Calculate the approximate cross-entropy of the n-gram model for a
        given text represented as a list of comma-separated strings.
        This is the average log probability of each word in the text.
        """
        normed_text = (self._check_against_vocab(word) for word in text)
        entropy = 0.0
        processed_ngrams = 0
        for ngram in self.ngram_counter.to_ngrams(normed_text):
            context, word = tuple(ngram[:-1]), ngram[-1]
            entropy += self.logscore(word, context)
            processed_ngrams += 1
        return - (entropy / processed_ngrams)

    # it is common to evaluate the predictive power of a model by measuring
# its perplexity, which we can compute in terms of entropy, as 2 to the power
# entropy: Perplexity is a normalized way of computing probability; the higher the conditional probability of a sequence of tokens, the lower its perplexity will be.
    def perplexity(self, text):
        """
        Given list of comma-separated strings, calculates the perplexity
        of the text.
        """
        return pow(2.0, self.entropy(text))


# # Unknown Words: Back-off and Smoothing
# Because natural language is so flexible, it would be naive to expect even a very large
# corpus to contain all possible n-grams. Therefore our models must also be sufficiently
# flexible to deal with n-grams it has never seen before (e.g., “the President of California,”
# “the United States of Canada”). Symbolic models deal with this problem of covn erage through backoff—if the probability for an n-gram does not exist, the model
# looks for the probability of the (n-1)-gram (“the President of,” “the United States of ”),
# and so forth, until it gets to single tokens, or unigrams. As a rule of thumb, we should
# recursively back off to smaller n-grams until we have enough data to get a probability
# estimate.

# Since our BaseNgramModel uses maximum likelihood estimation, some (perhaps
# many) n-grams will have a zero probability of occurring, resulting in a score() of
# zero and a perplexity score of + or - infinity. The means of addressing these zeroprobability
# n-grams is to implement smoothing. Smoothing consists of donating some
# of the probability mass of frequent n-grams to unseen n-grams. The simplest type of
# smoothing is “add-one,” or Laplace, smoothing, where the new term is assigned a frequency
# of 1 and the probabilities are recomputed, but there are many other types,
# such as “add-k,” which is a generalization of Laplace smoothing.
# We can easily implement both by creating an AddKNgramModel that inherits from our
# BaseNgramModel and overrides the score method by adding the smoothing value k to
# the n-gram count and dividing by the (n-1)-gram count, normalized by the unigram
# count multiplied by k:

class AddKNgramModel(BaseNgramModel):
    """
    Provides Add-k-smoothed scores.
    """

    def __init__(self, k, *args):
        """
        Expects an input value, k, a number by which
        to increment word counts during scoring.
        """
        super(AddKNgramModel, self).__init__(*args)

        self.k = k
        self.k_norm = len(self.ngram_counter.vocabulary) * k

    def score(self, word, context):
        """
        With Add-k-smoothing, the score is normalized with
        a k value.
        """
        context = self.check_context(context)
        context_freqdist = self.ngrams[context]
        word_count = context_freqdist[word]
        context_count = context_freqdist.N()
        return (word_count + self.k) / \
               (context_count + self.k_norm)


class LaplaceNgramModel(AddKNgramModel):
    """
    Implements Laplace (add one) smoothing.
    Laplace smoothing is the base case of Add-k smoothing,
    with k set to 1.
    """
    def __init__(self, *args):
        super(LaplaceNgramModel, self).__init__(1, *args)

#
# NLTK’s probability module exposes a number of ways of calculating probability,
# including some variations on maximum likelihood and add-k smoothing, as well as:
# • UniformProbDist, which assigns equal probability to every sample in a given set,
# and a zero probability to all other samples.
# • LidstoneProbDist, which smooths sample probabilities using a real number
# gamma between 0 and 1.
# • KneserNeyProbDist, which implements a version of back-off that counts how
# likely an n-gram is provided the (n-1)-gram has been seen in training.

# Kneser–Ney smoothing considers the frequency of a unigram not by itself but in relation
# to the n-grams it completes. While some words appear in many different contexts,
# others appear frequently, but only in certain contexts; we want to treat these
# differently.
# We can create a wrapper for NLTK’s convenient implementation of Kneser–Ney
# smoothing by creating a class KneserNeyModel that inherits from BaseNgramModel
# and overrides the score method to use nltk.KneserNeyProbDist

class KneserNeyModel(BaseNgramModel):
    """
    Implements Kneser-Ney smoothing
    """
    def __init__(self, *args):
        super(KneserNeyModel, self).__init__(*args)
        self.model = nltk.KneserNeyProbDist(self.ngrams)

    def score(self, word, context):
        """
        Use KneserNeyProbDist from NLTK to get score
        """
        trigram = tuple((context[0], context[1], word))
        return self.model.prob(trigram)

    # we will create two additional methods, samples and prob, so that we can
# access the list of all trigrams with nonzero probabilities and the probability of each
# sample.

    def samples(self):
        return self.model.samples()

    def prob(self, sample):
        return self.model.prob(sample)


if __name__ == '__main__':

#     # Now, we can create a simple function that takes input text, retrieves the probability of
# each possible trigram continuation of the last two words, and appends the most likely
# next word. If fewer than two words are provided, we ask for more input.

    corpus = PickledCorpusReader('C:/Users/810004/Desktop/Html_corpus/')
    tokens = [''.join(word) for word in corpus.words()]
    vocab = Counter(tokens)
    sents = list([word[0] for word in sent] for sent in corpus.sents())
    counter = count_ngrams(3, vocab, sents)
    # For unigrams, we can get the frequency distribution using the unigrams attribute.
    print(counter.unigrams)

    knm = KneserNeyModel(counter)


    def complete(input_text):
        tokenized = nltk.word_tokenize(input_text)
        if len(tokenized) < 2:
            response = "Say more."
        else:
            completions = {}
            for sample in knm.samples():
                if (sample[0], sample[1]) == (tokenized[-2], tokenized[-1]):
                    completions[sample[2]] = knm.prob(sample)
            if len(completions) == 0:
                response = "Can we talk about something else?"
            else:
                best = max(
                    completions.keys(), key=(lambda key: completions[key])
                )
                tokenized += [best]
                response = " ".join(tokenized)

        return response


    print(complete("The President of the United"))
    print(complete("This election year will"))

#
    # counter = count_ngrams(3, vocab, sents)
    # knm = KneserNeyModel(counter)
    #
    #
    # def complete(input_text):
    #     tokenized = nltk.word_tokenize(input_text)
    #     if len(tokenized) < 2:
    #         response = "Say more."
    #     else:
    #         completions = {}
    #         for sample in knm.samples():
    #             if (sample[0], sample[1]) == (tokenized[-2], tokenized[-1]):
    #                 completions[sample[2]] = knm.prob(sample)
    #         if len(completions) == 0:
    #             response = "Can we talk about something else?"
    #         else:
    #             best = max(
    #                 completions.keys(), key=(lambda key: completions[key])
    #             )
    #             tokenized += [best]
    #             response = " ".join(tokenized)
    #
    #     return response
    #
    # print(complete("The President of the United"))
    # print(complete("This election year will"))
