# # In order to fully leverage the data encoded in language, we must retrain our minds to
# # think of language not as intuitive and natural but as arbitrary and ambiguous. The
# # unit of text analysis is the token, a string of encoded bytes that represent text. By contrast,
# # words are symbols that are representative of meaning, and which map a textual
# # or verbal construct to a sound and sight component. Tokens are not words (though it
# # is hard for us to look at tokens and not see words).
#
# basic methodology of language-aware applications:
# clustering similar text into meaningful groups or classifying text with specific labels
#
#
# # Machine learning
# allows us to train (and retrain) statistical models on language as it changes. By building
# models of language on context-specific corpora, applications can leverage narrow
# windows of meaning to be accurate without interpretation. For example, building an
# automatic prescription application that reads medical charts requires a very different
# model than an application that summarizes and personalizes news.
#
# # the basic mechanism of a language application—the use of context to guess meaning.
# Language models also reveal the basic hypothesis behind applied machine learning
# on text: text is predictable. In fact, the mechanism used to score language models in
# an academic context, perplexity, is a measure of how predictable the text is by evaluating
# the entropy (the level of uncertainty or surprisal) of the language model’s probability
# distribution.
#
# # Consider the following partial phrases: “man’s best…” or “the witch flew on a…”.
# These low entropy phrases mean that language models would guess “friend” and
# “broomstick,” respectively, with a high likelihood (and in fact, English speakers would
# be surprised if the phrase wasn’t completed that way). On the other hand, high
# entropy phrases like “I’m going out to dinner tonight with my…” lend themselves to a
# lot of possibilities (“friend,” “mother,” and “work colleagues” could all be equally
# likely). Human listeners can use experience, imagination, and memory as well as situational
# context to fill in the blank.


# bag of words

# “bag-of-words” model. This model evaluates the frequency with which
# words co-occur with themselves and other words in a specific, limited context. Cooccurrences show which words are likely to proceed and succeed each other and by making inferences on limited pieces of text, large amounts of meaning can be captured. We can then use statistical inference methods to make predictions about word ordering.

#  n grams

# Extensions of the “bag-of-words” model consider not only single word cooccurrences,
# but also phrases that are highly correlated to indicate meaning. If “withdraw
# money at the bank” contributes a lot of information to the sense of “bank,” so
# does “fishing by the river bank.” This is called n-gram analysis, where n specifies a
# ordered sequence of either characters or words to scan on (e.g., a 3-gram is ('with
# draw', 'money', 'at') as opposed to the 5-gram ('withdraw', 'money', 'at',
# 'the', 'bank')).

#  semantics

# Semantics refer to meaning; they are deeply encoded in language and difficult to
# extract. If we think of an utterance (a simple phrase instead of a whole paragraph,
# such as “She borrowed a book from the library.”) in the abstract, we can see there is a
# template: a subject, the head verb, an object, and an instrument that relates back to
# the object (subject - predicate - object). Using such templates, ontologies can
# be constructed that specifically define the relationships between entities, but such
# work requires significant knowledge of the context and domain, and does not tend to
# scale well. Nonetheless, there is promising recent work on extracting ontologies from
# sources such as Wikipedia or DBPedia (e.g., DBPedia’s entry on libraries begins “A
# library is a collection of sources of information and similar resources, made accessible
# to a defined community for reference or borrowing.”).

Semantic analysis is not simply about understanding the meaning of text, but about
generating data structures to which logical reasoning can be applied. Text meaning
representations (or thematic meaning representations, TMRs) can be used to encode
sentences as predicate structures to which first-order logic or lambda calculus can be
applied. Other structures such as networks can be used to encode predicate interactions
of interesting features in the text. Traversal can then be used to analyze the centrality
of terms or subjects and reason about the relationships between items.
Although not necessarily a complete semantic analysis, graph analysis can produce
important insights.

A corpus can be broken down into categories of documents or individual documents.
Documents contained by a corpus can vary in size, from tweets to books, but they
contain text (and sometimes metadata) and a set of related ideas. Documents can in
turn be broken into paragraphs, units of discourse that generally each express a single
idea. Paragraphs can be further broken down into sentences, which are units of syntax;
a complete sentence is structurally sound as a specific expression. Sentences are
made up of words and punctuation, the lexical units that indicate general meaning
but are far more useful in combination. Finally, words themselves are made up of syllables,
phonemes, affixes, and characters, units that are only meaningful when combined
into words.

взять на сайт
It is very common to begin testing out a natural language model with a generic corpus.
There are, for instance, many examples and research papers that leverage readily
available datasets such as the Brown corpus, Wikipedia corpus, or Cornell movie dialogue
corpus. However, the best language models are often highly constrained and
application-specific.
Why is it that models trained in a specific field or domain of the language would perform
better than ones trained on general language? Different domains use different
language (vocabulary, acronyms, common phrases, etc.), so a corpus that is relatively
pure in domain will be able to be analyzed and modeled better than one that contains
documents from several different domains.

