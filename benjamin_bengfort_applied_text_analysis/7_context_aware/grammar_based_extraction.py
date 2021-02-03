# можно использовать для поиска ADJ+NOUN, etc в тексте

# Grammar-Based Feature Extraction
# Grammatical features such as parts-of-speech enable us to encode more contextual
# information about language
#
# To get information about the language in which the sentence is written, we need a set
# of grammatical rules that specify the components of well-structured sentences in that
# language; this is what a grammar provides. A grammar is a set of rules describing
# specifically how syntactic units (sentences, phrases, etc.) in a given language should
# be deconstructed into their constituent units. Here are some examples of these syntactic
# categories
#
# S Sentence
# NP Noun Phrase
# VP Verb Phrase
# PP Prepositional Phrase
# DT Determiner
# N Noun
# V Verb
# ADJ Adjective
# P Preposition
# TV Transitive Verb
# IV Intransitive Verb
#
# Context-Free Grammars
#
# We can use grammars to specify different rules that allow us to build up parts-ofspeech
# into phrases or chunks. A context-free grammar is a set of rules for combining
# syntactic components to form sensical strings. For instance, the noun phrase “the castle”
# has a determiner (denoted DT using the Penn Treebank tagset) and a noun (N).
# The prepositional phrase (PP) “in the castle” has a preposition (P) and a noun phrase
# (NP). The verb phrase (VP) “looks in the castle” has a verb (V) and a prepositional
# phrase (PP). The sentence (S) “Gwen looks in the castle” has a proper noun (NNP) and
# verb phrase (VP).
#
# In NLTK, nltk.grammar.CFG is an object that defines a context-free grammar, specifying
# how different syntactic components can be related. We can use CFG to parse our grammar as a string:

from nltk import CFG
import nltk

GRAMMAR = """
S -> NNP VP
VP -> V PP
PP -> P NP
NP -> DT N
NNP -> 'Gwen' | 'George'
V -> 'looks' | 'burns'
P -> 'in' | 'for'
DT -> 'the'
N -> 'castle' | 'ocean'
"""

cfg = nltk.CFG.fromstring(GRAMMAR)

print(cfg)
print(cfg.start())
print(cfg.productions())

# Syntactic Parsers

# Once we have defined a grammar, we need a mechanism to systematically search out
# the meaningful syntactic structures from our corpus; this is the role of the parser. If a
# grammar defines the search criterion for “meaningfulness” in the context of our language,
# the parser executes the search. A syntactic parser is a program that deconstructs
# sentences into a parse tree, which consists of hierarchical constituents, or
# syntactic categories.

# When a parser encounters a sentence, it checks to see if the structure of that sentence
# conforms to a known grammar. If so, it parses the sentence according to the rules of
# that grammar, producing a parse tree. Parsers are often used to identify important
# structures, like the subject and object of verbs in a sentence, or to determine which
# sequences of words in a sentence should be grouped together within each syntactic
# category.

# First, we define a GRAMMAR to identify sequences of text that match a part-of-speech
# pattern, and then instantiate an NLTK RegexpParser that uses our grammar to chunk the text into subsections:

from nltk.chunk.regexp import RegexpParser

# The GRAMMAR is a regular expression used by the NLTK RegexpParser to create trees with the label KT (key term). Our chunker will match phrases that start with an optional component composed of zero or more adjectives, followed by one or more of any type of noun and a preposition, and end with zero or more adjectives followed by one more of any type of noun. This grammar will chunk phrases like “red baseball bat” or “United States of America.”
string = 'Dusty Baker proposed a simple solution to the Washington National’s early-season bullpen troubles Monday afternoon and it had nothing to do with his maligned group of relievers'

GRAMMAR = r'KT: {(<JJ>* <NN.*>+ <IN>)? <JJ>* <NN.*>+}'
chunker = RegexpParser(GRAMMAR)

#  how to use it on string?


