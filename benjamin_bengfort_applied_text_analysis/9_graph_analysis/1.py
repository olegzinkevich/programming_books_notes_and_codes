# For instance, given a large number of news articles, how would you build a model of
# the narratives they contain—of actions taken by key players or enacted upon others,
# of the sequence of events, of cause and effect? Using the techniques in Chapter 7, you
# could extract the entities or keyphrases or look for themes using the topic modeling
# methods described in Chapter 6. But to model information about the relationships
# between those entities, phrases, and themes, you would need a different kind of data
# structure.

headlines = ['FDA approves gene therapy',
'Gene therapy reduces tumor growth',
'FDA recalls pacemakers']

# Traditionally, phrases like these are encoded using text meaning representations
# (TMRs). TMRs take the form of ('subject', 'predicate', 'object') triples (e.g.,
# ('FDA', 'recalls', 'pacemakers')), to which first-order logic or lambda calculus
# can be applied to achieve semantic reasoning.


# Unfortunately, the construction of TMRs often requires substantial prior knowledge.
# For instance, we need to know not only that the acronym “FDA” is an actor, but that
# “recalling” is an action that can be taken by some entities against others. For most language-aware data products, building a sufficient number of TMRs to support
# meaningful semantic analysis will not be practical.

However, if we shift our thinking slightly, we might also think of this subjectpredicate-
object as a graph, where the predicates are edges between subject and object
nodes,

This will allow us to use graph traversal to answer analytical questions like “Who are
the most influential actors to an event?” or “How do relationships change over time?”
While not necessarily a complete semantic analysis, this approach can produce useful
insights

In this chapter, we will analyze text data in this way, using graph algorithms. First, we
will build a graph-based thesaurus and identify some of the most useful graph metrics.
We will then extract a social graph from our Baleen corpus, connecting actors
that appear in the same documents together and employing some simple techniques
for extracting and analyzing subgraphs. Finally, we will introduce a graph-based
approach to entity resolution called fuzzy blocking.

NetworkX and Graph-tool are the two primary Python libraries
that implement graph algorithms and the property graph model
(which we’ll explore later in this chapter). Graph-tool scales significantly
better than NetworkX, but is a C++ implementation that
must be compiled from source. For graph-based visualization, we
frequently leverage non-Python tools, such as Gephi, D3.js, and
Cytoscape.js. To keep things simple in this chapter, we will stick to
NetworkX.

One of the primary exercises in graph analytics is to determine what exactly the
nodes and edges should be. Generally, nodes represent the real-world entities we
would like to analyze, and edges represent the different types (and magnitudes) of
relationships that exist between nodes

Once a schema is determined, graph extraction is fairly straightforward. Let’s consider
a simple example that models a thesaurus as a graph. A traditional thesaurus
maps words to sets of other words that have similar meanings, connotations, and usages.
A graph-based thesaurus, which instead represents words as nodes and synonyms
as edges, could add significant value, modeling semantic similarity as a function
of the path length and weight between any two connected terms.


Creating a Graph-Based Thesaurus



