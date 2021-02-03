#!/usr/bin/env python3
# Creates a graph of synonyms using WordNet

import re
import networkx as nx
import matplotlib.pyplot as plt

from nltk.corpus import wordnet as wn

# To implement the graph-based thesaurus just described, we will use WordNet,1 a large
# lexical database of English-language words that have been grouped into interlinked
# synsets, collections of cognitive synonyms that express distinct concepts. For our thesaurus,
# nodes will represent words from the WordNet synsets (which we can access
# via NLTK’s WordNet interface) and edges will represented synset relationships and
# interlinkages

# We will define a function, graph_synsets(), to construct the graph and add all the
# nodes and edges. Our function accepts a list of terms as well as a maximum depth,
# creates an undirected graph using NetworkX, and assigns it the name property for
# quick identification later. Then, an internal add_term_links() function adds synonyms
# by looking up the NLTK wn.synsets() function, which returns all possible definitions
# for the given word.

def graph_synsets(terms, pos=wn.NOUN, depth=2):
    """
    Create a networkx graph of the given terms to the given depth.
    """

    G = nx.Graph(
        name="WordNet Synsets Graph for {}".format(", ".join(terms)), depth=depth,
    )

    def add_term_links(G, term, current_depth):
        for syn in wn.synsets(term):
            for name in syn.lemma_names():
                G.add_edge(term, name)
                if current_depth < depth:
                    add_term_links(G, name, current_depth+1)

    for term in terms:
        add_term_links(G, term, 0)

    return G


def draw_text_graph(G):
    plt.figure(figsize=(18,12))
    pos = nx.spring_layout(G, scale=18)
    nx.draw_networkx_nodes(G, pos, node_color="white", linewidths=0, node_size=500)
    nx.draw_networkx_labels(G, pos, font_size=10)
    nx.draw_networkx_edges(G, pos)
    plt.xticks([])
    plt.yticks([])

if __name__ == '__main__':
    # import argparse
    #
    # # Create the argument parser
    # parser = argparse.ArgumentParser(
    #     description="graph synonyms for a term",
    # )
    #
    # parser.add_argument(
    #     '-d', '--depth', type=int, default=2, help="depth to extend graph",
    # )
    # parser.add_argument(
    #     '-o', '--outpath', type=str, default=None, help="file to write figure",
    # )
    # parser.add_argument(
    #     '-p', '--pos', type=str, default=wn.NOUN, help="part of speech of word(s)",
    # )
    # parser.add_argument(
    #     'words', nargs="+", help="the words to graph synonyms for",
    # )
    #
    # # parse the arguments
    # args = parser.parse_args()
    #
    # # run the graph computation
    # try:
    #     G = graph_synsets(args.words, args.pos, args.depth)
    #     draw_text_graph(G)
    #
    #     if args.outpath:
    #         plt.savefig(args.outpath)
    #     else:
    #         plt.show()
    #
    #     print(nx.info(G))
    # except Exception as e:
    #     parser.error(str(e))

    G = graph_synsets(["trinket"])
    print(nx.info(G))
    draw_text_graph(G)