import nltk
import sys

TERMINALS = """
Adj -> "country" | "dreadful" | "enigmatical" | "little" | "moist" | "red"
Adv -> "down" | "here" | "never"
Conj -> "and" | "until"
Det -> "a" | "an" | "his" | "my" | "the"
N -> "armchair" | "companion" | "day" | "door" | "hand" | "he" | "himself"
N -> "holmes" | "home" | "i" | "mess" | "paint" | "palm" | "pipe" | "she"
N -> "smile" | "thursday" | "walk" | "we" | "word"
P -> "at" | "before" | "in" | "of" | "on" | "to"
V -> "arrived" | "came" | "chuckled" | "had" | "lit" | "said" | "sat"
V -> "smiled" | "tell" | "were"
"""

# ✅ FIXED GRAMMAR 
NONTERMINALS = """
S -> NP VP
S -> S Conj S

NP -> N
NP -> Det N
NP -> Det Adj N
NP -> Det Adj Adj N
NP -> NP PP

VP -> V
VP -> V NP
VP -> V NP PP
VP -> V PP
VP -> VP Conj VP

PP -> P NP
"""

grammar = nltk.CFG.fromstring(NONTERMINALS + TERMINALS)
parser = nltk.ChartParser(grammar)


def main():

    if len(sys.argv) == 2:
        with open(sys.argv[1]) as f:
            s = f.read()
    else:
        s = input("Sentence: ")

    s = preprocess(s)

    try:
        trees = list(parser.parse(s))
    except ValueError as e:
        print(e)
        return

    if not trees:
        print("Could not parse sentence.")
        return

    for tree in trees:
        tree.pretty_print()

        print("Noun Phrase Chunks")
        for np in np_chunk(tree):
            print(" ".join(np.flatten()))


# =========================
# PREPROCESS
# =========================
def preprocess(sentence):
    words = nltk.word_tokenize(sentence.lower())
    return [w for w in words if any(c.isalpha() for c in w)]


# =========================
# NP CHUNK
# =========================
def np_chunk(tree):
    chunks = []

    for subtree in tree.subtrees():
        if subtree.label() == "NP":

            # only keep top-level NP (no nested NP inside)
            if not any(s.label() == "NP" and s != subtree for s in subtree.subtrees()):
                chunks.append(subtree)

    return chunks


if __name__ == "__main__":
    main()
