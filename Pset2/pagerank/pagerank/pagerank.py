import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")

    corpus = crawl(sys.argv[1])

    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")

    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    pages = dict()

    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue

        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename] if link in pages
        )

    return pages


# -----------------------------
# PageRank core functions
# -----------------------------

def transition_model(corpus, page, damping_factor):
    N = len(corpus)
    distribution = {}

    links = corpus[page]

    if links:
        for p in corpus:
            distribution[p] = (1 - damping_factor) / N

        for linked_page in links:
            distribution[linked_page] += damping_factor / len(links)

    else:
        for p in corpus:
            distribution[p] = 1 / N

    return distribution


def sample_pagerank(corpus, damping_factor, n):

    pages = list(corpus.keys())
    counts = {page: 0 for page in corpus}

    current = random.choice(pages)

    for _ in range(n):
        counts[current] += 1

        distribution = transition_model(corpus, current, damping_factor)

        current = random.choices(
            population=list(distribution.keys()),
            weights=list(distribution.values())
        )[0]

    for page in counts:
        counts[page] /= n

    return counts


def iterate_pagerank(corpus, damping_factor):

    N = len(corpus)
    ranks = {page: 1 / N for page in corpus}

    while True:
        new_ranks = {}

        for page in corpus:

            total = 0

            for other in corpus:
                if page in corpus[other]:

                    if len(corpus[other]) > 0:
                        total += ranks[other] / len(corpus[other])
                    else:
                        total += ranks[other] / N

            new_ranks[page] = (1 - damping_factor) / N + damping_factor * total

        # convergence check
        if all(abs(new_ranks[p] - ranks[p]) < 0.001 for p in corpus):
            return new_ranks

        ranks = new_ranks


if __name__ == "__main__":
    main()