import csv
import itertools
import sys

PROBS = {

    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {
        2: {True: 0.65, False: 0.35},
        1: {True: 0.56, False: 0.44},
        0: {True: 0.01, False: 0.99}
    },

    "mutation": 0.01
}


def main():

    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")

    people = load_data(sys.argv[1])

    probabilities = {
        person: {
            "gene": {2: 0, 1: 0, 0: 0},
            "trait": {True: 0, False: 0}
        }
        for person in people
    }

    names = set(people)

    for have_trait in powerset(names):

        if any(
            people[p]["trait"] is not None and
            people[p]["trait"] != (p in have_trait)
            for p in names
        ):
            continue

        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):

                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    normalize(probabilities)

    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                print(f"    {value}: {probabilities[person][field][value]:.4f}")


def load_data(filename):
    data = {}

    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            data[row["name"]] = {
                "name": row["name"],
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (
                    True if row["trait"] == "1"
                    else False if row["trait"] == "0"
                    else None
                )
            }

    return data


def powerset(s):
    s = list(s)
    return [
        set(combo)
        for r in range(len(s) + 1)
        for combo in itertools.combinations(s, r)
    ]


# ---------------------------
# CORE FUNCTIONS (FIXED)
# ---------------------------

def joint_probability(people, one_gene, two_genes, have_trait):

    probability = 1

    for person in people:

        mother = people[person]["mother"]
        father = people[person]["father"]

        if person in two_genes:
            gene_count = 2
        elif person in one_gene:
            gene_count = 1
        else:
            gene_count = 0

        # gene probability
        if mother is None and father is None:
            gene_prob = PROBS["gene"][gene_count]

        else:

            def pass_prob(parent):
                if parent in two_genes:
                    return 1 - PROBS["mutation"]
                elif parent in one_gene:
                    return 0.5
                else:
                    return PROBS["mutation"]

            mom = pass_prob(mother)
            dad = pass_prob(father)

            if gene_count == 2:
                gene_prob = mom * dad
            elif gene_count == 1:
                gene_prob = mom * (1 - dad) + (1 - mom) * dad
            else:
                gene_prob = (1 - mom) * (1 - dad)

        trait_prob = PROBS["trait"][gene_count][person in have_trait]

        probability *= gene_prob * trait_prob

    return probability


def update(probabilities, one_gene, two_genes, have_trait, p):

    for person in probabilities:

        if person in two_genes:
            g = 2
        elif person in one_gene:
            g = 1
        else:
            g = 0

        probabilities[person]["gene"][g] += p
        probabilities[person]["trait"][person in have_trait] += p


def normalize(probabilities):

    for person in probabilities:

        gene_total = sum(probabilities[person]["gene"].values())
        trait_total = sum(probabilities[person]["trait"].values())

        if gene_total > 0:
            for g in probabilities[person]["gene"]:
                probabilities[person]["gene"][g] /= gene_total

        if trait_total > 0:
            for t in probabilities[person]["trait"]:
                probabilities[person]["trait"][t] /= trait_total


if __name__ == "__main__":
    main()
