import sys
from crossword import *


class CrosswordCreator():

    def __init__(self, crossword):
        self.crossword = crossword
        self.domains = {
            var: self.crossword.words.copy()
            for var in self.crossword.variables
        }

    def letter_grid(self, assignment):
        letters = [
            [None for _ in range(self.crossword.width)]
            for _ in range(self.crossword.height)
        ]

        for variable, word in assignment.items():
            direction = variable.direction
            for k in range(len(word)):
                i = variable.i + (k if direction == Variable.DOWN else 0)
                j = variable.j + (k if direction == Variable.ACROSS else 0)
                letters[i][j] = word[k]

        return letters

    def print(self, assignment):
        letters = self.letter_grid(assignment)
        for i in range(self.crossword.height):
            for j in range(self.crossword.width):
                if self.crossword.structure[i][j]:
                    print(letters[i][j] or " ", end="")
                else:
                    print("█", end="")
            print()

    def save(self, assignment, filename):
        from PIL import Image, ImageDraw, ImageFont

        cell_size = 100
        cell_border = 2
        interior_size = cell_size - 2 * cell_border

        letters = self.letter_grid(assignment)

        img = Image.new(
            "RGBA",
            (self.crossword.width * cell_size,
             self.crossword.height * cell_size),
            "black"
        )

        font = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 80)
        draw = ImageDraw.Draw(img)

        for i in range(self.crossword.height):
            for j in range(self.crossword.width):

                rect = [
                    (j * cell_size + cell_border,
                     i * cell_size + cell_border),
                    ((j + 1) * cell_size - cell_border,
                     (i + 1) * cell_size - cell_border)
                ]

                if self.crossword.structure[i][j]:
                    draw.rectangle(rect, fill="white")
                    if letters[i][j]:
                        _, _, w, h = draw.textbbox((0, 0), letters[i][j], font=font)
                        draw.text(
                            (rect[0][0] + ((interior_size - w) / 2),
                             rect[0][1] + ((interior_size - h) / 2) - 10),
                            letters[i][j],
                            fill="black",
                            font=font
                        )

        img.save(filename)

    # ---------------- CSP CORE ----------------

    def solve(self):
        self.enforce_node_consistency()
        self.ac3()
        return self.backtrack(dict())

    def enforce_node_consistency(self):
        for var in self.domains:
            self.domains[var] = {
                word for word in self.domains[var]
                if len(word) == var.length
            }

    def revise(self, x, y):
        revised = False
        overlap = self.crossword.overlaps[x, y]

        if overlap is None:
            return False

        i, j = overlap

        to_remove = set()

        for wx in self.domains[x]:
            valid = False
            for wy in self.domains[y]:
                if wx[i] == wy[j]:
                    valid = True
                    break
            if not valid:
                to_remove.add(wx)

        if to_remove:
            self.domains[x] -= to_remove
            revised = True

        return revised

    def ac3(self, arcs=None):
        from collections import deque

        queue = deque()

        if arcs is None:
            for x in self.crossword.variables:
                for y in self.crossword.neighbors(x):
                    queue.append((x, y))
        else:
            queue.extend(arcs)

        while queue:
            x, y = queue.popleft()

            if self.revise(x, y):
                if not self.domains[x]:
                    return False

                for z in self.crossword.neighbors(x):
                    if z != y:
                        queue.append((z, x))

        return True

    def assignment_complete(self, assignment):
        return set(assignment.keys()) == set(self.crossword.variables)

    def consistent(self, assignment):
        # unary constraints
        for var, word in assignment.items():
            if len(word) != var.length:
                return False

        # all-different constraint
        if len(set(assignment.values())) != len(assignment):
            return False

        # binary constraints
        for x in assignment:
            for y in assignment:
                if x == y:
                    continue
                overlap = self.crossword.overlaps[x, y]
                if overlap is None:
                    continue

                i, j = overlap
                if assignment[x][i] != assignment[y][j]:
                    return False

        return True

    def order_domain_values(self, var, assignment):
        def conflicts(value):
            count = 0
            for neighbor in self.crossword.neighbors(var):
                if neighbor in assignment:
                    continue
                overlap = self.crossword.overlaps[var, neighbor]
                if overlap is None:
                    continue
                i, j = overlap
                for w in self.domains[neighbor]:
                    if value[i] != w[j]:
                        count += 1
            return count

        return sorted(self.domains[var], key=conflicts)

    def select_unassigned_variable(self, assignment):
        unassigned = [
            v for v in self.crossword.variables
            if v not in assignment
        ]

        def score(v):
            return (len(self.domains[v]), -len(self.crossword.neighbors(v)))

        return min(unassigned, key=score)

    def backtrack(self, assignment):
        if self.assignment_complete(assignment):
            return assignment

        var = self.select_unassigned_variable(assignment)

        for value in self.order_domain_values(var, assignment):
            new_assignment = assignment.copy()
            new_assignment[var] = value

            if self.consistent(new_assignment):
                result = self.backtrack(new_assignment)
                if result is not None:
                    return result

        return None


def main():
    if len(sys.argv) not in [3, 4]:
        sys.exit("Usage: python generate.py structure words [output]")

    structure = sys.argv[1]
    words = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) == 4 else None

    crossword = Crossword(structure, words)
    creator = CrosswordCreator(crossword)
    assignment = creator.solve()

    if assignment is None:
        print("No solution.")
    else:
        creator.print(assignment)
        if output:
            creator.save(assignment, output)


if __name__ == "__main__":
    main()
