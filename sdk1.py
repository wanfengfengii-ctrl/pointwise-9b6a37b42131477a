"""Offline SDK1: rational weighted paths, including convergent silent cycles."""
from dataclasses import dataclass
from fractions import Fraction
from exact import load_model, symbols_for, solve


@dataclass(frozen=True)
class Execution:
    evidence: Fraction
    visits: dict | None


@dataclass(frozen=True)
class Moments:
    evidence: Fraction
    visits: dict | None
    second: dict | None


def load(payload):
    return load_model(payload, silent=True, state_limit=12)


def _silent_matrix(model):
    """A[i][j] is the total silent weight moving from state i to state j."""
    n = len(model.states)
    matrix = [[Fraction(0) for _ in range(n)] for _ in range(n)]
    for edge in model.edges:
        if edge.symbol is None:
            matrix[edge.src][edge.dst] += edge.weight
    return matrix


def _silent_closure(system):
    """S[i][j] = ((I - Q)^-1)[i][j]: total silent weight from i to j."""
    n = len(system)
    columns = [solve(system, tuple(Fraction(row == column) for row in range(n)))
               for column in range(n)]
    return [[columns[j][i] for j in range(n)] for i in range(n)]


def evaluate(model, symbols):
    symbols = symbols_for(model, symbols)
    n, length = len(model.states), len(symbols)
    system = [[Fraction(i == j) for j in range(n)] for i in range(n)]
    for edge in model.edges:
        if edge.symbol is None:
            system[edge.src][edge.dst] -= edge.weight
    transpose = list(map(list, zip(*system)))
    # Backward suffixes include all silent moves at this input position.
    backward = [None] * (length + 1)
    backward[length] = solve(system, model.final)
    for pos in range(length - 1, -1, -1):
        rhs = [Fraction(0) for _ in range(n)]
        for edge in model.edges:
            if edge.symbol == symbols[pos]:
                rhs[edge.src] += edge.weight * backward[pos + 1][edge.dst]
        backward[pos] = solve(system, rhs)
    evidence = sum((a * b for a, b in zip(model.initial, backward[0])), Fraction(0))
    if not evidence:
        return Execution(evidence, None)
    arrival = model.initial
    numerators = [Fraction(0) for _ in range(n)]
    for pos in range(length + 1):
        forward = solve(transpose, arrival)
        for state in range(n):
            numerators[state] += forward[state] * backward[pos][state]
        if pos < length:
            arrival = [Fraction(0) for _ in range(n)]
            for edge in model.edges:
                if edge.symbol == symbols[pos]:
                    arrival[edge.dst] += forward[edge.src] * edge.weight
    return Execution(evidence, dict(zip(model.states, (v / evidence for v in numerators))))


def evaluate_moments(model, symbols):
    """Exact evidence, conditional visits and conditional second moments.

    Each visit to a state is an entry event: the initial entry or one silent
    or explicit edge traversal.  Two distinct visits can be split apart, so
    their joint weight factors through silent closures; the remaining
    same-visit pairs are exactly N_s.  Everything is exact rational and no
    path is truncated.
    """
    symbols = symbols_for(model, symbols)
    n, length = len(model.states), len(symbols)
    closure = _silent_matrix(model)
    system = [[Fraction(i == j) - closure[i][j] for j in range(n)] for i in range(n)]
    # S[i][j]: total silent weight from i to j, including the empty path.
    S = _silent_closure(system)

    # A[s][a]: total silent weight from a to an edge entering s.  A path that
    # singles out one silent entry into s contributes A[s][a] for that entry.
    incoming = [[] for _ in range(n)]
    for edge in model.edges:
        if edge.symbol is None:
            incoming[edge.dst].append((edge.src, edge.weight))
    A = [[sum((S[a][src] * weight for src, weight in incoming[s]), Fraction(0))
          for a in range(n)] for s in range(n)]

    def segment(v0, v1, v2):
        """Run all silent moves at one input position, tracking 0, 1, 2 marks."""
        # Marks made inside the silent region: the first mark is the last
        # silent entry into its state along the walk, so no silent factor
        # follows it; the second mark is any later silent entry.
        q0 = [sum((v0[a] * A[s][a] for a in range(n)), Fraction(0)) for s in range(n)]
        q1 = [[sum((v1[s][a] * A[t][a] for a in range(n)), Fraction(0))
               for t in range(n)] for s in range(n)]
        out0 = [sum((v0[a] * S[a][b] for a in range(n)), Fraction(0)) for b in range(n)]
        out1 = [[sum((v1[s][a] * S[a][b] for a in range(n)), Fraction(0))
                 + q0[s] * S[s][b] for b in range(n)] for s in range(n)]
        out2 = [[[Fraction(0) for _ in range(n)] for _ in range(n)] for _ in range(n)]
        for s in range(n):
            for t in range(n):
                rate = q1[s][t] + q0[s] * A[t][s]
                row = out2[s][t]
                previous = v2[s][t]
                for b in range(n):
                    row[b] = (sum((previous[a] * S[a][b] for a in range(n)), Fraction(0))
                              + rate * S[t][b])
        return out0, out1, out2

    symbol_edges = {}
    for symbol in model.alphabet:
        matrix = [[Fraction(0) for _ in range(n)] for _ in range(n)]
        for edge in model.edges:
            if edge.symbol == symbol:
                matrix[edge.src][edge.dst] += edge.weight
        symbol_edges[symbol] = matrix

    def explicit(v0, v1, v2, matrix):
        """Take one explicit edge; its destination is one entry event."""
        arrival = [sum((v0[b] * matrix[b][d] for b in range(n)), Fraction(0))
                   for d in range(n)]
        out1 = [[sum((v1[s][b] * matrix[b][d] for b in range(n)), Fraction(0))
                 for d in range(n)] for s in range(n)]
        for d in range(n):
            out1[d][d] += arrival[d]
        out2 = [[[sum((v2[s][t][b] * matrix[b][d] for b in range(n)), Fraction(0))
                  for d in range(n)] for t in range(n)] for s in range(n)]
        for s in range(n):
            for d in range(n):
                out2[s][d][d] += sum((v1[s][b] * matrix[b][d]
                                      for b in range(n)), Fraction(0))
        return arrival, out1, out2

    zero = Fraction(0)
    v0 = [Fraction(v) for v in model.initial]
    v1 = [[zero for _ in range(n)] for _ in range(n)]
    for state, weight in enumerate(model.initial):
        v1[state][state] += weight  # the initial entry may be the first mark
    v2 = [[[zero for _ in range(n)] for _ in range(n)] for _ in range(n)]

    for pos in range(length):
        v0, v1, v2 = segment(v0, v1, v2)
        v0, v1, v2 = explicit(v0, v1, v2, symbol_edges[symbols[pos]])
    v0, v1, v2 = segment(v0, v1, v2)

    final = [Fraction(v) for v in model.final]
    evidence = sum((v0[b] * final[b] for b in range(n)), zero)
    if not evidence:
        return Moments(evidence, None, None)
    singles = [sum((v1[s][b] * final[b] for b in range(n)), zero) for s in range(n)]
    ordered = [[sum((v2[s][t][b] * final[b] for b in range(n)), zero)
                for t in range(n)] for s in range(n)]
    visits = {state: singles[s] / evidence for s, state in enumerate(model.states)}
    second = {}
    for s, source in enumerate(model.states):
        for t, target in enumerate(model.states):
            pair = ordered[s][t] + ordered[t][s]
            if s == t:
                pair += singles[s]
            second[(source, target)] = pair / evidence
    return Moments(evidence, visits, second)
