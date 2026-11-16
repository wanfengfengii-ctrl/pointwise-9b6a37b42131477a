"""Independent exact second-moment reference for tests; no production executor.

The reference does not use the production execution path.  It builds the
position-expanded path equations over the truncated polynomial ring in formal
variables w_s of degree at most two and solves them with a test-local Gaussian
elimination.

``U(pos, s)`` sums, over every continuation from state ``s`` at input position
``pos`` (with ``s`` itself already entered), the path weight times
``prod_v (1 + w_v)**(entries into v strictly after this node)``.  Hence

    U(pos, s) = [pos == length] * final_s
                + sum_edges weight * (1 + w_dst) * U(next pos, dst)

and the whole model's generating function is
``sum_s initial_s * (1 + w_s) * U(0, s)``.  Its constant term is the
unnormalised evidence, the w_s coefficient is the unnormalised first moment,
the w_s**2 coefficient is E[N_s (N_s - 1)] / 2 and the w_s w_t coefficient is
E[N_s N_t], all times evidence.  Truncation is only in the auxiliary polynomial
degree, never in path length; every value is exact rational.
"""
from fractions import Fraction as F


class Series:
    """Truncated polynomial in w_0..w_{n-1} of total degree at most two."""

    __slots__ = ('terms', 'n')

    def __init__(self, n, terms=None):
        self.n = n
        self.terms = terms or {}

    @staticmethod
    def constant(n, value):
        return Series(n, {} if not value else {(): F(value)})

    @staticmethod
    def variable(n, index):
        return Series(n, {(index,): F(1)})

    def __add__(self, other):
        terms = dict(self.terms)
        for key, value in other.terms.items():
            total = terms.get(key, F(0)) + value
            if total:
                terms[key] = total
            else:
                terms.pop(key, None)
        return Series(self.n, terms)

    def __neg__(self):
        return Series(self.n, {key: -value for key, value in self.terms.items()})

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        if isinstance(other, (int, F)):
            return Series(self.n, {k: v * F(other) for k, v in self.terms.items()
                                   if v * other})
        terms = {}
        for left, lvalue in self.terms.items():
            for right, rvalue in other.terms.items():
                key = tuple(sorted(left + right))
                if len(key) > 2:
                    continue
                terms[key] = terms.get(key, F(0)) + lvalue * rvalue
        return Series(self.n, {k: v for k, v in terms.items() if v})

    def inverse(self):
        a = self.terms.get((), F(0))
        if not a:
            raise ZeroDivisionError('series is not a unit')
        rest = Series(self.n, {k: v for k, v in self.terms.items() if k})
        scaled = rest * (F(1) / a)
        return (Series.constant(self.n, F(1)) - scaled + scaled * scaled) \
            * (F(1) / a)

    def coefficient(self, *key):
        return self.terms.get(tuple(sorted(key)), F(0))


def solve(matrix, rhs):
    size = len(rhs)
    rows = [list(row) + [value] for row, value in zip(matrix, rhs)]
    for col in range(size):
        pivot = next((r for r in range(col, size)
                      if rows[r][col].coefficient()), None)
        if pivot is None:
            raise ZeroDivisionError('singular reference system')
        rows[col], rows[pivot] = rows[pivot], rows[col]
        inverse = rows[col][col].inverse()
        rows[col] = [entry * inverse for entry in rows[col]]
        for row in range(size):
            if row != col and rows[row][col].terms:
                scale = rows[row][col]
                rows[row] = [left - scale * right
                             for left, right in zip(rows[row], rows[col])]
    return [row[-1] for row in rows]


def reference(payload, symbols):
    """Return (evidence, visits, second) as exact Fractions."""
    states = payload['states']
    n, length = len(states), len(symbols)
    nodes = [(pos, state) for pos in range(length + 1) for state in range(n)]
    index = {node: i for i, node in enumerate(nodes)}
    size = len(nodes)
    matrix = [[Series.constant(n, F(i == j)) for j in range(size)]
              for i in range(size)]
    rhs = [Series.constant(n, 0) for _ in range(size)]
    for pos, state in nodes:
        row = index[(pos, state)]
        if pos == length:
            rhs[row] = Series.constant(n, payload['final'][state])
        for edge in payload['edges']:
            src, dst = states.index(edge['src']), states.index(edge['dst'])
            if src != state:
                continue
            if edge['symbol'] is None:
                target = (pos, dst)
            elif pos < length and edge['symbol'] == symbols[pos]:
                target = (pos + 1, dst)
            else:
                continue
            coefficient = (Series.constant(n, 1) + Series.variable(n, dst)) \
                * F(edge['weight'])
            matrix[row][index[target]] = matrix[row][index[target]] - coefficient
    values = solve(matrix, rhs)

    initial = [F(v) for v in payload['initial']]
    total = Series.constant(n, 0)
    for state in range(n):
        if initial[state]:
            total = total + (Series.constant(n, 1) + Series.variable(n, state)) \
                * (Series.constant(n, initial[state]) * values[index[(0, state)]])
    evidence = total.coefficient()
    if not evidence:
        return evidence, None, None
    visits, second = {}, {}
    for s, source in enumerate(states):
        visits[source] = total.coefficient(s) / evidence
        for t, target in enumerate(states):
            if s == t:
                second[(source, target)] = (
                    2 * total.coefficient(s, s) + total.coefficient(s)) / evidence
            else:
                second[(source, target)] = total.coefficient(s, t) / evidence
    return evidence, visits, second
