"""SDK1-backed record classification, audit consumer and joint risk audit."""
from dataclasses import dataclass
from fractions import Fraction
import sdk1
from exact import UnknownSymbol, rational


class ZeroEvidence(ValueError):
    pass


@dataclass(frozen=True)
class Classification:
    evidence: dict
    probabilities: dict
    visits: dict


@dataclass(frozen=True)
class Risk:
    classification: Classification
    second: dict
    mean: dict
    covariance: dict
    cost_mean: Fraction
    cost_variance: Fraction


def cost_values(costs, coordinates):
    """Validate a cost mapping against the complete joint coordinate list."""
    if not isinstance(costs, dict):
        raise ValueError('costs must be a dictionary')
    allowed = set(coordinates)
    result = {}
    for key, value in costs.items():
        if not isinstance(key, tuple) or len(key) != 2 or key not in allowed:
            raise ValueError('cost key is not a joint coordinate')
        try:
            result[key] = rational(value)
        except ValueError as exc:
            raise ValueError('invalid cost value') from exc
    if set(result) != allowed:
        raise ValueError('costs must cover every joint coordinate exactly')
    return result


class Classifier:
    def __init__(self, catalog):
        self.catalog = catalog

    def classify(self, task_id, symbols):
        if isinstance(symbols, (str, bytes)):
            raise UnknownSymbol('pass an iterable of symbols, not a string')
        symbols = tuple(symbols)
        with self.catalog.atomic():
            choices = self.catalog.task(task_id)
            evidence, visits, weighted, bindings = {}, {}, {}, {}
            for category, model_id, revision, prior in choices:
                model = sdk1.load(self.catalog.get_model(model_id, revision))
                execution = sdk1.evaluate(model, symbols)
                evidence[category] = execution.evidence
                visits[category] = execution.visits
                weighted[category] = prior * execution.evidence
                bindings[category] = (model_id, revision)
            total = sum(weighted.values(), Fraction(0))
            if not total:
                raise ZeroEvidence('task has zero prior-weighted evidence')
            result = Classification(evidence, {c: v / total for c, v in weighted.items()}, visits)
            self.catalog.record(task_id, symbols, bindings, result)
        return result

    def classify_risk(self, task_id, symbols, costs):
        if isinstance(symbols, (str, bytes)):
            raise UnknownSymbol('pass an iterable of symbols, not a string')
        symbols = tuple(symbols)
        with self.catalog.atomic():
            choices = self.catalog.task(task_id)
            evidence, visits, weighted, bindings, moments, states = {}, {}, {}, {}, {}, {}
            coordinates = []
            for category, model_id, revision, prior in choices:
                model = sdk1.load(self.catalog.get_model(model_id, revision))
                value = sdk1.evaluate_moments(model, symbols)
                evidence[category] = value.evidence
                visits[category] = value.visits
                moments[category] = value.second
                weighted[category] = prior * value.evidence
                bindings[category] = (model_id, revision)
                states[category] = tuple(model.states)
                coordinates.extend((category, state) for state in model.states)
            total = sum(weighted.values(), Fraction(0))
            if not total:
                raise ZeroEvidence('task has zero prior-weighted evidence')
            costs = cost_values(costs, coordinates)
            classification = Classification(
                evidence, {c: v / total for c, v in weighted.items()}, visits)

            probabilities = classification.probabilities
            mean = {coordinate: Fraction(0) for coordinate in coordinates}
            for category in probabilities:
                if not probabilities[category] or visits[category] is None:
                    continue
                for state in states[category]:
                    mean[(category, state)] = (
                        probabilities[category] * visits[category][state])

            covariance = {}
            for left in coordinates:
                for right in coordinates:
                    if left[0] != right[0] or not probabilities[left[0]]:
                        covariance[(left, right)] = (
                            -mean[left] * mean[right])
                        continue
                    category = left[0]
                    p = probabilities[category]
                    covariance[(left, right)] = (
                        p * moments[category][(left[1], right[1])]
                        - p * p * visits[category][left[1]] * visits[category][right[1]])

            cost_mean = sum((costs[c] * mean[c] for c in coordinates), Fraction(0))
            cost_variance = sum(
                (costs[left] * costs[right] * covariance[(left, right)]
                 for left in coordinates for right in coordinates), Fraction(0))

            result = Risk(classification, moments, mean, covariance,
                          cost_mean, cost_variance)
            self.catalog.record_risk(task_id, costs, symbols, bindings, result)
        return result
