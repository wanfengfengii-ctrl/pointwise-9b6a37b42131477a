"""Real legacy database, fixed-revision classification, risk, reopen, SDK2 port."""
from fractions import Fraction
import json
from pathlib import Path
import tempfile

from catalog import Catalog, result_wire
from classifier import Classifier
from samples import build_legacy, explicit_model, retry_model
import sdk2


def printable(value):
    if isinstance(value, dict):
        return {repr(k) if isinstance(k, tuple) else k: printable(v)
                for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [printable(v) for v in value]
    if isinstance(value, Fraction):
        return str(value)
    return value


def risk_view(value, saved):
    return {
        'ordinary': result_wire(value.classification),
        'second': printable(value.second),
        'mean': printable(value.mean),
        'covariance': printable(value.covariance),
        'cost_mean': str(value.cost_mean),
        'cost_variance': str(value.cost_variance),
        'saved_risk': printable(saved),
    }


def main():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'records.sqlite'
        build_legacy(path)
        costs = {('retry', 'collector'): Fraction(1),
                 ('plain', 'collector'): Fraction(1)}
        with Catalog(path) as catalog:
            classifier = Classifier(catalog)
            frozen = classifier.classify('frozen', ['x'])
            assert catalog.risk_audits('frozen') == ()
            frozen_risk = classifier.classify_risk('frozen', ['x'], costs)
            assert frozen_risk.cost_variance == Fraction(32, 9)
            first_risk = catalog.risk_audits('frozen')
            catalog.save_model('retrying', retry_model('0'))
            catalog.create_task('continued', {
                'retry': ('retrying', None, 1), 'plain': ('plain', None, 2),
            })
            continued = classifier.classify('continued', ['x'])
            continued_risk = classifier.classify_risk('continued', ['x'], costs)
            assert frozen.visits['retry']['collector'] == Fraction(4)
            assert continued.visits['retry']['collector'] == Fraction(2)
            assert catalog.risk_audits('frozen') == first_risk
        with Catalog(path) as catalog:
            assert Classifier(catalog).classify('frozen', ['x']) == frozen
            audit = catalog.audits('frozen')[-1]
            reopened = catalog.risk_audits('frozen')
            assert reopened == first_risk
            print(json.dumps({
                'frozen': result_wire(frozen), 'continued': result_wire(continued),
                'reopened_bindings': audit['bindings'],
                'durable_frozen_runs': len(catalog.audits('frozen')),
                'frozen_risk': risk_view(frozen_risk, reopened),
                'continued_risk': risk_view(continued_risk,
                                            catalog.risk_audits('continued')),
                'head_updated_but_snapshot_unchanged': True,
            }, ensure_ascii=False, indent=2))
        explicit = sdk2.evaluate(sdk2.load(explicit_model()), ['x', 'x'])
        assert explicit.evidence == Fraction(1)
        assert sum(explicit.edge_counts.values()) == Fraction(2)
        print(json.dumps({
            'sdk2_evidence': str(explicit.evidence),
            'sdk2_edge_counts': {k: str(v) for k, v in explicit.edge_counts.items()},
        }, indent=2))


if __name__ == '__main__':
    main()
