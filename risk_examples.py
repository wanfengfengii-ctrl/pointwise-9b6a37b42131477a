"""Public hand cases and audit-denial fixture; no second-moment implementation."""
from contextlib import contextmanager
from fractions import Fraction as F
import json
from pathlib import Path
import sqlite3
import sys
import tempfile

from catalog import Catalog, result_wire
from classifier import Classifier
from samples import build_legacy, populate, retry_model
import sdk1


def frozen_costs():
    return {('retry', 'collector'): F(1), ('plain', 'collector'): F(1)}


def hand_cases():
    return {
        'retry_empty': {'symbols': (), 'evidence': F(2), 'mean': F(2), 'second': F(6)},
        'retry_x': {'symbols': ('x',), 'evidence': F(4, 3), 'mean': F(4), 'second': F(20)},
        'frozen_x': {'cost_mean': F(10, 3), 'cost_variance': F(32, 9),
                     'cross_category_covariance': F(-16, 9)},
    }


@contextmanager
def refuse_classification(catalog):
    catalog.db.execute(
        "CREATE TEMP TRIGGER risk_example_refuse BEFORE INSERT ON classification_runs "
        "BEGIN SELECT RAISE(ABORT,'classification audit refused'); END"
    )
    try:
        yield
    finally:
        catalog.db.execute('DROP TRIGGER risk_example_refuse')


def baseline_check():
    with populate(Catalog()) as catalog:
        classifier = Classifier(catalog)
        result = classifier.classify('frozen', (s for s in ['x']))
        assert result.probabilities == {'retry': F(2, 3), 'plain': F(1, 3)}
        before = catalog.audits('frozen')
        with refuse_classification(catalog):
            try:
                classifier.classify('frozen', ())
            except sqlite3.IntegrityError as exc:
                assert 'classification audit refused' in str(exc)
            else:
                raise AssertionError('audit denial was ignored')
        assert catalog.audits('frozen') == before
        return {'existing_classification': result_wire(result),
                'ordinary_audit_denial_checked': True,
                'expected_risk_hand_cases': hand_cases()}


def risk_check_and_demo():
    model = sdk1.load(retry_model())
    for key in ('retry_empty', 'retry_x'):
        case = hand_cases()[key]
        value = sdk1.evaluate_moments(model, iter(case['symbols']))
        assert value.evidence == case['evidence']
        assert value.visits == {'collector': case['mean']}
        assert value.second == {('collector', 'collector'): case['second']}
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'risk.sqlite'
        build_legacy(path)
        with Catalog(path) as catalog:
            classifier = Classifier(catalog)
            assert catalog.risk_audits('frozen') == ()
            before = len(catalog.audits('frozen'))
            costs = frozen_costs()
            value = classifier.classify_risk('frozen', (s for s in ['x']), costs)
            assert value.cost_mean == F(10, 3)
            assert value.cost_variance == F(32, 9)
            a, b = ('retry', 'collector'), ('plain', 'collector')
            assert value.mean == {a: F(8, 3), b: F(2, 3)}
            assert value.covariance == {
                (a, a): F(56, 9), (a, b): F(-16, 9),
                (b, a): F(-16, 9), (b, b): F(8, 9),
            }
            assert len(catalog.audits('frozen')) == before + 1
            saved = catalog.risk_audits('frozen')
            assert len(saved) == 1
            assert saved[0]['run_id'] == catalog.audits('frozen')[-1]['run_id']
            assert saved[0]['result']['classification'] == result_wire_values(value.classification)
            assert saved[0]['costs'] == frozen_costs()
            assert saved[0]['result']['second'] == value.second
            assert saved[0]['result']['cost_variance'] == F(32, 9)
            costs[a] = F(999)
            value.mean[a] = F(999)
            assert catalog.risk_audits('frozen') == saved
            modified = catalog.risk_audits('frozen')
            modified[0]['result']['mean'][a] = F(777)
            assert catalog.risk_audits('frozen') == saved
            catalog.save_model('retrying', retry_model('0'))
            assert catalog.risk_audits('frozen') == saved
            normal_before = catalog.audits('frozen')
            with refuse_classification(catalog):
                try:
                    classifier.classify_risk('frozen', ['x'], frozen_costs())
                except sqlite3.IntegrityError:
                    pass
                else:
                    raise AssertionError('classification audit denial was ignored')
            assert catalog.audits('frozen') == normal_before
            assert catalog.risk_audits('frozen') == saved

            def abort():
                raise RuntimeError('outer publication refused')

            try:
                with catalog.atomic(before_commit=abort):
                    classifier.classify_risk('frozen', ['x'], frozen_costs())
            except RuntimeError as exc:
                assert str(exc) == 'outer publication refused'
            else:
                raise AssertionError('publication failure was ignored')
            assert catalog.audits('frozen') == normal_before
            assert catalog.risk_audits('frozen') == saved
        with Catalog(path) as reopened:
            assert reopened.risk_audits('frozen') == saved
            again = Classifier(reopened).classify_risk('frozen', ['x'], frozen_costs())
            assert again.cost_variance == F(32, 9)
            return {'reopened_first_risk': saved, 'fixed_task_variance_after_head_update': again.cost_variance}


def result_wire_values(result):
    return {'evidence': result.evidence, 'probabilities': result.probabilities,
            'visits': result.visits}


def printable(value):
    if isinstance(value, dict):
        return {repr(k) if isinstance(k, tuple) else k: printable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [printable(v) for v in value]
    if isinstance(value, F):
        return str(value)
    return value


if __name__ == '__main__':
    if sys.argv[1:] not in ([], ['--risk']):
        raise SystemExit('usage: risk_examples.py [--risk]')
    result = risk_check_and_demo() if sys.argv[1:] else baseline_check()
    print(json.dumps(printable(result), ensure_ascii=False, indent=2))
