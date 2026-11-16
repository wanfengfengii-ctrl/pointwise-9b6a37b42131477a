"""Regression and independent exact reconciliation for joint risk auditing.

Expected values come from ``moment_reference`` (a separate truncated
generating-function solver over rational values) and from direct hand algebra;
no expected value is produced by the production moment or risk code.
"""
import copy
from fractions import Fraction as F
import random
import sqlite3
import tempfile
import unittest
from pathlib import Path

from catalog import Catalog
from classifier import Classifier, ZeroEvidence
from exact import UnknownSymbol
import sdk1
from samples import (build_legacy, cycle_model, explicit_model, plain_model,
                     populate, retry_model)
from moment_reference import reference


def ref_moments(payload, symbols):
    return reference(payload, symbols)


def ref_risk(payloads, priors, symbols, costs):
    """Independent posterior mixture over independently computed moments."""
    categories = list(payloads)
    evidence, moments, conditionals = {}, {}, {}
    for category in categories:
        evidence[category], conditionals[category], moments[category] = (
            reference(payloads[category], symbols))
    weighted = {c: priors[c] * evidence[c] for c in categories}
    total = sum(weighted.values(), F(0))
    if not total:
        return None
    probabilities = {c: weighted[c] / total for c in categories}
    coordinates = [(c, s) for c in categories for s in payloads[c]['states']]
    mean = {coordinate: F(0) for coordinate in coordinates}
    for category in categories:
        if not probabilities[category] or conditionals[category] is None:
            continue
        for state in payloads[category]['states']:
            mean[(category, state)] = (
                probabilities[category] * conditionals[category][state])
    covariance = {}
    for left in coordinates:
        for right in coordinates:
            category = left[0]
            if category != right[0] or not probabilities[category]:
                covariance[(left, right)] = -mean[left] * mean[right]
                continue
            p = probabilities[category]
            covariance[(left, right)] = (
                p * moments[category][(left[1], right[1])]
                - p * p * conditionals[category][left[1]]
                * conditionals[category][right[1]])
    cost_mean = sum((costs[c] * mean[c] for c in coordinates), F(0))
    cost_variance = sum((costs[left] * costs[right] * covariance[(left, right)]
                         for left in coordinates for right in coordinates), F(0))
    return probabilities, moments, mean, covariance, cost_mean, cost_variance


class IndependentMomentTests(unittest.TestCase):
    def test_hand_cases_and_variance(self):
        model = sdk1.load(retry_model())
        empty = sdk1.evaluate_moments(model, ())
        self.assertEqual(empty.evidence, F(2))
        self.assertEqual(empty.visits, {'collector': F(2)})
        self.assertEqual(empty.second, {('collector', 'collector'): F(6)})
        self.assertEqual(empty.second[('collector', 'collector')]
                         - empty.visits['collector'] ** 2, F(2))
        single = sdk1.evaluate_moments(model, (s for s in ['x']))
        self.assertEqual(single.evidence, F(4, 3))
        self.assertEqual(single.visits, {'collector': F(4)})
        self.assertEqual(single.second, {('collector', 'collector'): F(20)})
        self.assertEqual(single.second[('collector', 'collector')]
                         - single.visits['collector'] ** 2, F(4))

    def test_matches_independent_reference_public_models(self):
        cases = [
            (retry_model(), [(), ('x',), ('x', 'x')]),
            (plain_model(), [(), ('x',), ('x', 'x', 'x')]),
            (cycle_model(), [(), ('x',), ('y',), ('x', 'y'), ('x', 'x')]),
            (explicit_model(), [(), ('x',), ('x', 'x'), ('x', 'x', 'x')]),
        ]
        for payload, runs in cases:
            model = sdk1.load(payload)
            for symbols in runs:
                with self.subTest(states=payload['states'], symbols=symbols):
                    actual = sdk1.evaluate_moments(model, symbols)
                    expected = reference(payload, symbols)
                    self.assertEqual(actual.evidence, expected[0])
                    self.assertEqual(actual.visits, expected[1])
                    self.assertEqual(actual.second, expected[2])
                    if not actual.evidence:
                        self.assertIsNone(actual.visits)
                        self.assertIsNone(actual.second)
                    else:
                        # Full ordered state-pair coverage in model order.
                        expected_keys = [(s, t) for s in payload['states']
                                         for t in payload['states']]
                        self.assertEqual(list(actual.second), expected_keys)

    def test_random_multistate_silent_cycles_and_edges(self):
        rng = random.Random(7311)
        for case in range(12):
            states = ['a', 'b', 'c', 'd'][:1 + case % 4]
            payload = {
                'states': states, 'alphabet': ['x', 'y'],
                'initial': [str(rng.randrange(3)) for _ in states],
                'final': [str(rng.randrange(3)) for _ in states],
                'edges': [],
            }
            for i, src in enumerate(states):
                silent_total = F(0)
                for j, dst in enumerate(states):
                    for symbol in (None, 'x', 'y'):
                        if symbol is None:
                            weight = F(rng.randrange(3), 6)
                            if silent_total + weight > F(1, 2):
                                weight = F(0)
                            else:
                                silent_total += weight
                        else:
                            weight = F(rng.randrange(3), 4)
                        payload['edges'].append({
                            'id': str((i, j, symbol)), 'src': src, 'dst': dst,
                            'symbol': symbol, 'weight': str(weight),
                        })
            model = sdk1.load(payload)
            for symbols in ((), ('x',), ('y', 'x'), ('x', 'x', 'y')):
                with self.subTest(case=case, symbols=symbols):
                    actual = sdk1.evaluate_moments(model, symbols)
                    self.assertEqual((actual.evidence, actual.visits, actual.second),
                                     reference(payload, symbols))

    def test_parallel_edges_unreachable_and_zero_weight(self):
        payload = {
            'states': ['start', 'loop', 'island'], 'alphabet': ['x'],
            'initial': ['1', '0', '0'], 'final': ['1', '1', '1'],
            'edges': [
                {'id': 's1', 'src': 'start', 'dst': 'loop', 'symbol': None, 'weight': '1/4'},
                {'id': 's2', 'src': 'start', 'dst': 'loop', 'symbol': None, 'weight': '1/4'},
                {'id': 'x1', 'src': 'loop', 'dst': 'loop', 'symbol': 'x', 'weight': '1/2'},
                {'id': 'x2', 'src': 'loop', 'dst': 'loop', 'symbol': 'x', 'weight': '1/3'},
                {'id': 'x3', 'src': 'loop', 'dst': 'start', 'symbol': 'x', 'weight': '1/3'},
                {'id': 'z', 'src': 'start', 'dst': 'island', 'symbol': None, 'weight': '0'},
            ],
        }
        model = sdk1.load(payload)
        for symbols in ((), ('x',), ('x', 'x')):
            actual = sdk1.evaluate_moments(model, symbols)
            self.assertEqual((actual.evidence, actual.visits, actual.second),
                             reference(payload, symbols))
            self.assertTrue(all(value == 0 for key, value in actual.second.items()
                                if 'island' in key))
        dead = copy.deepcopy(payload)
        dead['final'] = ['0', '0', '0']
        empty = sdk1.evaluate_moments(sdk1.load(dead), ())
        self.assertEqual(empty.evidence, F(0))
        self.assertIsNone(empty.visits)
        self.assertIsNone(empty.second)

    def test_moment_input_semantics(self):
        model = sdk1.load(cycle_model())
        for symbols in ('x', ['unknown'], [None], [1]):
            with self.assertRaises(UnknownSymbol):
                sdk1.evaluate_moments(model, symbols)
        generator = (s for s in ['x', 'y'])
        self.assertEqual(sdk1.evaluate_moments(model, generator),
                         sdk1.evaluate_moments(model, iter(('x', 'y'))))

    def test_evidence_and_visits_match_evaluate(self):
        rng = random.Random(5501)
        for case in range(8):
            states = ['a', 'b', 'c']
            payload = {
                'states': states, 'alphabet': ['x', 'y'],
                'initial': [str(rng.randrange(3)) for _ in states],
                'final': [str(rng.randrange(3)) for _ in states],
                'edges': [],
            }
            for i, src in enumerate(states):
                silent_total = F(0)
                for j, dst in enumerate(states):
                    for symbol in (None, 'x', 'y'):
                        weight = (F(rng.randrange(3), 6) if symbol is None
                                  else F(rng.randrange(3), 4))
                        if symbol is None:
                            if silent_total + weight > F(1, 2):
                                weight = F(0)
                            else:
                                silent_total += weight
                        payload['edges'].append({
                            'id': str((i, j, symbol)), 'src': src, 'dst': dst,
                            'symbol': symbol, 'weight': str(weight),
                        })
            model = sdk1.load(payload)
            for symbols in ((), ('x',), ('y', 'x'), ('x', 'x')):
                with self.subTest(case=case, symbols=symbols):
                    plain = sdk1.evaluate(model, symbols)
                    moments = sdk1.evaluate_moments(model, symbols)
                    self.assertEqual(moments.evidence, plain.evidence)
                    self.assertEqual(moments.visits, plain.visits)
                    if plain.visits is not None:
                        self.assertEqual(set(moments.second),
                                         {(s, t) for s in states for t in states})


class JointRiskTests(unittest.TestCase):
    def setUp(self):
        self.catalog = populate(Catalog())
        self.classifier = Classifier(self.catalog)

    def tearDown(self):
        self.catalog.close()

    def test_frozen_hand_values_and_linkage(self):
        costs = {('retry', 'collector'): F(1), ('plain', 'collector'): F(1)}
        before = len(self.catalog.audits('frozen'))
        value = self.classifier.classify_risk('frozen', (s for s in ['x']), costs)
        self.assertEqual(value.classification.probabilities,
                         {'retry': F(2, 3), 'plain': F(1, 3)})
        self.assertEqual(value.second, {
            'retry': {('collector', 'collector'): F(20)},
            'plain': {('collector', 'collector'): F(4)},
        })
        a, b = ('retry', 'collector'), ('plain', 'collector')
        self.assertEqual(value.mean, {a: F(8, 3), b: F(2, 3)})
        self.assertEqual(value.covariance, {
            (a, a): F(56, 9), (a, b): F(-16, 9),
            (b, a): F(-16, 9), (b, b): F(8, 9),
        })
        self.assertEqual(value.cost_mean, F(10, 3))
        self.assertEqual(value.cost_variance, F(32, 9))
        saved = self.catalog.risk_audits('frozen')
        self.assertEqual(len(saved), 1)
        self.assertEqual(saved[0]['run_id'], self.catalog.audits('frozen')[-1]['run_id'])
        self.assertEqual(len(self.catalog.audits('frozen')), before + 1)
        self.assertEqual(set(saved[0]), {'run_id', 'costs', 'result'})
        self.assertEqual(set(saved[0]['result']),
                         {'classification', 'second', 'mean', 'covariance',
                          'cost_mean', 'cost_variance'})
        self.assertEqual(saved[0]['costs'], costs)
        self.assertEqual(saved[0]['result']['second'], value.second)
        self.assertEqual(saved[0]['result']['mean'], value.mean)
        self.assertEqual(saved[0]['result']['covariance'], value.covariance)
        self.assertEqual(saved[0]['result']['cost_mean'], F(10, 3))
        self.assertEqual(saved[0]['result']['cost_variance'], F(32, 9))
        self.assertEqual(saved[0]['result']['classification'], {
            'evidence': value.classification.evidence,
            'probabilities': value.classification.probabilities,
            'visits': value.classification.visits,
        })

    def test_joint_mixture_matches_independent_composition(self):
        cases = [
            ('frozen', (), {'retry': F(1), 'plain': F(2)},
             {('retry', 'collector'): F(3, 2), ('plain', 'collector'): F(-1, 4)}),
            ('frozen', ('x',), {'retry': F(1), 'plain': F(2)},
             {('retry', 'collector'): F(1), ('plain', 'collector'): F(0)}),
            ('current', ('x',), {'retry': F(1), 'plain': F(2)},
             {('retry', 'collector'): F(-2), ('plain', 'collector'): F(5, 3)}),
        ]
        for task_id, symbols, priors, costs in cases:
            payloads = {category: self.catalog.get_model(model_id, revision)
                        for category, model_id, revision, _ in self.catalog.task(task_id)}
            expected = ref_risk(payloads, priors, symbols, costs)
            with self.subTest(task=task_id, symbols=symbols):
                value = self.classifier.classify_risk(task_id, symbols, costs)
                probabilities, moments, mean, covariance, cost_mean, cost_variance = expected
                self.assertEqual(value.classification.probabilities, probabilities)
                self.assertEqual(value.second, moments)
                self.assertEqual(value.mean, mean)
                self.assertEqual(value.covariance, covariance)
                self.assertEqual(value.cost_mean, cost_mean)
                self.assertEqual(value.cost_variance, cost_variance)

    def test_same_state_name_in_distinct_categories(self):
        other = plain_model()
        other['edges'][0]['weight'] = '1/4'
        self.catalog.save_model('other', other)
        self.catalog.create_task('same-name', {
            'one': ('plain', None, 1), 'two': ('other', None, 1),
        })
        costs = {('one', 'collector'): F(5), ('two', 'collector'): F(-3, 2)}
        value = self.classifier.classify_risk('same-name', ['x'], costs)
        payloads = {'one': plain_model(), 'two': other}
        priors = {'one': F(1), 'two': F(1)}
        expected = ref_risk(payloads, priors, ('x',), costs)
        self.assertEqual(value.second, expected[1])
        self.assertEqual(value.mean, expected[2])
        self.assertEqual(value.covariance, expected[3])
        self.assertEqual(value.cost_mean, expected[4])
        self.assertEqual(value.cost_variance, expected[5])
        self.assertIn(('one', 'collector'), value.mean)
        self.assertIn(('two', 'collector'), value.mean)
        self.assertNotEqual(value.mean[('one', 'collector')],
                            value.mean[('two', 'collector')])

    def test_zero_prior_and_zero_evidence_categories(self):
        dead = plain_model()
        dead['final'] = ['0']
        self.catalog.save_model('dead', dead)
        self.catalog.create_task('mixed', {
            'dead': ('dead', None, 1), 'live': ('plain', None, 1),
            'ignored': ('retrying', 1, 0),
        })
        costs = {('dead', 'collector'): F(2), ('live', 'collector'): F(3),
                 ('ignored', 'collector'): F(7)}
        value = self.classifier.classify_risk('mixed', (), costs)
        self.assertEqual(value.classification.probabilities,
                         {'dead': F(0), 'live': F(1), 'ignored': F(0)})
        self.assertIsNone(value.second['dead'])
        self.assertEqual(value.second['ignored'], {('collector', 'collector'): F(6)})
        self.assertEqual(value.classification.visits['ignored'], {'collector': F(2)})
        self.assertEqual(value.mean[('dead', 'collector')], F(0))
        self.assertEqual(value.mean[('ignored', 'collector')], F(0))
        self.assertEqual(value.mean[('live', 'collector')], F(1))
        for left in value.covariance:
            for right in value.covariance:
                if 'dead' in left or 'dead' in right or 'ignored' in left or 'ignored' in right:
                    self.assertEqual(value.covariance[(left, right)], F(0))
        self.assertEqual(value.cost_mean, F(3))
        self.assertEqual(value.cost_variance, F(0))
        self.assertEqual(len(self.catalog.risk_audits('mixed')), 1)

    def test_zero_weighted_evidence_adds_no_audit(self):
        dead = plain_model()
        dead['final'] = ['0']
        self.catalog.save_model('dead', dead)
        self.catalog.create_task('impossible', {'dead': ('dead', None, 1)})
        with self.assertRaises(ZeroEvidence):
            self.classifier.classify_risk('impossible', (),
                                          {('dead', 'collector'): F(1)})
        self.assertEqual(self.catalog.audits('impossible'), ())
        self.assertEqual(self.catalog.risk_audits('impossible'), ())

    def test_invalid_costs_leave_no_audit(self):
        valid = {('retry', 'collector'): F(1), ('plain', 'collector'): F(1)}
        cases = [
            {},
            {('retry', 'collector'): F(1)},
            {**valid, ('plain', 'other'): F(1)},
            {**valid, ('missing', 'collector'): F(1)},
            {**valid, 'retry': F(1)},
            {**valid, (('retry', 'collector'),): F(1)},
            {('retry', 'collector', 'extra'): F(1), ('plain', 'collector'): F(1)},
            {('retry', 'collector'): 0.5, ('plain', 'collector'): F(1)},
            {('retry', 'collector'): True, ('plain', 'collector'): F(1)},
            {('retry', 'collector'): 'not-a-number', ('plain', 'collector'): F(1)},
            {('retry', 'collector'): None, ('plain', 'collector'): F(1)},
            [F(1), F(1)],
        ]
        normal = len(self.catalog.audits('frozen'))
        for costs in cases:
            with self.subTest(costs=costs), self.assertRaises(ValueError):
                self.classifier.classify_risk('frozen', ['x'], costs)
        self.assertEqual(len(self.catalog.audits('frozen')), normal)
        self.assertEqual(self.catalog.risk_audits('frozen'), ())
        self.assertFalse(self.catalog.db.in_transaction)
        self.classifier.classify_risk('frozen', ['x'], valid)
        self.assertEqual(len(self.catalog.audits('frozen')), normal + 1)

    def test_unknown_symbol_and_task_semantics(self):
        costs = {('retry', 'collector'): F(1), ('plain', 'collector'): F(1)}
        normal = len(self.catalog.audits('frozen'))
        with self.assertRaises(UnknownSymbol):
            self.classifier.classify_risk('frozen', ['bad'], costs)
        with self.assertRaises(UnknownSymbol):
            self.classifier.classify_risk('frozen', 'x', costs)
        with self.assertRaises(KeyError):
            self.classifier.classify_risk('missing-task', ['x'], costs)
        self.assertEqual(len(self.catalog.audits('frozen')), normal)
        self.assertEqual(self.catalog.risk_audits('frozen'), ())
        with self.assertRaises(KeyError):
            self.catalog.risk_audits('missing-task')

    def test_fixed_revision_and_value_isolation(self):
        costs = {('retry', 'collector'): F(1), ('plain', 'collector'): F(1)}
        first = self.classifier.classify_risk('frozen', ['x'], costs)
        saved = self.catalog.risk_audits('frozen')
        costs[('retry', 'collector')] = F(999)
        first.mean[('retry', 'collector')] = F(999)
        first.second['retry'][('collector', 'collector')] = F(999)
        first.covariance[(('retry', 'collector'), ('retry', 'collector'))] = F(999)
        first.classification.visits['retry']['collector'] = F(999)
        self.assertEqual(self.catalog.risk_audits('frozen'), saved)
        first_read = self.catalog.risk_audits('frozen')
        first_read[0]['result']['mean'][('retry', 'collector')] = F(777)
        first_read[0]['costs'][('retry', 'collector')] = F(777)
        first_read[0]['result']['second']['retry'][('collector', 'collector')] = F(777)
        self.assertEqual(self.catalog.risk_audits('frozen'), saved)
        revision = self.catalog.save_model('retrying', retry_model('0'))
        self.assertEqual(revision, 3)
        self.assertEqual(self.catalog.task('frozen')[0], ('retry', 'retrying', 1, F(1)))
        self.assertEqual(self.catalog.risk_audits('frozen'), saved)
        again = self.classifier.classify_risk('frozen', ['x'],
                                              {('retry', 'collector'): F(1),
                                               ('plain', 'collector'): F(1)})
        self.assertEqual(again.cost_variance, F(32, 9))
        self.assertEqual(len(self.catalog.risk_audits('frozen')), 2)

    def test_ordinary_classify_adds_no_risk(self):
        self.classifier.classify('frozen', ['x'])
        self.assertEqual(self.catalog.risk_audits('frozen'), ())

    def test_moment_visits_agree_with_plain_classify(self):
        costs = {('retry', 'collector'): F(1), ('plain', 'collector'): F(1)}
        plain = self.classifier.classify('frozen', ['x', 'x'])
        risky = self.classifier.classify_risk('frozen', ['x', 'x'], costs)
        self.assertEqual(risky.classification.evidence, plain.evidence)
        self.assertEqual(risky.classification.probabilities, plain.probabilities)
        self.assertEqual(risky.classification.visits, plain.visits)


class PublicationFailureTests(unittest.TestCase):
    def setUp(self):
        self.catalog = populate(Catalog())
        self.classifier = Classifier(self.catalog)
        self.costs = {('retry', 'collector'): F(1), ('plain', 'collector'): F(1)}

    def tearDown(self):
        self.catalog.close()

    def test_ordinary_audit_refused_leaves_no_risk(self):
        self.classifier.classify_risk('frozen', ['x'], self.costs)
        risk_before = self.catalog.risk_audits('frozen')
        normal_before = self.catalog.audits('frozen')
        self.catalog.db.execute(
            "CREATE TEMP TRIGGER deny_normal BEFORE INSERT ON classification_runs "
            "BEGIN SELECT RAISE(ABORT,'normal refused'); END"
        )
        try:
            with self.assertRaisesRegex(sqlite3.IntegrityError, 'normal refused'):
                self.classifier.classify_risk('frozen', ['x'], self.costs)
        finally:
            self.catalog.db.execute('DROP TRIGGER deny_normal')
        self.assertEqual(self.catalog.audits('frozen'), normal_before)
        self.assertEqual(self.catalog.risk_audits('frozen'), risk_before)
        self.assertFalse(self.catalog.db.in_transaction)
        self.classifier.classify_risk('frozen', ['x'], self.costs)
        self.assertEqual(len(self.catalog.risk_audits('frozen')), len(risk_before) + 1)

    def test_risk_audit_refused_leaves_no_normal_run(self):
        self.classifier.classify_risk('frozen', ['x'], self.costs)
        risk_before = self.catalog.risk_audits('frozen')
        normal_before = self.catalog.audits('frozen')
        self.catalog.db.execute(
            "CREATE TEMP TRIGGER deny_risk BEFORE INSERT ON risk_runs "
            "BEGIN SELECT RAISE(ABORT,'risk refused'); END"
        )
        try:
            with self.assertRaisesRegex(sqlite3.IntegrityError, 'risk refused'):
                self.classifier.classify_risk('frozen', ['x'], self.costs)
        finally:
            self.catalog.db.execute('DROP TRIGGER deny_risk')
        self.assertEqual(self.catalog.audits('frozen'), normal_before)
        self.assertEqual(self.catalog.risk_audits('frozen'), risk_before)
        self.assertFalse(self.catalog.db.in_transaction)
        self.classifier.classify_risk('frozen', ['x'], self.costs)
        self.assertEqual(len(self.catalog.audits('frozen')), len(normal_before) + 1)
        self.assertEqual(len(self.catalog.risk_audits('frozen')), len(risk_before) + 1)

    def test_outer_commit_failure_leaves_nothing(self):
        risk_before = self.catalog.risk_audits('frozen')
        normal_before = self.catalog.audits('frozen')
        before = tuple(self.catalog.db.iterdump())

        def abort():
            raise RuntimeError('outer publication refused')

        with self.assertRaisesRegex(RuntimeError, 'outer publication refused'):
            with self.catalog.atomic(before_commit=abort):
                self.classifier.classify_risk('frozen', ['x'], self.costs)
        self.assertEqual(tuple(self.catalog.db.iterdump()), before)
        self.assertEqual(self.catalog.audits('frozen'), normal_before)
        self.assertEqual(self.catalog.risk_audits('frozen'), risk_before)


class LegacyAndDiskTests(unittest.TestCase):
    def test_legacy_has_no_backfilled_risk(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'legacy.sqlite'
            build_legacy(path)
            with Catalog(path) as catalog:
                self.assertEqual(catalog.db.execute(
                    "SELECT value FROM meta WHERE key='schema_version'").fetchone(), ('1',))
                self.assertEqual(catalog.risk_audits('frozen'), ())
                self.assertEqual(catalog.risk_audits('current'), ())
                self.assertEqual(len(catalog.audits('frozen')), 2)
                self.assertEqual(len(catalog.audits('current')), 1)
                present = catalog.db.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='risk_runs'"
                ).fetchone()
                self.assertIsNone(present)
            with Catalog(path) as catalog:
                self.assertEqual(catalog.risk_audits('frozen'), ())
                self.assertEqual(len(catalog.audits('frozen')), 2)

    def test_file_reopen_head_update_and_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'risk.sqlite'
            build_legacy(path)
            costs = {('retry', 'collector'): F(1), ('plain', 'collector'): F(1)}
            with Catalog(path) as catalog:
                value = Classifier(catalog).classify_risk('frozen', ['x'], costs)
                saved = catalog.risk_audits('frozen')
                run_id = saved[0]['run_id']
                catalog.save_model('retrying', retry_model('0'))
                self.assertEqual(catalog.risk_audits('frozen'), saved)
            with Catalog(path) as reopened:
                self.assertEqual(reopened.risk_audits('frozen'), saved)
                audit = reopened.audits('frozen')[-1]
                self.assertEqual(audit['run_id'], run_id)
                self.assertEqual(audit['bindings']['retry'], ('retrying', 1))
                self.assertEqual(reopened.task('frozen')[0][2], 1)
                again = Classifier(reopened).classify_risk('frozen', ['x'], costs)
                self.assertEqual(again.cost_variance, value.cost_variance)
                self.assertEqual(reopened.db.execute(
                    'PRAGMA foreign_key_check').fetchall(), [])
                self.assertEqual(len(reopened.risk_audits('frozen')), 2)


if __name__ == '__main__':
    unittest.main()
