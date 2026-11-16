"""Verify entry point: full original regression plus the new risk suite/demo."""
import sys
import unittest
from fractions import Fraction

import risk_examples


def main():
    suite = unittest.defaultTestLoader.discover('tests')
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        return 1
    summary = risk_examples.risk_check_and_demo()
    assert summary['fixed_task_variance_after_head_update'] == Fraction(32, 9)
    print('risk_examples --risk: ok')
    return 0


if __name__ == '__main__':
    sys.exit(main())
