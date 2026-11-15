from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train


class CharTokenizer:
    """Um token por caractere; 0 = BOS (como o gemma), 1 = EOS."""
    eos_token_id = 1

    def __call__(self, text, add_special_tokens=True):
        return {'input_ids': ([0] if add_special_tokens else []) + [ord(c) for c in text]}


def cve(**fields):
    row = {'language': 'C', 'cwe_id': 'CWE-787', 'cwe_description': 'Out-of-bounds\n  write.',
           'vulnerable_code': 'int f(){a[9]=1;}', 'fixed_code': 'int f(){}'}
    row.update(fields)
    return row


class EncodeGenerationTest(unittest.TestCase):
    def test_only_the_answer_and_eos_are_learned(self):
        encoded = train.encode_generation(CharTokenizer(), 'ab', 'xy', 10)
        self.assertEqual(encoded['input_ids'], [0, 97, 98, 120, 121, 1])
        self.assertEqual(encoded['labels'], [-100, -100, -100, 120, 121, 1])
        self.assertEqual(encoded['attention_mask'], [1] * 6)

    def test_example_that_does_not_fit_is_dropped_not_truncated(self):
        self.assertIsNone(train.encode_generation(CharTokenizer(), 'ab', 'xy', 5))


class CveRecordTest(unittest.TestCase):
    def test_c_record_becomes_pipeline_fields(self):
        self.assertEqual(train.cve_record(cve()),
                         {'code': 'int f(){a[9]=1;}', 'cwe_output': 'CWE-787: Out-of-bounds write.',
                          'cwe_ids': ['CWE-787'], 'fixed_code': 'int f(){}'})

    def test_cpp_is_kept(self):
        self.assertIsNotNone(train.cve_record(cve(language='C++')))

    def test_unusable_rows_are_dropped(self):
        for fields in ({'language': 'PHP'}, {'cwe_id': 'NVD-CWE-noinfo'}, {'cwe_id': 'NVD-CWE-Other'},
                       {'cwe_id': 'CWE-79, CWE-20'}, {'fixed_code': ''}, {'vulnerable_code': '  '},
                       {'cwe_description': None}, {'vulnerable_code': 'x' * 1500, 'fixed_code': 'y' * 501}):
            with self.subTest(fields=fields):
                self.assertIsNone(train.cve_record(cve(**fields)))


def bigvul(**fields):
    row = {'vul': 1, 'CWE ID': 'CWE-787', 'func_before': 'int f(){a[9]=1;}', 'func_after': 'int f(){}'}
    row.update(fields)
    return row


class BigVulRecordTest(unittest.TestCase):
    def test_vulnerable_row_uses_the_shared_cwe_description(self):
        record = train.bigvul_record(bigvul())
        self.assertEqual(record['cwe_output'], f"CWE-787: {train.pipeline.CWE_DESCRIPTIONS['CWE-787']}")
        self.assertEqual((record['code'], record['fixed_code'], record['cwe_ids']),
                         ('int f(){a[9]=1;}', 'int f(){}', ['CWE-787']))

    def test_unusable_rows_are_dropped(self):
        for fields in ({'vul': 0}, {'CWE ID': ''}, {'CWE ID': 'CWE-99999'}, {'func_after': ''},
                       {'func_after': 'int  f(){a[9]=1;}'}, {'func_before': 'x' * 1500, 'func_after': 'y' * 501}):
            with self.subTest(fields=fields):
                self.assertIsNone(train.bigvul_record(bigvul(**fields)))


class LeakageTest(unittest.TestCase):
    def test_rows_whose_code_is_in_the_test_set_are_dropped_ignoring_whitespace(self):
        rows = [{'code': 'int f() { return 1; }'}, {'code': 'int g(){}'}]
        self.assertEqual(train.drop_seen(rows, ['int f(){\n  return 1;\n}']), [{'code': 'int g(){}'}])


class BalancedTest(unittest.TestCase):
    def test_takes_half_of_each_class(self):
        rows = [{'label': 1, 'i': i} for i in range(10)] + [{'label': 0, 'i': i} for i in range(100)]
        picked = train.balanced(rows, 8, seed=1)
        self.assertEqual(sorted(r['label'] for r in picked), [0] * 4 + [1] * 4)

    def test_rare_class_limits_the_sample(self):
        rows = [{'label': 1}] * 3 + [{'label': 0}] * 100
        self.assertEqual(sorted(r['label'] for r in train.balanced(rows, 50, seed=1)), [0] * 3 + [1] * 3)


class SplitTest(unittest.TestCase):
    def test_splits_are_disjoint_and_capped(self):
        rows = [{'i': i} for i in range(100)]
        tr, va, te = train.split(rows, max_train=50, max_test=5, seed=1)
        self.assertEqual((len(tr), len(va), len(te)), (50, 5, 5))
        ids = [r['i'] for r in tr + va + te]
        self.assertEqual(len(set(ids)), 60)

    def test_same_seed_same_split(self):
        rows = [{'i': i} for i in range(100)]
        self.assertEqual(train.split(rows, 80, 10, seed=3), train.split(rows, 80, 10, seed=3))


class FakeFixModel:
    """Responde vazio ao segundo registro: o predict da pipeline transforma isso em falha."""
    def __init__(self, *args):
        self.answers = iter(['int f(){}', ''])

    def prefetch(self, texts):
        pass

    def generate(self, prompt):
        return next(self.answers)

    def close(self):
        pass


class EvaluateTest(unittest.TestCase):
    def test_failure_counts_as_wrong_answer_instead_of_leaving_the_average(self):
        test = [{'code': 'int f(){a[9]=1;}', 'cwe_output': 'CWE-787: x', 'cwe_ids': ['CWE-787'],
                 'fixed_code': 'int f(){}'}] * 2
        seen = {}
        with patch.object(train.pipeline, 'LocalModel', FakeFixModel), \
                patch.object(train.pipeline, 'codebleu_scores', lambda refs, preds, lang: seen.update(preds=preds) or {}):
            report = train.evaluate('fix', 'ckpt', test, 'cpu', 8)
        self.assertEqual((report['evaluated'], report['failed']), (2, 1))
        self.assertEqual(seen['preds'], ['int f(){}', ''])


class BaselinesTest(unittest.TestCase):
    def test_detection_baseline_says_everything_is_vulnerable(self):
        scores = train.baselines('detection', [], [{'label': 1}, {'label': 0}, {'label': 0}, {'label': 0}])
        self.assertAlmostEqual(scores['baseline_all_vulnerable']['precision'], 0.25)
        self.assertEqual(scores['baseline_all_vulnerable']['recall'], 1.0)

    def test_cwe_baseline_answers_the_most_common_training_cwe(self):
        train_rows = [{'cwe_ids': ['CWE-787']}] * 2 + [{'cwe_ids': ['CWE-79']}]
        scores = train.baselines('cwe', train_rows, [{'cwe_ids': ['CWE-787']}, {'cwe_ids': ['CWE-20']}])
        self.assertEqual(scores['baseline_majority_cwe']['cwe'], 'CWE-787')
        self.assertAlmostEqual(scores['baseline_majority_cwe']['f1_micro'], 0.5)


if __name__ == '__main__':
    unittest.main()
