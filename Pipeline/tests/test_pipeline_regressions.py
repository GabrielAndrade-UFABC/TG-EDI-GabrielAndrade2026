import contextlib
import json
import importlib.util
from pathlib import Path
import sys
import types
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pipeline


class ModelTokenLimits(unittest.TestCase):
    def test_roberta_default_leaves_room_for_prompt_and_answer(self):
        self.assertEqual(pipeline.model_token_limits(512, 'fix', 512, 512), (256, 256))

    def test_oversized_output_never_consumes_whole_context(self):
        self.assertEqual(pipeline.model_token_limits(512, 'fix', 512, 1024), (256, 256))

    def test_small_prompt_can_use_more_than_half_context_for_output(self):
        self.assertEqual(pipeline.model_token_limits(512, 'fix', 128, 512), (128, 384))

    def test_existing_smaller_output_budget_is_preserved(self):
        self.assertEqual(pipeline.model_token_limits(512, 'fix', 512, 128), (384, 128))

    def test_large_context_and_detection_are_unchanged(self):
        self.assertEqual(pipeline.model_token_limits(32768, 'fix', 512, 512), (512, 512))
        self.assertEqual(pipeline.model_token_limits(512, 'detection', 512, 512), (512, 512))

    def test_impossible_or_nonpositive_limits_are_rejected(self):
        for args in [(1, 'fix', 512, 512), (512, 'fix', 0, 512), (512, 'fix', 512, 0)]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                pipeline.model_token_limits(*args)


class OfficialLabels(unittest.TestCase):
    def test_target_overrides_legacy_vul_and_keeps_safe_cwe_out_of_references(self):
        rows = [{'target': 0, 'vul': 1, 'cwe': ['CWE-787']},
                {'target': 1, 'vul': 0, 'cwe': ['CWE-120']}]
        refs = pipeline.references_from_rows(rows)
        self.assertEqual(refs['0'], {'vulnerable': False})
        self.assertTrue(refs['1']['vulnerable'])
        self.assertEqual(refs['1']['cwe_ids'], ['CWE-120'])

    def test_hub_uses_official_target_without_deriving_vul_from_cwe(self):
        dataset = types.SimpleNamespace(column_names=['func', 'target', 'cwe'])
        module = types.SimpleNamespace(load_dataset=lambda *a, **k: {'test': dataset})
        for name in ('bstee615/diversevul', 'colin/PrimeVul'):
            with self.subTest(name=name), patch.dict(sys.modules, {'datasets': module}):
                self.assertIs(pipeline.load_hub_dataset(name), dataset)

    def test_hub_rejects_missing_official_target(self):
        dataset = types.SimpleNamespace(column_names=['func', 'cwe'])
        module = types.SimpleNamespace(load_dataset=lambda *a, **k: {'test': dataset})
        with patch.dict(sys.modules, {'datasets': module}), self.assertRaisesRegex(ValueError, 'target'):
            pipeline.load_hub_dataset('colin/PrimeVul')


class Generation(unittest.TestCase):
    def model(self):
        model = pipeline.LocalModel.__new__(pipeline.LocalModel)
        model.kind = 'fix'
        model.cache = {}
        model.torch = types.SimpleNamespace(inference_mode=contextlib.nullcontext)
        model.encode = lambda texts: texts
        return model

    def test_empty_generation_is_retried_without_returning_input_as_fallback(self):
        model = self.model()
        attempts = []
        def generate(tokens, retry=False):
            attempts.append(retry)
            return ['int f(){return 1;}' if retry else '']
        model.generate_answers = generate
        self.assertEqual(model.generate('prompt'), 'int f(){return 1;}')
        self.assertEqual(attempts, [False, True])

    def test_nonempty_copy_from_model_is_preserved(self):
        code = 'int f(){return 1;}'
        model = types.SimpleNamespace(generate=lambda prompt: code)
        self.assertEqual(pipeline.predict({'code': code}, 'fix', model, 0.5)['fixed_code'], code)

    def test_duplicate_prompt_uses_same_batch_cache_without_regeneration(self):
        model = self.model()
        model.cache = {'prompt': 'int f(){}'}
        model.generate_answers = lambda *a, **k: self.fail('cache miss')
        self.assertEqual(model.generate('prompt'), 'int f(){}')
        self.assertEqual(model.generate('prompt'), 'int f(){}')

    def test_initial_fixed_header_is_not_mistaken_for_empty_answer(self):
        model = self.model()
        model.tokenizer = types.SimpleNamespace(decode=lambda *a, **k: '\n### Fixed code:\nint f(){}\n### Buggy code:\nx')
        self.assertEqual(model.answer([]), 'int f(){}')

    def test_legacy_fix_prompt_has_no_empty_cwe_header(self):
        expected = '\n    ### Buggy code:\n    int f(){}\n\n    ### Fixed code:\n    '
        self.assertEqual(pipeline.fix_prompt('int f(){}'), expected)


class MissingPredictions(unittest.TestCase):
    def test_codebleu_empty_output_is_exactly_zero_without_dependency(self):
        with patch.dict(sys.modules, {'codebleu': None}):
            self.assertEqual(pipeline.codebleu_scores(['int f(){}'], [''], 'c')['codebleu'], 0.0)

    def test_failed_prediction_stays_in_codebleu_denominator(self):
        calls = []
        def score(**kwargs):
            calls.append(kwargs['predictions'])
            return dict.fromkeys(('codebleu', 'ngram_match_score', 'weighted_ngram_match_score',
                                  'syntax_match_score', 'dataflow_match_score'), 0.8)
        with patch.dict(sys.modules, {'codebleu': types.SimpleNamespace(calc_codebleu=score)}):
            result = pipeline.codebleu_scores(['a', 'b'], ['int f(){}', ''], 'c')
        self.assertEqual(result['codebleu'], 0.4)
        self.assertEqual(result['codebleu_generated'], 0.8)
        self.assertEqual(calls, [['int f(){}']])

    def test_error_and_skipped_outputs_are_scored_with_real_coverage(self):
        records = [dict(id='a', fixed_code='int f(){}', cwe_ids=['CWE-120'], cwe_output='CWE-120: a b c d'),
                   dict(id='b', fixed_code=None, cwe_ids=[], cwe_output=None, status='error', error={'stage': 'fix'})]
        refs = {key: dict(fixed_code='int f(){}', cwe_ids=['CWE-120'], cwe_description='a b c d') for key in ('a', 'b')}
        def scores(references, predictions, language):
            self.assertEqual(predictions, ['int f(){}', ''])
            return {'codebleu': 0.5}
        with patch.object(pipeline, 'codebleu_scores', scores):
            report = pipeline.compute_metrics(records, refs, phases=('cwe', 'fix'))
        for name in ('cwe_id', 'cwe_description', 'fix'):
            self.assertEqual((report[name]['evaluated'], report[name]['predicted'], report[name]['missing']), (2, 1, 1))
            self.assertEqual(report[name]['coverage'], 0.5)
        self.assertEqual(report['cwe_id']['f1_micro'], 0.666667)
        self.assertLess(report['cwe_description']['bleu'], 1.0)

    def test_metrics_only_reuses_saved_predictions_and_official_target_without_loading_models(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, refs, output = (root / name for name in ('saved.json', 'refs.json', 'metrics.json'))
            row = pipeline.records_from_rows([{'code': 'int f(){}'}], 'code')[0]
            pipeline.run_pipeline([row], lambda kind: types.SimpleNamespace(
                detect=lambda code: 0.1, close=lambda: None), stop_after='detection')
            pipeline.write_json(source, [row])
            pipeline.write_json(refs, [{'code': 'int f(){}', 'target': 0, 'vul': 1}])
            with patch.object(pipeline, 'LocalModel', side_effect=AssertionError('must not load weights')):
                code = pipeline.main(['--metrics-only', '--resume-from', str(source), '--references', str(refs),
                                      '--output', str(output), '--stop-after', 'detection'])
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(output.read_text())['detection']['tn'], 1)


if __name__ == '__main__':
    unittest.main()
