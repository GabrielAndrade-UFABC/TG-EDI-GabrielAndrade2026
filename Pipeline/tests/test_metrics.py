"""Métricas do wrapper: F1 (detecção e ID da CWE), BLEU (descrição da CWE) e CodeBLEU (correção).

Os valores de referência foram conferidos contra sklearn (F1) e sacreBLEU (BLEU) em centenas de
casos aleatórios; aqui ficam valores fixos para o teste não depender dessas bibliotecas.
"""
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pipeline

HAS_CODEBLEU = importlib.util.find_spec('codebleu') is not None
CWE_TEXT = 'improper restriction of operations within the bounds of a memory buffer'


class MetricMath(unittest.TestCase):
    def test_detection_f1_uses_vulnerable_as_positive_class(self):
        scores = pipeline.detection_scores([(True, True), (True, True), (True, False), (False, False), (False, True)])
        self.assertEqual((scores['tp'], scores['fp'], scores['fn'], scores['tn']), (2, 1, 1, 1))
        self.assertAlmostEqual(scores['precision'], 2 / 3)
        self.assertAlmostEqual(scores['recall'], 2 / 3)
        self.assertAlmostEqual(scores['f1'], 2 / 3)
        self.assertAlmostEqual(scores['accuracy'], 3 / 5)

    def test_detection_f1_is_zero_without_true_positives(self):
        self.assertEqual(pipeline.detection_scores([(True, False), (False, False)])['f1'], 0.0)

    def test_cwe_id_f1_macro_and_micro(self):
        pairs = [({'CWE-120'}, {'CWE-120'}), ({'CWE-787'}, {'CWE-120'}), ({'CWE-79'}, set())]
        scores = pipeline.cwe_id_scores(pairs)
        # CWE-120: tp=1 fp=1 -> F1 2/3; CWE-787 e CWE-79: só fn -> 0.
        self.assertAlmostEqual(scores['f1_macro'], (2 / 3) / 3)
        self.assertAlmostEqual(scores['f1_micro'], 0.4)
        self.assertEqual(scores['classes'], 3)

    def test_bleu_matches_sacrebleu_reference_value(self):
        refs = ['improper restriction of operations within the bounds of a memory buffer',
                'improper neutralization of special elements used in an sql command']
        hyps = ['improper restriction of operations within a memory buffer',
                'improper neutralization of special elements in an sql command']
        self.assertAlmostEqual(pipeline.corpus_bleu(refs, hyps)['bleu'], 0.5711434564419329, places=12)

    def test_bleu_edges(self):
        self.assertEqual(pipeline.corpus_bleu([CWE_TEXT], [CWE_TEXT])['bleu'], 1.0)
        self.assertEqual(pipeline.corpus_bleu([CWE_TEXT], [''])['bleu'], 0.0)
        self.assertEqual(pipeline.corpus_bleu([], [])['bleu'], 0.0)
        # Sem 4-gramas em comum não há suavização: BLEU zera, como no `evaluate`.
        self.assertEqual(pipeline.corpus_bleu(['a b c d'], ['d c b a'])['bleu'], 0.0)

    def test_bleu_brevity_penalty_for_short_hypothesis(self):
        result = pipeline.corpus_bleu(['a b c d e f'], ['a b c d'])
        self.assertAlmostEqual(result['precisions'][0], 1.0)
        self.assertLess(result['brevity_penalty'], 1.0)


class ReferenceParsing(unittest.TestCase):
    def test_cwe_ids_are_normalized(self):
        self.assertEqual(pipeline.parse_cwe_ids('CWE-079'), ['CWE-79'])
        self.assertEqual(pipeline.parse_cwe_ids(' cwe_120 , CWE-787 '), ['CWE-120', 'CWE-787'])
        self.assertEqual(pipeline.parse_cwe_ids(['CWE-20', 'CWE-20']), ['CWE-20'])
        self.assertEqual(pipeline.parse_cwe_ids(79), ['CWE-79'])
        self.assertEqual(pipeline.parse_cwe_ids('[]'), [])
        self.assertEqual(pipeline.parse_cwe_ids('NVD-CWE-Other'), [])
        self.assertEqual(pipeline.parse_cwe_ids(None), [])

    def test_description_drops_the_cwe_id_on_both_sides(self):
        self.assertEqual(pipeline.description_text('CWE-120: Buffer Copy  without Checking'), 'Buffer Copy without Checking')
        self.assertEqual(pipeline.description_text('Buffer Copy'), 'Buffer Copy')
        self.assertEqual(pipeline.description_text('CWE-79'), '')

    def test_vulnerable_label_accepts_common_forms_and_rejects_noise(self):
        for value in (1, True, '1', 'true', 'Vulnerable'):
            self.assertIs(pipeline.parse_expected_vulnerable(value), True)
        for value in (0, False, '0', 'false', 'safe'):
            self.assertIs(pipeline.parse_expected_vulnerable(value), False)
        with self.assertRaises(ValueError):
            pipeline.parse_expected_vulnerable('talvez')

    def test_read_references_from_dataset_columns(self):
        rows = [
            {'id': 'a', 'code': 'x', 'vul': 1, 'cwe_id': 'CWE-120', 'cwe_description': 'CWE-120: Overflow',
             'fixed_code': 'int f(void) { return 0; }'},
            {'id': 'b', 'code': 'y'},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'in.json'
            path.write_text(json.dumps(rows))
            refs = pipeline.read_references(path)
        self.assertEqual(refs, {'a': {'vulnerable': True, 'cwe_ids': ['CWE-120'], 'cwe_description': 'Overflow',
                                      'fixed_code': 'int f(void) { return 0; }'}})

    def test_safe_rows_keep_only_the_vulnerability_label(self):
        rows = [{'id': 'a', 'code': 'x', 'vul': 0, 'CWE ID': 'CWE-119', 'func_after': 'x'},
                {'id': 'b', 'code': 'y', 'vul': 1, 'CWE ID': 'CWE-119', 'func_after': 'z'}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'in.json'
            path.write_text(json.dumps(rows))
            with patch.dict(pipeline.CWE_DESCRIPTIONS, {}, clear=True):
                refs = pipeline.read_references(path)
        self.assertEqual(refs, {'a': {'vulnerable': False},
                                'b': {'vulnerable': True, 'cwe_ids': ['CWE-119'], 'fixed_code': 'z'}})

    def read_with_descriptions(self, rows):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'in.json'
            path.write_text(json.dumps(rows))
            with patch.dict(pipeline.CWE_DESCRIPTIONS, {'CWE-119': 'Buffer bounds.', 'CWE-20': 'Bad input.'}, clear=True):
                return pipeline.read_references(path)

    def test_missing_description_is_filled_from_the_cwe_id(self):
        refs = self.read_with_descriptions([{'id': 'a', 'vul': 1, 'CWE ID': 'CWE-119'},
                                            {'id': 'b', 'vul': 1, 'cwe_id': ['CWE-20', 'CWE-119']},
                                            {'id': 'c', 'vul': 1, 'CWE ID': 'CWE-999'},
                                            {'id': 'd', 'vul': 0, 'CWE ID': 'CWE-119'}])
        self.assertEqual(refs['a']['cwe_description'], 'Buffer bounds.')
        self.assertEqual(refs['b']['cwe_description'], 'Bad input., Buffer bounds.')
        self.assertNotIn('cwe_description', refs['c'])
        self.assertEqual(refs['d'], {'vulnerable': False})

    def test_description_given_in_the_input_wins(self):
        refs = self.read_with_descriptions([{'id': 'a', 'CWE ID': 'CWE-119', 'cwe_description': 'Own text.'}])
        self.assertEqual(refs['a']['cwe_description'], 'Own text.')

    def test_shipped_descriptions_cover_common_c_cwes(self):
        for cwe in ('CWE-119', 'CWE-20', 'CWE-787', 'CWE-476', 'CWE-416'):
            self.assertTrue(pipeline.CWE_DESCRIPTIONS.get(cwe), cwe)

    def test_cwe_answer_is_cut_at_the_first_line(self):
        class Model(FakeModel):
            def generate(self, prompt):
                return 'CWE-476: NULL dereference.\nif (x) { CWE-79 }'
        result = pipeline.predict({'code': 'x'}, 'cwe', Model('cwe'), 0.5)
        self.assertEqual(result['cwe_output'], 'CWE-476: NULL dereference.')
        self.assertEqual(result['cwe_ids'], ['CWE-476'])

    def test_cwe_answer_uses_the_completion_and_recovers_the_graphcodebert_prefix(self):
        class Model(FakeModel):
            def generate(self, prompt):
                return '/* CWE-79 */\n### Identify the CWE:\nWE-0416: Use After Free.\n### Vulnerable code:'
        result = pipeline.predict({'code': 'x'}, 'cwe', Model('cwe'), 0.5)
        self.assertEqual(result['cwe_ids'], ['CWE-416'])
        self.assertEqual(result['cwe_output'], 'CWE-416: Use After Free.')
        self.assertEqual(pipeline.description_text(result['cwe_output']), 'Use After Free.')

    def test_cwe_cleanup_keeps_a_blank_completion_unidentified(self):
        class Model(FakeModel):
            def generate(self, prompt):
                return '/* CWE-79 */\n### Identify the CWE:\n  '
        result = pipeline.predict({'code': 'x'}, 'cwe', Model('cwe'), 0.5)
        self.assertEqual(result['status'], 'cwe_unidentified')
        self.assertEqual(result['cwe_ids'], [])
        self.assertEqual(result['cwe_output'], '')

    def test_references_that_look_like_pipeline_output_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'out.json'
            path.write_text(json.dumps([{'id': 'a', 'code': 'x', 'vulnerable': True,
                                         'vulnerability_probability': 0.9, 'phase_results': {}}]))
            with self.assertRaises(ValueError):
                pipeline.read_references(path)

    def test_bad_label_names_the_record_and_column(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'in.json'
            path.write_text(json.dumps([{'id': 'z', 'code': 'x', 'label': 'talvez'}]))
            with self.assertRaisesRegex(ValueError, "z.*label"):
                pipeline.read_references(path)

    def test_plain_code_file_has_no_references(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'f.c'
            path.write_text('int main(void) { return 0; }')
            self.assertEqual(pipeline.read_references(path), {})


class FakeModel:
    def __init__(self, kind):
        self.kind = kind

    def detect(self, code):
        return 0.9 if 'strcpy' in code else 0.1

    def generate(self, prompt):
        return f'CWE-120: {CWE_TEXT}' if self.kind == 'cwe' else 'int f(void) { return 0; }'

    def close(self):
        pass


def make_records():
    return [
        {'id': 'r1', 'code': 'strcpy(a, b);', 'vul': 1, 'cwe_id': 'CWE-120',
         'cwe_description': f'CWE-120: {CWE_TEXT}', 'fixed_code': 'int f(void) { return 0; }'},
        {'id': 'r2', 'code': 'int sum(int a, int b) { return a + b; }', 'vul': 0},
        {'id': 'r3', 'code': 'strcpy(c, d);', 'vul': 1, 'cwe_id': 'CWE-787',
         'cwe_description': 'Out-of-bounds Write to a buffer of fixed size', 'fixed_code': 'int g(void) { return 1; }'},
        # Vulnerável de verdade, mas o modelo de Detection diz que é seguro (falso negativo).
        {'id': 'r4', 'code': 'int leak(void) { return 0; }', 'vul': 1, 'cwe_id': 'CWE-79',
         'cwe_description': 'Cross-site Scripting'},
    ]


class MetricsEndToEnd(unittest.TestCase):
    def setUp(self):
        for manager in (redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO())):
            manager.__enter__()
            self.addCleanup(manager.__exit__, None, None, None)

    def run_main(self, directory, extra=()):
        source = Path(directory) / 'input.json'
        source.write_text(json.dumps(make_records()))
        output = Path(directory) / 'out.json'
        with patch.object(pipeline, 'LocalModel', lambda path, kind, *args: FakeModel(kind)):
            code = pipeline.main(['--input', str(source), '--output', str(output), *extra])
        return code, next(Path(directory).glob('out_phases_*'))

    def test_all_four_metrics_are_computed_and_saved(self):
        with tempfile.TemporaryDirectory() as directory:
            code, phases = self.run_main(directory)
            self.assertEqual(code, 0)
            report = json.loads((phases / 'metrics.json').read_text())
            summary = json.loads((phases / 'summary.json').read_text())

        # Detection: r1 e r3 acertam; r4 é falso negativo; r2 é negativo verdadeiro.
        self.assertEqual(report['detection']['evaluated'], 4)
        self.assertAlmostEqual(report['detection']['f1'], 0.8)
        self.assertAlmostEqual(report['detection']['precision'], 1.0)
        self.assertAlmostEqual(report['detection']['recall'], 2 / 3, places=5)

        # CWE (ID): r1 acerta, r3 erra (CWE-120 no lugar de CWE-787), r4 foi pulado -> conjunto vazio.
        cwe = report['cwe_id']
        self.assertEqual((cwe['expected'], cwe['evaluated']), (3, 3))
        self.assertAlmostEqual(cwe['f1_micro'], 0.4)
        self.assertAlmostEqual(cwe['f1_macro'], (2 / 3) / 3, places=5)

        # CWE (descrição): r1 igual à esperada; r3 diferente; r4 não teve resposta de CWE.
        desc = report['cwe_description']
        self.assertEqual((desc['expected'], desc['evaluated'], desc['missing']), (3, 3, 1))
        self.assertEqual(desc['predicted'], 2)
        expected_bleu = pipeline.corpus_bleu([CWE_TEXT, 'Out-of-bounds Write to a buffer of fixed size',
                                              'Cross-site Scripting'], [CWE_TEXT, CWE_TEXT, ''])['bleu']
        self.assertAlmostEqual(desc['bleu'], round(expected_bleu, 6), places=6)

        # Fix: duas correções esperadas e geradas.
        fix = report['fix']
        self.assertEqual(fix['expected'], 2)
        if HAS_CODEBLEU:
            self.assertEqual(fix['evaluated'], 2)
            self.assertTrue(fix['available'])
            self.assertGreater(fix['codebleu'], 0.0)
        else:
            self.assertEqual((fix['evaluated'], fix['missing']), (0, 2))
            self.assertFalse(fix['available'])
            self.assertIn('codebleu', fix['reason'])

        self.assertEqual(summary['metrics'], report)

    def test_stop_after_detection_only_reports_detection(self):
        with tempfile.TemporaryDirectory() as directory:
            _, phases = self.run_main(directory, ['--stop-after', 'detection'])
            report = json.loads((phases / 'metrics.json').read_text())
        self.assertIn('detection', report)
        self.assertNotIn('cwe_id', report)
        self.assertNotIn('fix', report)

    def test_missing_codebleu_does_not_break_the_run_or_other_metrics(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {'codebleu': None}):
            code, phases = self.run_main(directory)
            report = json.loads((phases / 'metrics.json').read_text())
        self.assertEqual(code, 0)
        self.assertFalse(report['fix']['available'])
        self.assertIn('codebleu', report['fix']['reason'])
        self.assertTrue(report['detection']['available'])
        self.assertTrue(report['cwe_description']['available'])

    def test_fix_skipped_by_detection_counts_as_empty_fix(self):
        record = {'id': 'r', 'code': 'x', 'vulnerable': False, 'cwe_ids': [], 'cwe_output': None,
                  'fixed_code': None, 'status': 'not_vulnerable', 'error': None}
        seen = {}
        with patch.object(pipeline, 'codebleu_scores', lambda refs, preds, lang: seen.update(preds=preds) or {}):
            report = pipeline.compute_metrics([record], {'r': {'fixed_code': 'int f(){}'}}, ('fix',))
        self.assertEqual(seen['preds'], [''])
        self.assertEqual(report['fix']['evaluated'], 1)

    def test_no_metrics_flag_and_unlabeled_input_write_no_report(self):
        with tempfile.TemporaryDirectory() as directory:
            _, phases = self.run_main(directory, ['--no-metrics'])
            self.assertFalse((phases / 'metrics.json').exists())
            self.assertNotIn('metrics', json.loads((phases / 'summary.json').read_text()))

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'input.json'
            source.write_text(json.dumps([{'id': 'a', 'code': 'strcpy(a, b);'}]))
            output = Path(directory) / 'out.json'
            with patch.object(pipeline, 'LocalModel', lambda path, kind, *args: FakeModel(kind)):
                self.assertEqual(pipeline.main(['--input', str(source), '--output', str(output)]), 0)
            self.assertFalse(list(Path(directory).glob('out_phases_*/metrics.json')))

    def test_resume_uses_references_file(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'input.json'
            source.write_text(json.dumps(make_records()))
            first = Path(directory) / 'first.json'
            with patch.object(pipeline, 'LocalModel', lambda path, kind, *args: FakeModel(kind)):
                pipeline.main(['--input', str(source), '--output', str(first), '--stop-after', 'detection',
                               '--no-metrics'])
                final = Path(directory) / 'final.json'
                pipeline.main(['--resume-from', str(first), '--output', str(final), '--references', str(source)])
            report = json.loads(next(Path(directory).glob('final_phases_*/metrics.json')).read_text())
        self.assertAlmostEqual(report['detection']['f1'], 0.8)
        self.assertEqual(report['cwe_id']['expected'], 3)


if __name__ == '__main__':
    unittest.main()
