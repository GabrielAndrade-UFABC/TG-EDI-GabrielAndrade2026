import csv
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pipeline

EXAMPLE = Path(__file__).resolve().parents[1] / 'examples/pipeline_input.json'


class FakeModel:
    def __init__(self, kind):
        self.kind = kind
    def detect(self, code):
        return 0.9 if 'strcpy' in code else 0.1
    def generate(self, prompt):
        return 'CWE-120: overflow' if self.kind == 'cwe' else 'void corrected(void) {}'
    def close(self):
        pass


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.stdout = redirect_stdout(io.StringIO())
        self.stderr = redirect_stderr(io.StringIO())
        self.stdout.__enter__()
        self.stderr.__enter__()
        self.addCleanup(self.stdout.__exit__, None, None, None)
        self.addCleanup(self.stderr.__exit__, None, None, None)

    def test_reports_are_available_before_next_phase(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calls = []
            prompts = []
            class Model(FakeModel):
                def __init__(self, kind):
                    calls.append(kind)
                    previous = {'cwe': '01_detection', 'fix': '02_cwe'}.get(kind)
                    if previous:
                        for suffix in ('csv', 'json', 'html'):
                            assert (root / f'{previous}.{suffix}').is_file()
                    super().__init__(kind)
                def generate(self, prompt):
                    prompts.append(prompt)
                    return super().generate(prompt)
            rows = pipeline.run_pipeline(pipeline.read_records(EXAMPLE), Model,
                phase_finished=lambda kind, records: pipeline.write_phase_results(root, kind, records))
            self.assertEqual(calls, ['detection', 'cwe', 'fix'])
            self.assertEqual(prompts[-1], pipeline.fix_prompt(rows[0]['code']))
            self.assertEqual(rows[0]['status'], 'generated_unvalidated')
            self.assertEqual(rows[1]['status'], 'not_vulnerable')
            snapshot = pipeline.read_checkpoint(root / '01_detection.json')
            self.assertIsNone(snapshot[0]['fixed_code'])

    def test_csv_formula_and_html_injection_are_neutralized(self):
        rows = pipeline.read_records(EXAMPLE)
        rows[0]['id'] = '=HYPERLINK("https://example.com")'
        rows[0]['code'] = '<script>alert(1)</script>\n=2+2'
        with tempfile.TemporaryDirectory() as directory:
            path = pipeline.write_phase_results(directory, 'detection', rows)
            with path.open(encoding='utf-8-sig', newline='') as stream:
                table = list(csv.DictReader(stream))
            self.assertTrue(table[0]['id'].startswith("'="))
            html = path.with_suffix('.html').read_text()
            self.assertNotIn('<script>', html)
            self.assertIn('&lt;script&gt;', html)
            raw = json.loads(path.with_suffix('.json').read_text())
            self.assertEqual(raw[0]['id'], rows[0]['id'])
        for value in ('=1', '  +1', '-2', '@SUM(1)', '\t=1', '\r=1', '\n=1'):
            self.assertTrue(pipeline.spreadsheet_cell(value).startswith("'"))

    def test_resume_rejects_forged_history_and_invalid_probability(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint.json'
            rows = pipeline.read_records(EXAMPLE)
            rows[0]['phase_results'] = {'detection': {'status': 'success'}}
            pipeline.write_json(path, rows)
            with self.assertRaisesRegex(ValueError, 'inconsistente'):
                pipeline.read_checkpoint(path)
            rows[0]['phase_results'] = {'detection': 'broken'}
            pipeline.write_json(path, rows)
            with self.assertRaisesRegex(ValueError, 'histórico'):
                pipeline.read_checkpoint(path)
            rows[0]['phase_results'] = {}
            rows[0]['vulnerability_probability'] = 9
            pipeline.write_json(path, rows)
            with self.assertRaisesRegex(ValueError, 'probabilidade'):
                pipeline.read_checkpoint(path)

    def test_unknown_cwe_allows_fix_without_execution_error(self):
        calls = []
        class Model(FakeModel):
            def __init__(self, kind):
                calls.append(kind)
                super().__init__(kind)
            def generate(self, prompt):
                return 'NVD-CWE-noinfo'
        rows = pipeline.run_pipeline(pipeline.read_records(EXAMPLE), Model)
        self.assertIn('fix', calls)
        self.assertEqual(rows[0]['status'], 'generated_unvalidated')
        self.assertIsNone(rows[0]['error'])
        self.assertEqual(pipeline.phase_outcome(rows[0], 'cwe')[0], 'unidentified')
        self.assertEqual(pipeline.phase_outcome(rows[0], 'fix')[0], 'success')

    def test_nan_detection_is_error(self):
        class Model(FakeModel):
            def detect(self, code):
                return float('nan')
        rows = pipeline.run_pipeline(pipeline.read_records(EXAMPLE), Model)
        self.assertTrue(all(r['status'] == 'error' for r in rows))
        self.assertTrue(all(r['error']['stage'] == 'detection' for r in rows))

    def test_model_loading_failure_keeps_reports_and_skips_later_models(self):
        calls = []
        def fail_load(kind):
            calls.append(kind)
            raise OSError('pesos indisponíveis')
        with tempfile.TemporaryDirectory() as directory:
            rows = pipeline.run_pipeline(pipeline.read_records(EXAMPLE), fail_load,
                phase_finished=lambda kind, records: pipeline.write_phase_results(directory, kind, records))
            self.assertEqual(calls, ['detection'])
            self.assertTrue(all(r['error']['message'] == 'pesos indisponíveis' for r in rows))
            self.assertTrue((Path(directory) / '03_fix.html').is_file())

    def test_retry_errors_resumes_failed_records_through_all_stages(self):
        def fail_load(kind):
            raise OSError('pesos indisponíveis')
        rows = pipeline.run_pipeline(pipeline.read_records(EXAMPLE), fail_load)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint.json'
            pipeline.write_json(path, rows)
            resumed = pipeline.read_checkpoint(path)
            self.assertEqual(pipeline.reset_errors(resumed), len(resumed))
            pipeline.run_pipeline(resumed, FakeModel)
        self.assertTrue(all(r['error'] is None and r['status'] != 'error' for r in resumed))
        self.assertTrue(any(r['status'] == 'generated_unvalidated' for r in resumed))

    def test_retry_cwe_failure_keeps_detection_and_fix(self):
        class BadCwe(FakeModel):
            def generate(self, prompt):
                if self.kind == 'cwe':
                    raise RuntimeError('oom')
                return super().generate(prompt)
        rows = pipeline.run_pipeline(pipeline.read_records(EXAMPLE), BadCwe)
        self.assertEqual(rows[0]['error']['stage'], 'cwe')
        self.assertEqual(pipeline.phase_outcome(rows[0], 'fix')[0], 'success')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint.json'
            pipeline.write_json(path, rows)
            resumed = pipeline.read_checkpoint(path)
            self.assertEqual(pipeline.reset_errors(resumed), 1)
            calls = []
            def factory(kind):
                calls.append(kind)
                return FakeModel(kind)
            pipeline.run_pipeline(resumed, factory)
        self.assertEqual(calls, ['cwe'])
        self.assertEqual(resumed[0]['fixed_code'], rows[0]['fixed_code'])
        self.assertEqual(resumed[0]['status'], 'generated_unvalidated')

    def test_checkpoint_write_failure_is_not_labeled_as_inference_error(self):
        closed = []
        class Model(FakeModel):
            def close(self):
                closed.append(self.kind)
        def fail_save(records):
            raise OSError('sem espaço')
        rows = pipeline.read_records(EXAMPLE)
        with self.assertRaisesRegex(OSError, 'sem espaço'):
            pipeline.run_pipeline(rows, Model, checkpoint=fail_save)
        self.assertEqual(rows[0]['status'], 'detected')
        self.assertIsNone(rows[0]['error'])
        self.assertEqual(closed, ['detection'])

    def test_failed_atomic_write_preserves_file_and_cleans_temp(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'output.json'
            path.write_text('original')
            with self.assertRaises(ValueError):
                pipeline.write_json(path, {'value': float('nan')})
            self.assertEqual(path.read_text(), 'original')
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_stop_and_resume_preserve_finished_predictions(self):
        calls = []
        def factory(kind):
            calls.append(kind)
            return FakeModel(kind)
        rows = pipeline.run_pipeline(pipeline.read_records(EXAMPLE), factory, stop_after='detection')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint.json'
            pipeline.write_json(path, rows)
            resumed = pipeline.read_checkpoint(path)
            pipeline.run_pipeline(resumed, factory)
        self.assertEqual(calls, ['detection', 'cwe', 'fix'])
        self.assertEqual(resumed[0]['vulnerability_probability'], 0.9)

    def test_default_main_and_reports_with_simulated_models(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'examples').mkdir()
            (root / 'examples/pipeline_input.json').write_text(EXAMPLE.read_text())
            with patch.object(pipeline, 'ROOT', root), patch.object(pipeline, 'LocalModel', lambda path, kind, *args: FakeModel(kind)):
                self.assertEqual(pipeline.main([]), 0)
            phase_dir = next((root / 'results').glob('pipeline_results_phases_*'))
            self.assertEqual(json.loads((root / 'results/pipeline_results.json').read_text()),
                             json.loads((phase_dir / 'pipeline_results.json').read_text()))
            self.assertEqual(json.loads((phase_dir / 'summary.json').read_text())['execution_state'], 'completed')
            self.assertTrue((phase_dir / '03_fix.html').exists())

    def test_main_passes_independent_input_token_limits_to_models(self):
        cases = (
            (['--detection-max-input-tokens', '256', '--cwe-max-input-tokens', '80', '--fix-max-input-tokens', '640'],
             {'detection': 256, 'cwe': 80, 'fix': 640}),
            (['--cwe-max-input-tokens', '80'], {'detection': 512, 'cwe': 80, 'fix': 512}),
        )
        for flags, expected in cases:
            with self.subTest(flags=flags), tempfile.TemporaryDirectory() as directory:
                limits = {}

                def load_model(path, kind, device, max_new_tokens, max_input_tokens):
                    self.assertEqual(max_new_tokens, 200)
                    limits[kind] = max_input_tokens
                    return FakeModel(kind)

                root = Path(directory)
                with patch.object(pipeline, 'LocalModel', load_model):
                    self.assertEqual(pipeline.main(['--input', str(EXAMPLE), '--output',
                                                    str(root / 'out.json'), '--no-metrics',
                                                    '--max-new-tokens', '200', *flags]), 0)
                self.assertEqual(limits, expected)
                summary = next(root.glob('out_phases_*/summary.json'))
                self.assertEqual(json.loads(summary.read_text(encoding='utf-8'))['input_token_limits'], expected)

    def test_debug_prints_records_and_partial_metrics_only_when_requested(self):
        rows = [{'id': f'r{i}', 'code': 'strcpy(a, b);' if i % 2 else 'int f(void) { return 1; }',
                 'label': i % 2} for i in range(4)]
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'labeled.json'
            source.write_text(json.dumps(rows))
            logs = {}
            for flag in ([], ['--yolo', '--metrics-every', '2']):
                err = io.StringIO()
                output = Path(directory) / f'out{len(flag)}.json'
                with patch.object(pipeline, 'LocalModel', lambda path, kind, *args: FakeModel(kind)), redirect_stderr(err):
                    self.assertEqual(pipeline.main(['--input', str(source), '--output', str(output), *flag]), 0)
                logs[len(flag)] = err.getvalue()
            self.assertNotIn('[debug]', logs[0])
            self.assertIn('Métricas até esta fase', logs[0])
            self.assertIn('[debug] detection r1', logs[3])
            self.assertIn('métricas parciais de detection (2/4', logs[3])

    def test_hardlinked_input_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'input.json'
            output = Path(directory) / 'output.json'
            source.write_text(EXAMPLE.read_text())
            os.link(source, output)
            with self.assertRaises(SystemExit):
                pipeline.main(['--input', str(source), '--output', str(output)])
            self.assertEqual(source.read_text(), EXAMPLE.read_text())

    def test_interruption_is_recorded_and_can_be_resumed(self):
        class Model(FakeModel):
            def generate(self, prompt):
                raise KeyboardInterrupt()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'output.json'
            with patch.object(pipeline, 'LocalModel', lambda path, kind, *args: Model(kind)):
                self.assertEqual(pipeline.main(['--input', str(EXAMPLE), '--output', str(output)]), 130)
            phase_dir = next(Path(directory).glob('output_phases_*'))
            summary = json.loads((phase_dir / 'summary.json').read_text())
            self.assertEqual(summary['execution_state'], 'interrupted')
            rows = pipeline.read_checkpoint(phase_dir / 'pipeline_results.json')
            self.assertEqual(rows[0]['status'], 'detected')


class BatchingTests(unittest.TestCase):
    def setUp(self):
        self.stderr = redirect_stderr(io.StringIO())
        self.stderr.__enter__()
        self.addCleanup(self.stderr.__exit__, None, None, None)
        self.records = [{'id': f'r{i}', 'code': f'strcpy(b, s{i:02});', 'status': 'pending'} for i in range(18)]

    def test_each_phase_prefetches_its_inputs_in_chunks_of_batch_size(self):
        batches = []
        class Model(FakeModel):
            def prefetch(self, inputs):
                batches.append((self.kind, list(inputs)))
        rows = pipeline.run_pipeline(self.records, Model)
        self.assertEqual([(kind, len(inputs)) for kind, inputs in batches],
                         [(kind, size) for kind in pipeline.PHASES for size in (16, 2)])
        codes = [r['code'] for r in rows]
        self.assertEqual(batches[0][1] + batches[1][1], codes)
        self.assertEqual(batches[2][1], [pipeline.cwe_prompt(c) for c in codes[:16]])
        self.assertEqual(batches[4][1], [pipeline.fix_prompt(c) for c in codes[:16]])
        self.assertTrue(all(r['status'] == 'generated_unvalidated' for r in rows))

    def test_failed_prefetch_falls_back_to_one_record_at_a_time(self):
        class Model(FakeModel):
            def prefetch(self, inputs):
                raise RuntimeError('CUDA out of memory')
        rows = pipeline.run_pipeline(self.records, Model)
        self.assertTrue(all(r['status'] == 'generated_unvalidated' for r in rows))


if __name__ == '__main__':
    unittest.main()
