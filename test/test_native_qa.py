"""Checks for answer scoring and joining labels to the correct prompts."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.experiments import native_qa


class NativeQATests(unittest.TestCase):
    def test_hotpot_scoring(self):
        self.assertEqual(native_qa.score('The Eiffel Tower!', 'eiffel tower'), (1.0, 1.0))
        self.assertAlmostEqual(native_qa.score('red blue', 'red green')[1], 0.5)
        self.assertEqual(native_qa.score('yes indeed', 'yes'), (0.0, 0.0))
        self.assertEqual(native_qa.score('no', 'yes'), (0.0, 0.0))
        self.assertEqual(native_qa.score('', ''), (1.0, 1.0))

    def test_labels_must_match_prompt_split_and_identity(self):
        samples = [{'id': str(i), 'split': 'validation', 'source_row': i, 'text': f'prompt {i}',
                    'text_sha256': hashlib.sha256(f'prompt {i}'.encode()).hexdigest()} for i in range(200)]
        manifest = json.dumps({'models': native_qa.MODELS, 'samples': samples}).encode()
        manifest_sha = hashlib.sha256(manifest).hexdigest()
        labels = {'manifest_sha256': manifest_sha, 'labels': [
            {'id': str(i), 'split': 'validation', 'source_row': i, 'answer': 'answer'} for i in range(200)]}
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path, labels_path = Path(tmp) / 'manifest.json', Path(tmp) / 'labels.json'
            manifest_path.write_bytes(manifest)
            def load():
                data = json.dumps(labels).encode()
                labels_path.write_bytes(data)
                with patch.object(native_qa, 'MANIFEST_SHA256', manifest_sha), patch.object(native_qa, 'LABELS_SHA256', hashlib.sha256(data).hexdigest()):
                    return native_qa.load_inputs(manifest_path, labels_path)
            self.assertEqual(len(load()[1]), 200)
            labels['labels'][0]['split'] = 'train'
            with self.assertRaisesRegex(ValueError, 'provenance'):
                load()
            labels['labels'][0]['split'] = 'validation'
            labels['labels'][0]['id'] = labels['labels'][1]['id']
            with self.assertRaisesRegex(ValueError, 'IDs'):
                load()


if __name__ == '__main__':
    unittest.main()
