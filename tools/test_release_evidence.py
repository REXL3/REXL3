"""Synthetic CPU-only fixtures: never qualification or native execution evidence."""
import copy
import hashlib
import importlib
import importlib.util
from unittest.mock import patch
import io
import json
import os
from pathlib import Path
import tarfile
import tempfile
import unittest

GATES = ('doctor', 'lifecycle', 'capability', 'vision', 'context', 'exactness',
         'prefix_cache', 'performance', 'quality')


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2) + '\n').encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write(root, name, data):
    p = Path(root) / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    p.chmod(0o644)
    return p


def archive(path, entries):
    with tarfile.open(path, 'w:gz') as out:
        for name, data, kind in entries:
            info = tarfile.TarInfo(name)
            info.mode = 0o644
            info.type = kind
            if kind in (tarfile.SYMTYPE, tarfile.LNKTYPE):
                info.linkname = '../escape'
            info.size = len(data) if kind == tarfile.REGTYPE else 0
            out.addfile(info, io.BytesIO(data) if info.isfile() else None)


class ReceiptContract(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.expected = dict(runtime_id='synthetic-demo', profile='single',
                             engine_commit='a' * 40, engine_tree='b' * 40,
                             binary_sha256='c' * 64, profile_sha256='d' * 64,
                             model_lock_sha256='e' * 64, release_lock_sha256='f' * 64,
                             effective_profile_digest='sha256:' + sha(json.dumps({'max_context': 1048576}, sort_keys=True, separators=(',', ':')).encode()),
                             effective_profile_source='owned-launcher-observation',
                             effective_max_context=1048576,
                             capabilities={'images': True, 'audio': False}, topology={'ranks': 1})
        self.resolved = {'max_context': 1048576, 'capabilities': self.expected['capabilities'],
                         'ranks': [{'rank': 0, 'argv': ['synthetic-serve', '--max-model-len', '1048576'],
                                    'env': {'SYNTHETIC_FIXTURE': '1'}}]}
        self.expected['effective_profile_digest'] = 'sha256:' + sha(json.dumps(self.resolved, sort_keys=True, separators=(',', ':')).encode())
        self.doc = dict(schema='rexl3.qualification.v1', **self.expected, gates={})
        for gate in GATES:
            raw = {'schema': 'rexl3.gate-evidence.v1', 'gate': gate,
                   **self.expected, 'cases': [{'name': 'synthetic-positive', 'status': 'PASS',
                                             'modality': 'image' if gate == 'vision' else 'text',
                                             'expectation': 'supported'}]}
            p = write(self.root, gate + '.json', encoded(raw))
            self.doc['gates'][gate] = dict(status='PASS', passed=1, total=1, failed=0,
                                           skipped=0, evidence=[{'path': p.name, 'sha256': sha(p.read_bytes())}])
        observation = {'schema': 'rexl3.launcher-observation.v1', **self.expected,
                       'resolved_config': self.resolved, 'ranks': [
                           {'rank': 0, 'pid': 123, 'start_time_ticks': 456, 'hostname': 'synthetic-host',
                            'binary_sha256': self.expected['binary_sha256'],
                            'engine_commit': self.expected['engine_commit'], 'engine_tree': self.expected['engine_tree'],
                            'argv': ['synthetic-serve', '--max-model-len', '1048576'],
                            'env': {'SYNTHETIC_FIXTURE': '1'}, 'ready': True}]}
        p = write(self.root, 'identity.json', encoded(observation))
        self.doc['identity_receipt'] = {'path': p.name, 'sha256': sha(p.read_bytes())}
        self.receipt = self.root / 'receipt.json'

    def test_naked_digest_is_not_observation(self):
        del self.doc['identity_receipt']
        with self.assertRaises(ValueError):
            self.validate()

    def test_resolved_observation_and_all_rank_identity_bound(self):
        path = self.root / 'identity.json'
        original = json.loads(path.read_bytes())
        for field, value in [('resolved_config', {'max_context': 32}), ('ranks', []),
                             ('effective_profile_source', '/health')]:
            with self.subTest(field=field):
                changed = copy.deepcopy(original)
                changed[field] = value
                path.write_bytes(encoded(changed))
                self.doc['identity_receipt']['sha256'] = sha(path.read_bytes())
                with self.assertRaises(ValueError):
                    self.validate()

    def api(self):
        spec = importlib.util.find_spec('tools.release_evidence')
        self.assertIsNotNone(spec, 'release evidence acceptance boundary is missing')
        return importlib.import_module('tools.release_evidence')

    def validate(self):
        self.receipt.write_bytes(encoded(self.doc))
        return self.api().validate_receipt(self.receipt, evidence_root=self.root, expected=self.expected)

    def test_valid_actual_files(self):
        self.assertEqual(self.validate(), self.doc)

    def test_gate_truthy_strings_fail_closed(self):
        for value in ('FAIL', 'PASS', True, '15/15'):
            with self.subTest(value=value):
                self.doc['gates']['doctor'] = value
                with self.assertRaises(ValueError):
                    self.validate()

    def test_every_required_gate_is_mandatory(self):
        original = copy.deepcopy(self.doc)
        for gate in GATES:
            with self.subTest(gate=gate):
                self.doc = copy.deepcopy(original)
                del self.doc['gates'][gate]
                with self.assertRaises(ValueError):
                    self.validate()

    def test_zero_skips_failure_and_boolean_counts_are_not_passes(self):
        original = copy.deepcopy(self.doc)
        for field, value in [('status', 'FAIL'), ('status', 'NOT_RUN'), ('total', 0),
                             ('passed', True), ('total', True), ('failed', 1), ('skipped', 1),
                             ('passed', 0), ('passed', '1'), ('evidence', [])]:
            with self.subTest(field=field, value=value):
                self.doc = copy.deepcopy(original)
                self.doc['gates']['doctor'][field] = value
                with self.assertRaises(ValueError):
                    self.validate()

    def test_all_identity_drift_rejected(self):
        original = copy.deepcopy(self.doc)
        for key in self.expected:
            with self.subTest(key=key):
                self.doc = copy.deepcopy(original)
                self.doc[key] = None
                with self.assertRaises(ValueError):
                    self.validate()

    def test_evidence_bytes_missing_symlink_and_paths_rejected(self):
        p = self.root / 'doctor.json'
        p.write_bytes(b'changed')
        with self.assertRaises(ValueError):
            self.validate()
        p.unlink()
        with self.assertRaises(ValueError):
            self.validate()
        p.symlink_to(self.root / 'vision.json')
        with self.assertRaises(ValueError):
            self.validate()
        for name in ('../escape', '/absolute', 'a//b', './doctor.json', 'a\\b'):
            self.doc['gates']['doctor']['evidence'][0]['path'] = name
            with self.subTest(path=name), self.assertRaises(ValueError):
                self.validate()

    def test_audio_refusal_is_not_positive_vision(self):
        p = self.root / 'vision.json'
        raw = json.loads(p.read_bytes())
        raw['cases'][0].update(modality='audio', expectation='refused')
        p.write_bytes(encoded(raw))
        self.doc['gates']['vision']['evidence'][0]['sha256'] = sha(p.read_bytes())
        with self.assertRaises(ValueError):
            self.validate()

    def test_duplicate_json_keys_refused(self):
        self.receipt.write_bytes(b'{"schema": "x", "schema": "rexl3.qualification.v1"}')
        with self.assertRaises(ValueError):
            self.api().validate_receipt(self.receipt, evidence_root=self.root, expected=self.expected)


class ArchiveContract(unittest.TestCase):
    def test_rejects_every_unsafe_tar_before_destination_exists(self):
        from tools import promote_runtime as promotion
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name, kind in [('../escape', tarfile.REGTYPE), ('/absolute', tarfile.REGTYPE),
                               ('a/../escape', tarfile.REGTYPE), ('a//b', tarfile.REGTYPE),
                               ('link', tarfile.SYMTYPE), ('link', tarfile.LNKTYPE),
                               ('device', tarfile.CHRTYPE), ('fifo', tarfile.FIFOTYPE),
                               ('ok', tarfile.REGTYPE)]:
                with self.subTest(name=name, kind=kind):
                    archive(root / 'bad.tgz', [('ok', b'ok', tarfile.REGTYPE), (name, b'bad', kind)])
                    dest = root / 'output'
                    # RED runs must not exercise the known write vulnerability.
                    with patch.object(promotion.os, 'makedirs', side_effect=AssertionError('write before preflight')):
                        with self.assertRaises((ValueError, SystemExit)):
                            promotion.extract_tree(root / 'bad.tgz', dest, {})
                    self.assertFalse(dest.exists(), 'preflight must complete before any output')


if __name__ == '__main__':
    unittest.main()
