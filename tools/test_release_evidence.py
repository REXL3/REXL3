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
from tools import release_evidence as e

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


def refusal_detail(**changes):
    """Measured detail a vision-off harness stores for the refused image case."""
    facts = {'http_status': 400, 'error': {'message': 'image input is not supported by this server (started without image support)'},
             'wall_s': 0.02, 'positive_capability': False, 'server_healthy_after': True}
    facts.update(changes)
    return {'case': {'id': 'vision/image_refused'}, 'details': facts, 'error': None}


def stage_assets(dest, f, doc):
    """Copy the sealed release archives into the family/model assets layout and extract bound evidence."""
    import shutil
    from tools import release_evidence as e
    dest.mkdir(parents=True, exist_ok=True)
    for name in ['approval.json'] + [a['archive'] for a in e.approval_assets(doc)]:
        shutil.copyfile(f.assets / name, dest / name)
    evidence = doc['evidence_asset']
    e.extract_files(dest / evidence['archive'], dest, evidence['members'])
    return dest


class ReceiptContract(unittest.TestCase):
    images = True

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.expected = dict(runtime_id='synthetic-demo', profile='single', pack='target',
                             engine_commit='a' * 40, engine_tree='b' * 40,
                             binary_sha256='c' * 64, profile_sha256='d' * 64,
                             model_lock_sha256='e' * 64, release_lock_sha256='f' * 64,
                             effective_profile_digest='sha256:' + sha(json.dumps({'max_context': 1048576}, sort_keys=True, separators=(',', ':')).encode()),
                             effective_profile_source='owned-launcher-observation',
                             effective_max_context=1048576,
                             capabilities={'images': self.images, 'audio': False}, topology={'ranks': 1})
        self.argv = ['synthetic-serve', '--max-model-len', '1048576'] + ([] if self.images else ['--no-vision'])
        self.resolved = {'max_context': 1048576, 'capabilities': self.expected['capabilities'],
                         'ranks': [{'rank': 0, 'argv': self.argv, 'env': {'SYNTHETIC_FIXTURE': '1'}}]}
        self.expected['effective_profile_digest'] = 'sha256:' + sha(json.dumps(self.resolved, sort_keys=True, separators=(',', ':')).encode())
        self.doc = dict(schema='rexl3.qualification.v1', **self.expected, gates={})
        for gate in GATES:
            case = {'name': 'synthetic-positive', 'status': 'PASS',
                    'modality': 'image' if gate == 'vision' else 'text', 'expectation': 'supported'}
            extra = []
            if gate == 'vision' and not self.images:
                detail = write(self.root, 'vision-detail.json', encoded(refusal_detail()))
                self.detail_ref = {'path': detail.name, 'sha256': sha(detail.read_bytes())}
                case.update(name='image_refused', expectation='refused', result=self.detail_ref)
                extra = [self.detail_ref]
            raw = {'schema': 'rexl3.gate-evidence.v1', 'gate': gate, **self.expected, 'cases': [case]}
            p = write(self.root, gate + '.json', encoded(raw))
            self.doc['gates'][gate] = dict(status='PASS', passed=1, total=1, failed=0, skipped=0,
                                           evidence=[{'path': p.name, 'sha256': sha(p.read_bytes())}, *extra])
        observation = {'schema': 'rexl3.launcher-observation.v1', **self.expected,
                       'resolved_config': self.resolved, 'ranks': [
                           {'rank': 0, 'pid': 123, 'start_time_ticks': 456, 'hostname': 'synthetic-host',
                            'binary_sha256': self.expected['binary_sha256'],
                            'engine_commit': self.expected['engine_commit'], 'engine_tree': self.expected['engine_tree'],
                            'argv': self.argv,
                            'env': {'SYNTHETIC_FIXTURE': '1'}, 'ready': True}]}
        p = write(self.root, 'identity.json', encoded(observation))
        self.doc['identity_receipt'] = {'path': p.name, 'sha256': sha(p.read_bytes())}
        if not self.images:
            self.doc['vision_exception'] = dict(granted_by='am423', granted='2026-10-25',
                                                expires='2099-12-31', reason='synthetic owner exception')
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

    def test_vision_on_accepts_a_valid_exception_record(self):
        # A vision-on release under an exception (images accepted, colour answers excused) carries its record.
        self.doc['vision_exception'] = dict(granted_by='am423', granted='2026-10-25',
                                            expires='2099-12-31', reason='synthetic owner exception')
        self.assertEqual(self.validate(), self.doc)

    def test_vision_on_refuses_an_expired_exception_record(self):
        self.doc['vision_exception'] = dict(granted_by='am423', granted='2026-10-25',
                                            expires='2020-01-01', reason='synthetic owner exception')
        with self.assertRaisesRegex(ValueError, 'expired on 2020-01-01'):
            self.validate()

    def test_vision_on_refuses_a_malformed_exception_record(self):
        self.doc['vision_exception'] = dict(granted_by='mallory', granted='2026-10-25',
                                            expires='2099-12-31', reason='synthetic owner exception')
        with self.assertRaisesRegex(ValueError, 'granted_by must be am423 or vcruz305'):
            self.validate()
        self.doc['vision_exception'] = dict(granted_by='am423', granted='2026-10-25', expires='2099-12-31')
        with self.assertRaisesRegex(ValueError, 'must carry granted_by'):
            self.validate()

    def test_a_record_is_required_only_when_vision_is_off(self):
        self.doc.pop('vision_exception', None)
        if self.images:
            self.assertEqual(self.validate(), self.doc)
        else:
            with self.assertRaisesRegex(ValueError, 'documented owner vision exception'):
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

    def test_a_receipt_without_the_pack_identity_key_is_refused(self):
        del self.doc['pack']
        with self.assertRaisesRegex(ValueError, 'incomplete receipt identity'):
            self.validate()
        self.doc['pack'] = 'Not A Role'
        with self.assertRaisesRegex(ValueError, 'invalid pack'):
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

    def rewrite_vision(self, mutate):
        p = self.root / 'vision.json'
        raw = json.loads(p.read_bytes())
        mutate(raw)
        p.write_bytes(encoded(raw))
        self.doc['gates']['vision']['evidence'][0]['sha256'] = sha(p.read_bytes())

    def test_vision_with_zero_image_cases_is_rejected(self):
        # Text-only vision evidence proves neither a served nor a refused image.
        self.rewrite_vision(lambda raw: raw['cases'][0].update(modality='text'))
        with self.assertRaisesRegex(ValueError, 'positive supported image|requires a PASS refused image'):
            self.validate()

    def test_vision_on_rejects_a_refusal_only_gate(self):
        # images=true must keep requiring positive evidence; a clean refusal is not vision.
        self.rewrite_vision(lambda raw: raw['cases'][0].update(expectation='refused'))
        with self.assertRaisesRegex(ValueError, 'positive supported image evidence required'):
            self.validate()

    def test_forged_capability_maps_are_rejected(self):
        # The receipt claims the opposite of what the profile/observation pinned.
        self.doc['capabilities'] = dict(self.doc['capabilities'], images=not self.images)
        with self.assertRaisesRegex(ValueError, 'receipt identity drift'):
            self.validate()
        self.doc['capabilities'] = dict(self.expected['capabilities'])
        expected = copy.deepcopy(self.expected)
        expected['capabilities']['images'] = not self.images
        self.receipt.write_bytes(encoded(self.doc))
        with self.assertRaisesRegex(ValueError, 'receipt identity drift'):
            self.api().validate_receipt(self.receipt, evidence_root=self.root, expected=expected)

    def test_duplicate_json_keys_refused(self):
        self.receipt.write_bytes(b'{"schema": "x", "schema": "rexl3.qualification.v1"}')
        with self.assertRaises(ValueError):
            self.api().validate_receipt(self.receipt, evidence_root=self.root, expected=self.expected)


class VisionOffReceiptContract(ReceiptContract):
    """The whole receipt contract again, for a release that ships with images=false."""
    images = False
    test_vision_on_rejects_a_refusal_only_gate = None  # vision-on contract only

    def test_vision_off_receipt_is_valid_with_refusal_evidence(self):
        self.assertEqual(self.expected['capabilities']['images'], False)
        self.assertEqual(self.validate(), self.doc)

    def test_vision_off_rejects_a_supported_image_case(self):
        self.rewrite_vision(lambda raw: raw['cases'][0].update(expectation='supported'))
        with self.assertRaisesRegex(ValueError, 'images=false forbids supported image cases'):
            self.validate()

    def test_vision_off_rejects_supported_case_next_to_valid_refusal(self):
        def add(raw):
            raw['cases'].append({'name': 'red', 'status': 'PASS', 'modality': 'image', 'expectation': 'supported'})
        self.rewrite_vision(add)
        self.doc['gates']['vision'].update(passed=2, total=2)
        with self.assertRaisesRegex(ValueError, 'images=false forbids supported image cases'):
            self.validate()

    def test_vision_off_rejects_audio_refusal_as_vision(self):
        self.rewrite_vision(lambda raw: raw['cases'][0].update(modality='audio'))
        with self.assertRaisesRegex(ValueError, 'requires a PASS refused image case'):
            self.validate()

    def test_vision_off_refusal_must_be_measured_4xx_not_5xx_or_timeout(self):
        bad = {'5xx': dict(http_status=503), '500': dict(http_status=500), 'timeout': dict(http_status=None),
               '200': dict(http_status=200), 'bool_status': dict(http_status=True),
               'missing_status': None, 'no_message': dict(error={'message': ''}),
               'no_error_object': dict(error=None), 'hung': dict(wall_s=61.0), 'negative_wall': dict(wall_s=-1),
               'unhealthy_after': dict(server_healthy_after=False), 'claims_positive': dict(positive_capability=True)}
        for name, change in bad.items():
            with self.subTest(case=name):
                detail = refusal_detail()
                if change is None:
                    del detail['details']['http_status']
                else:
                    detail['details'].update(change)
                path = self.root / 'vision-detail.json'
                path.write_bytes(encoded(detail))
                digest = sha(path.read_bytes())
                self.rewrite_vision(lambda raw: raw['cases'][0]['result'].update(sha256=digest))
                self.doc['gates']['vision']['evidence'][1]['sha256'] = digest
                with self.assertRaises(ValueError):
                    self.validate()

    def test_vision_off_refusal_detail_failure_or_unbound_reference_rejected(self):
        path = self.root / 'vision-detail.json'
        detail = refusal_detail()
        detail['error'] = 'TimeoutError: read timed out'
        path.write_bytes(encoded(detail))
        digest = sha(path.read_bytes())
        self.rewrite_vision(lambda raw: raw['cases'][0]['result'].update(sha256=digest))
        self.doc['gates']['vision']['evidence'][1]['sha256'] = digest
        with self.assertRaisesRegex(ValueError, 'records a failure'):
            self.validate()
        # No result pointer, or a pointer outside the gate's referenced evidence.
        self.setUp()
        self.rewrite_vision(lambda raw: raw['cases'][0].pop('result'))
        with self.assertRaisesRegex(ValueError, 'detail reference required'):
            self.validate()
        self.setUp()
        del self.doc['gates']['vision']['evidence'][1]
        with self.assertRaisesRegex(ValueError, 'referenced gate evidence'):
            self.validate()

    def test_vision_off_observed_process_without_no_vision_rejected(self):
        # Receipt claims images=false but the observed process was not started --no-vision.
        for mode in ('absent', 'duplicate', 'attached'):
            self.setUp()
            path = self.root / 'identity.json'
            observation = json.loads(path.read_bytes())
            argv = ['synthetic-serve', '--max-model-len', '1048576']
            argv += {'absent': [], 'duplicate': ['--no-vision', '--no-vision'], 'attached': ['--no-vision=true']}[mode]
            observation['ranks'][0]['argv'] = argv
            observation['resolved_config']['ranks'][0]['argv'] = argv
            digest = 'sha256:' + sha(json.dumps(observation['resolved_config'], sort_keys=True, separators=(',', ':')).encode())
            observation['effective_profile_digest'] = digest
            self.expected['effective_profile_digest'] = digest
            for target in (self.doc, observation):
                target['effective_profile_digest'] = digest
            for gate in GATES:
                p = self.root / (gate + '.json')
                raw = json.loads(p.read_bytes())
                raw['effective_profile_digest'] = digest
                p.write_bytes(encoded(raw))
                self.doc['gates'][gate]['evidence'][0]['sha256'] = sha(p.read_bytes())
            path.write_bytes(encoded(observation))
            self.doc['identity_receipt']['sha256'] = sha(path.read_bytes())
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, 'exactly one bare --no-vision'):
                self.validate()

    def test_vision_off_refuses_a_missing_exception_record(self):
        del self.doc['vision_exception']
        with self.assertRaisesRegex(ValueError, 'documented owner vision exception'):
            self.validate()

    def test_vision_off_refuses_an_expired_exception_record(self):
        self.doc['vision_exception'] = dict(granted_by='am423', granted='2026-10-25',
                                            expires='2020-01-01', reason='synthetic owner exception')
        with self.assertRaisesRegex(ValueError, 'expired on 2020-01-01'):
            self.validate()

    def test_vision_off_refuses_a_malformed_exception_record(self):
        base = dict(granted_by='am423', granted='2026-10-25', expires='2099-12-31',
                    reason='synthetic owner exception')
        for change in ({'granted_by': 'mallory'}, {'granted': '2026-99-99'}, {'expires': 'not-a-date'},
                       {'reason': ' '}, {'tracking': 'x'}):
            with self.subTest(change=change):
                self.setUp()
                self.doc['vision_exception'] = dict(base, **change)
                with self.assertRaises(ValueError):
                    self.validate()


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


class PrecomputedInventoryTest(unittest.TestCase):
    """Precomputed inventory pass-through: digest equality stays mandatory everywhere.

    A precomputed member inventory may only replace re-derivation, never the
    fresh streaming digest of the compressed transport bytes.
    """

    def setUp(self):
        from tools.test_approve_profile import SyntheticSeal
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.f = SyntheticSeal(self.tmp.name)
        self.approval, self.doc, self.registry = self.f.published_fixture()

    def precomputed(self, **exclude):
        skip = {exclude.get('skip', '')}
        return {asset['archive']: asset['members'] for asset in e.approval_assets(self.doc)
                if asset['archive'] not in skip}

    def test_default_full_inspection_path_is_unchanged(self):
        verified, archive = e.verify_assets(self.approval, registry=self.registry,
                                             approved_templates=self.f.approved)
        self.assertEqual(verified, self.doc)
        self.assertEqual(archive, self.f.archive)

    def test_precomputed_inventories_skip_decompression(self):
        with patch('tools.release_evidence.inspect_archive',
                   side_effect=AssertionError('must not run')):
            verified, archive = e.verify_assets(self.approval, registry=self.registry,
                                                approved_templates=self.f.approved,
                                                precomputed=self.precomputed())
        self.assertEqual(verified, self.doc)
        self.assertEqual(archive, self.f.archive)

    def test_precomputed_still_requires_fresh_digest_of_compressed_bytes(self):
        name = self.doc['bundle']['archive']
        path = self.f.assets / name
        data = path.read_bytes()
        path.write_bytes(data[:-1] + bytes([data[-1] ^ 0xFF]))
        with self.assertRaisesRegex(ValueError, 'asset hash/bytes mismatch'):
            e.verify_assets(self.approval, registry=self.registry,
                             approved_templates=self.f.approved,
                             precomputed=self.precomputed())

    def test_precomputed_member_map_tamper_is_refused(self):
        precomputed = self.precomputed()
        name = self.doc['bundle']['archive']
        flipped = copy.deepcopy(precomputed[name])
        first = next(iter(flipped))
        flipped[first] = dict(flipped[first], sha256='f' * 64)
        precomputed[name] = flipped
        with self.assertRaisesRegex(ValueError, 'precomputed inventory mismatch'):
            e.verify_assets(self.approval, registry=self.registry,
                             approved_templates=self.f.approved,
                             precomputed=precomputed)

    def test_missing_inventory_entry_falls_back_to_full_inspection(self):
        # Cache-miss fallback: an archive absent from precomputed is fully
        # re-inspected and still must match the approval record.
        precomputed = self.precomputed(skip=self.doc['engine_source_asset']['archive'])
        verified, _ = e.verify_assets(self.approval, registry=self.registry,
                                       approved_templates=self.f.approved,
                                       precomputed=precomputed)
        self.assertEqual(verified, self.doc)

    def test_validate_bound_archive_inventory_skips_first_pass_and_binds_content(self):
        from tools.corresponding_source_identity import validate_bound_archive
        asset = self.doc['engine_source_asset']
        with patch('tools.release_evidence.inspect_archive',
                   side_effect=AssertionError('must not run')):
            identity = validate_bound_archive(self.f.engine_source_asset, commit=self.f.commit,
                                              tree=self.f.tree, asset=asset,
                                              inventory=asset['members'])
        self.assertEqual(identity['source_commit'], self.f.commit)
        self.assertEqual(identity['source_tree'], self.f.tree)

    def test_validate_bound_archive_inventory_still_rejects_tampered_archive(self):
        from tools.corresponding_source_identity import validate_bound_archive
        asset = self.doc['engine_source_asset']
        target = 'crates/server/src/main.rs'
        files = {}
        with tarfile.open(self.f.engine_source_asset, 'r:*') as tar:
            for member in tar:
                if member.isfile():
                    files[member.name] = (member.mode, tar.extractfile(member).read())
        files[target] = (files[target][0], b'fn forged() {}\n')
        with tarfile.open(self.f.engine_source_asset, 'w:gz') as out:
            for name, (mode, data) in sorted(files.items()):
                info = tarfile.TarInfo(name)
                info.mode, info.size = mode, len(data)
                out.addfile(info, io.BytesIO(data))
        with self.assertRaises(ValueError):
            validate_bound_archive(self.f.engine_source_asset, commit=self.f.commit,
                                   tree=self.f.tree, asset=asset, inventory=asset['members'])

    def test_validate_bound_archive_rejects_unlisted_engine_source_member(self):
        from tools.corresponding_source_identity import validate_bound_archive
        asset = self.doc['engine_source_asset']
        files = {}
        with tarfile.open(self.f.engine_source_asset, 'r:*') as tar:
            for member in tar:
                if member.isfile():
                    files[member.name] = (member.mode, tar.extractfile(member).read())
        files['crates/server/src/extra.rs'] = (0o644, b'fn extra() {}\n')
        with tarfile.open(self.f.engine_source_asset, 'w:gz') as out:
            for name, (mode, data) in sorted(files.items()):
                info = tarfile.TarInfo(name)
                info.mode, info.size = mode, len(data)
                out.addfile(info, io.BytesIO(data))
        with self.assertRaisesRegex(ValueError, 'unlisted engine source member'):
            validate_bound_archive(self.f.engine_source_asset, commit=self.f.commit,
                                   tree=self.f.tree, asset=asset, inventory=asset['members'])


if __name__ == '__main__':
    unittest.main()
