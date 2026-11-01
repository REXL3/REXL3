"""CPU-only synthetic seals with real disposable Git objects; not native qualification."""
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import struct
import sys
import tarfile
import tempfile
import unittest
from tools import approve_profile as a
from tools import release_evidence as e
from tools import runtime_template_digest as td
from tools import build_corresponding_source as exporter
from tools.test_release_evidence import GATES, encoded, refusal_detail, sha, write


class SyntheticSeal:
    def __init__(self, root, tp2=False, images=True, alt_profile=None, edit_lock=None):
        """``alt_profile``: overrides for a second single-node choice named ``alt`` (for example its own
        ``pack`` role and ``runtime_id``), with its own receipt; its pack role is added to the lock.
        ``edit_lock``: a callable applied to the lock's ``packs`` before anything is bound to them."""
        self.images = images
        self.root = Path(root)
        self.repo = self.root / 'repo'
        self.repo.mkdir()
        self.assets = self.root / 'assets'
        self.assets.mkdir()
        self.template = self.repo / 'runtimes/template-v1'
        write(self.template, 'launch.sh', b'#!/bin/sh\n# SYNTHETIC TEST ONLY\n')
        # Explicitly synthetic build inputs, but real Git blobs/commit/tree.
        # Never compile or execute these; the protected exporter is NOT imported
        # from the disposable repository (nor from its source archive).
        engine = {
            'Cargo.toml': b'[workspace]\nmembers = ["crates/server", "crates/cuda", "crates/cluster"]\n[workspace.package]\nlicense = "AGPL-3.0-only"\n',
            'Cargo.lock': b'version = 4\n',
            'rust-toolchain.toml': b'[toolchain]\nchannel = "1.89.0"\n',
            '.cargo/config.toml': b'# SYNTHETIC TEST ONLY\n',
            'LICENSE': b'SYNTHETIC AGPL license fixture; not a license artifact\n',
            'NOTICE': b'SYNTHETIC notice\n',
            'THIRD_PARTY_NOTICES.md': b'SYNTHETIC third-party notice\n',
            'crates/server/Cargo.toml': b'[package]\nname = "gb-server"\nversion = "0.0.0"\n',
            'crates/server/src/main.rs': b'const TEST: &str = include_str!("../../../docs/examples/compile.json");\nfn main() {}\n',
            'crates/server/helper.sh': b'#!/bin/sh\n# SYNTHETIC; never executed\n',
            'crates/cluster/Cargo.toml': b'[package]\nname = "gb-cluster"\nversion = "0.0.0"\n',
            'crates/cluster/src/main.rs': b'fn main() {}\n',
            'crates/cuda/Cargo.toml': b'[package]\nname = "gb-cuda"\nversion = "0.0.0"\n',
            'crates/cuda/src/lib.rs': b'// SYNTHETIC native interface\n',
            'crates/cuda/build.rs': b'fn main() {}\n',
            'crates/cuda/csrc/kernel.cu': b'// SYNTHETIC native source\n',
            'crates/cuda/csrc/header.h': b'/* SYNTHETIC native header */\n',
            'third_party/toks/tokenizer.c': b'/* SYNTHETIC vendor input */\n',
            'third_party/toks/LICENSE': b'SYNTHETIC vendor notice\n',
            'testdata/fixture.json': b'{}\n',
            'docs/examples/compile.json': b'{}\n',
            exporter.TEMPLATE: (Path(__file__).resolve().parents[1] / exporter.TEMPLATE).read_bytes(),
            'tools/build_corresponding_source.py': b'# SYNTHETIC; never imported\n',
            'tools/test_build_corresponding_source.py': b'# SYNTHETIC; never imported\n',
        }
        # The exporter requires an exact-content committed review for every enabled Cargo build script.
        build_script = engine['crates/cuda/build.rs']
        engine[exporter.INPUTS] = encoded({'schema': 'rexl3.source-inputs.v1', 'files': [], 'build_scripts': [{
            'path': 'crates/cuda/build.rs', 'sha256': hashlib.sha256(build_script).hexdigest(),
            'inputs': [], 'include_dirs': [], 'external_inputs': [],
            'reason': 'Synthetic build script reads nothing; reviewed for this fixture only.'}]})
        for name, data in engine.items():
            write(self.repo, name, data)
        (self.repo / 'crates/server/helper.sh').chmod(0o755)
        self.git('init', '-q')
        self.git('add', '.')
        self.git('commit', '-qm', 'synthetic template')
        template_commit = self.git('rev-parse', 'HEAD')
        digest = td.digest_tree(td.read_tree(self.template))
        write(self.template, 'TEMPLATE.json', encoded(dict(schema=td.SCHEMA, version='v1', digest=digest,
              approved_commit=template_commit, approved_by='synthetic', approved_at='synthetic')))
        config = b'{"synthetic":true}\n'
        write(self.repo, 'example.json', config)
        self.git('add', '.')
        self.git('commit', '-qm', 'synthetic source')
        upstream = self.git('rev-parse', 'HEAD')
        self.profile = dict(schema='rexl3.runtime-profile.v1', state='final', runtime_id='synthetic-demo',
            source={'repository': 'REXL3/REXL3-dev', 'commit': upstream, 'path': 'example.json', 'sha256': sha(config)},
            target_repo='synthetic/model', default_profile='single', profiles={'single': {
                'kind': 'single', 'argv': ['--max-model-len', '1048576'] + ([] if images else ['--no-vision']),
                'env': {}, 'files': {},
                'capabilities': {key: key != 'audio' for key in ('text', 'sse', 'reasoning', 'tools', 'images', 'video', 'audio', 'json_schema', 'prefix_cache')}}})
        self.profile['profiles']['single']['capabilities']['images'] = images
        if alt_profile is not None:
            self.profile['profiles']['alt'] = dict(copy.deepcopy(self.profile['profiles']['single']),
                                                   argv=['--max-model-len', '262144'] + ([] if images else ['--no-vision']),
                                                   **alt_profile)
        tp2_role = None
        if tp2:
            from tools.test_release_profile_contract import native_tp2_fixture
            tp_choice, native_cluster, tp_observation = native_tp2_fixture(images)
            # The explicit vision state may differ from the tracked cluster file: bind the exact bytes used.
            cluster = encoded(native_cluster)
            tp_choice['files'] = {tp_choice['cluster_file']: sha(cluster)}
            self.profile['profiles']['tp2'] = tp_choice
            tp2_role = tp_choice.get('pack', 'target')
        self.models = dict(schema='rexl3.model-lock.v1', packs={'target': {'repo': 'synthetic/model',
            'revision': 'a' * 40, 'fingerprint': 'sha256:' + 'b' * 64, 'files': {'config.json': {'sha256': sha(config), 'bytes': len(config)}}}})
        self.models['packs']['draft'] = {'repo': 'synthetic/draft',
            'revision': 'c' * 40, 'fingerprint': 'sha256:' + 'd' * 64,
            'files': {'config.json': {'sha256': sha(config), 'bytes': len(config)}}}
        if tp2_role and tp2_role != 'target':
            # The synthetic lock must carry every role the adopted real TP2 choice references.
            self.models['packs'][tp2_role] = copy.deepcopy(self.models['packs']['target'])
        alt_role = (alt_profile or {}).get('pack')
        if alt_role and alt_role not in self.models['packs']:
            self.models['packs'][alt_role] = dict(copy.deepcopy(self.models['packs']['target']), repo='synthetic/alt-model')
        if edit_lock is not None:
            edit_lock(self.models['packs'])
        self.source = {'profile.json': encoded(self.profile), 'model.lock.json': encoded(self.models),
                       'launch.sh': (self.template / 'launch.sh').read_bytes()}
        if tp2:
            self.source[tp_choice['cluster_file']] = cluster
        for name in ('profile.json', 'model.lock.json'):
            write(self.repo, 'runtimes/models/synthetic-demo/' + name, self.source[name])
        self.git('add', '.')
        self.git('commit', '-qm', 'synthetic profile')
        self.commit = self.git('rev-parse', 'HEAD')
        self.tree = self.git('rev-parse', 'HEAD^{tree}')
        self.engine_source_asset = self.assets / 'synthetic-demo-1-engine-source.tar.gz'
        self.engine_source_asset.write_bytes(exporter.archive_bytes(exporter.inspect_source(self.repo, self.commit)))
        self.source['source.manifest.json'] = encoded(dict(schema='rexl3.generated-source.v1',
            template={'version': 'v1', 'digest': digest}, files=self.inventory(self.source)))
        self.source_asset = self.assets / 'synthetic-demo-1-source.tar.gz'
        self.tar(self.source_asset, self.source)
        # Structurally ELF-shaped, explicitly synthetic; NEVER executed/qualified.
        synthetic_elf = bytearray(64)
        synthetic_elf[:7] = b'\x7fELF\x02\x01\x01'
        struct.pack_into('<HH', synthetic_elf, 16, 2, 183)
        self.bundle = dict(self.source, **{'bin/rexl3-serve': bytes(synthetic_elf) + b'SYNTHETIC TEST ONLY'})
        if tp2:
            self.bundle['bin/rexl3-cluster'] = self.bundle['bin/rexl3-serve']
        self.release = dict(schema='rexl3.release.v1', runtime_id='synthetic-demo', target_repo='synthetic/model',
            production_commit=self.commit, production_tree=self.tree, version='0.1.2',
            platform={'system': 'Linux', 'machine': 'aarch64', 'gpu_compute_capability': '12.1'},
            native_version_fields={'commit': self.commit, 'version': '0.1.2', 'system_fingerprint': 'fp_synthetic_test',
                'nvcc': '13.0.88', 'cuda_arch': 'sm_121', 'kernel_hash': '1' * 16, 'cudart': '13.0',
                'cuda_driver': '13.0', 'cublas': '13.1.1'},
            files=self.inventory(self.bundle), profile_sha256=sha(self.source['profile.json']),
            model_lock_sha256=sha(self.source['model.lock.json']))
        self.bundle['release.lock.json'] = encoded(self.release)
        self.archive = self.assets / 'synthetic-demo-1.tar.gz'
        self.tar(self.archive, self.bundle)
        caps = self.profile['profiles']['single']['capabilities']
        self.resolved = {'max_context': 1048576, 'capabilities': caps, 'ranks': [
            {'rank': 0, 'argv': ['synthetic-serve', '--max-model-len', '1048576'] + ([] if images else ['--no-vision']), 'env': {}}]}
        identity = dict(runtime_id='synthetic-demo', profile='single', pack='target', engine_commit=self.commit, engine_tree=self.tree,
            binary_sha256=sha(self.bundle['bin/rexl3-serve']), profile_sha256=sha(self.source['profile.json']),
            model_lock_sha256=sha(self.source['model.lock.json']), release_lock_sha256=sha(self.bundle['release.lock.json']),
            effective_profile_digest='sha256:' + sha(json.dumps(self.resolved, sort_keys=True, separators=(',', ':')).encode()),
            effective_profile_source='owned-launcher-observation', effective_max_context=1048576,
            capabilities=caps, topology={'ranks': 1})
        observation = dict(schema='rexl3.launcher-observation.v1', **identity, resolved_config=self.resolved,
            ranks=[dict(self.resolved['ranks'][0], pid=123, start_time_ticks=456, hostname='synthetic-host', ready=True,
                        **{k: identity[k] for k in ('binary_sha256', 'engine_commit', 'engine_tree')})])
        write(self.assets, 'identity.json', encoded(observation))
        self.receipt = dict(schema=e.RECEIPT_SCHEMA, **identity, gates={}, identity_receipt={'path': 'identity.json', 'sha256': sha(encoded(observation))})
        for gate in GATES:
            case = {'name': 'synthetic', 'status': 'PASS', 'modality': 'image' if gate == 'vision' else 'system', 'expectation': 'supported'}
            extra = []
            if gate == 'vision' and not images:
                detail = write(self.assets, 'vision-detail.json', encoded(refusal_detail()))
                extra = [{'path': detail.name, 'sha256': sha(detail.read_bytes())}]
                case.update(name='image_refused', expectation='refused', result=extra[0])
            raw = dict(schema='rexl3.gate-evidence.v1', gate=gate, **identity, cases=[case])
            write(self.assets, gate + '.json', encoded(raw))
            self.receipt['gates'][gate] = dict(status='PASS', passed=1, total=1, failed=0, skipped=0,
                evidence=[{'path': gate + '.json', 'sha256': sha(encoded(raw))}, *extra])
        self.receipt_path = write(self.assets, 'qualification.json', encoded(self.receipt))
        self.kwargs = dict(runtime_id='synthetic-demo', commit=self.commit, tree=self.tree, template_dir=self.template,
            archive=self.archive, source_asset=self.source_asset, engine_source_asset=self.engine_source_asset,
            receipts=[self.receipt_path], evidence_root=self.assets,
            repo=self.repo, version=1, approved_by='synthetic-test')
        if tp2:
            second = copy.deepcopy(self.receipt)
            resolved = copy.deepcopy(tp_observation['resolved_config'])
            second.update(profile='tp2', runtime_id=tp_choice.get('runtime_id', 'synthetic-demo'),
                          pack=tp_choice.get('pack', 'target'), topology={'ranks': 2},
                          capabilities=tp_choice['capabilities'],
                          effective_profile_digest=tp_observation['effective_profile_digest'])
            identity = {key: second[key] for key in e.IDENTITY_KEYS}
            observed = dict(schema='rexl3.launcher-observation.v1', **identity, resolved_config=resolved,
                ranks=[dict(row, pid=123 + row['rank'], start_time_ticks=456, hostname='synthetic-host-' + str(row['rank']), ready=True,
                            **{key: identity[key] for key in ('binary_sha256', 'engine_commit', 'engine_tree')}) for row in resolved['ranks']])
            path = write(self.assets, 'tp2/identity.json', encoded(observed))
            second['identity_receipt'] = {'path': 'tp2/identity.json', 'sha256': sha(path.read_bytes())}
            for gate in GATES:
                raw = e.load_json(self.assets / (gate + '.json'))
                raw.update(identity)
                extra = []
                if raw['cases'][0].get('result'):  # vision-off refusal detail travels with the tp2 receipt
                    detail = write(self.assets, 'tp2/vision-detail.json', (self.assets / 'vision-detail.json').read_bytes())
                    extra = [{'path': 'tp2/vision-detail.json', 'sha256': sha(detail.read_bytes())}]
                    raw['cases'][0]['result'] = extra[0]
                path = write(self.assets, 'tp2/' + gate + '.json', encoded(raw))
                second['gates'][gate]['evidence'] = [{'path': 'tp2/' + gate + '.json', 'sha256': sha(path.read_bytes())}, *extra]
            self.kwargs['receipts'].append(write(self.assets, 'tp2/qualification.json', encoded(second)))
        if alt_profile is not None:
            self.kwargs['receipts'].append(self._alt_receipt())
        self.approved = {'v1': digest}

    def _alt_receipt(self):
        """The second single-node choice's receipt: its own identity (runtime_id, pack, context), observation and gates."""
        choice = self.profile['profiles']['alt']
        context = int(choice['argv'][choice['argv'].index('--max-model-len') + 1])
        resolved = {'max_context': context, 'capabilities': choice['capabilities'], 'ranks': [
            {'rank': 0, 'argv': ['synthetic-serve'] + choice['argv'], 'env': {}}]}
        second = copy.deepcopy(self.receipt)
        second.update(profile='alt', runtime_id=choice.get('runtime_id', 'synthetic-demo'), pack=choice.get('pack', 'target'),
                      capabilities=choice['capabilities'], effective_max_context=context,
                      effective_profile_digest='sha256:' + sha(json.dumps(resolved, sort_keys=True, separators=(',', ':')).encode()))
        identity = {key: second[key] for key in e.IDENTITY_KEYS}
        observed = dict(schema='rexl3.launcher-observation.v1', **identity, resolved_config=resolved,
            ranks=[dict(resolved['ranks'][0], pid=124, start_time_ticks=456, hostname='synthetic-host', ready=True,
                        **{key: identity[key] for key in ('binary_sha256', 'engine_commit', 'engine_tree')})])
        path = write(self.assets, 'alt/identity.json', encoded(observed))
        second['identity_receipt'] = {'path': 'alt/identity.json', 'sha256': sha(path.read_bytes())}
        for gate in GATES:
            raw = e.load_json(self.assets / (gate + '.json'))
            raw.update(identity)
            extra = []
            if raw['cases'][0].get('result'):  # vision-off refusal detail travels with this receipt
                detail = write(self.assets, 'alt/vision-detail.json', (self.assets / 'vision-detail.json').read_bytes())
                extra = [{'path': 'alt/vision-detail.json', 'sha256': sha(detail.read_bytes())}]
                raw['cases'][0]['result'] = extra[0]
            path = write(self.assets, 'alt/' + gate + '.json', encoded(raw))
            second['gates'][gate]['evidence'] = [{'path': 'alt/' + gate + '.json', 'sha256': sha(path.read_bytes())}, *extra]
        self.alt_receipt_path = write(self.assets, 'alt/qualification.json', encoded(second))
        return self.alt_receipt_path

    def git(self, *args):
        return subprocess.run(['git', '-C', str(self.repo), '-c', 'user.name=synthetic-test', '-c',
                               'user.email=synthetic@example.invalid', *args], check=True, capture_output=True, text=True).stdout.strip()

    @staticmethod
    def inventory(files):
        return {k: {'sha256': sha(v), 'bytes': len(v), 'mode': 0o755 if k.endswith('.sh') or k.startswith('bin/') else 0o644}
                for k, v in files.items()}

    @classmethod
    def tar(cls, path, files):
        with tarfile.open(path, 'w:gz') as out:
            for name, data in sorted(files.items()):
                info = tarfile.TarInfo(name)
                info.size = len(data)
                info.mode = cls.inventory({name: data})[name]['mode']
                out.addfile(info, io.BytesIO(data))

    def seal(self, **changes):
        return a.build_approval(**(self.kwargs | changes))

    def published_fixture(self):
        doc = self.seal()
        e.write_evidence_archive(self.assets, doc['evidence_asset']['members'], self.assets / 'evidence.tar.gz')
        path = write(self.assets, 'approval.json', encoded(doc))
        registry = {'schema': 'rexl3.approval-registry.v1', 'approvals': {'synthetic-demo/v1': {
            'approval_sha256': sha(path.read_bytes()), 'source_commit': self.commit, 'source_tree': self.tree}}}
        return path, doc, registry

    def published_fixture_model(self, family, model):
        """A fixture in the prod family/model layout: registry key <family>/<model>/v<N> + runtime_id."""
        path, doc, registry = self.published_fixture()
        version = doc['version']
        old = f'synthetic-demo/v{version}'
        entry = registry['approvals'].pop(old)
        entry['runtime_id'] = doc['runtime_id']
        registry['approvals'][f'{family}/{model}/v{version}'] = entry
        return path, doc, registry

    def model_doc(self, family, model, **overrides):
        """A valid MODEL.json doc for the synthetic fixture."""
        from tools import model_layout as m
        doc = dict(schema=m.MODEL_SCHEMA, family=family, model=model, display_name='Synthetic Demo',
                   architectures=['SyntheticForConditionalGeneration'], engine_family='synthetic',
                   runtime_id='synthetic-demo', pack={'format': 'exl3', 'bits': '4.00'})
        doc.update(overrides)
        return doc


class ApprovalTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.f = SyntheticSeal(self.tmp.name)

    def test_seal_measures_all_assets_and_receipts(self):
        doc = self.f.seal()
        self.assertEqual(doc['schema'], 'rexl3.profile-approval.v2')
        self.assertEqual(doc['bundle'], e.asset_record(self.f.archive))
        self.assertEqual(doc['source_asset'], e.asset_record(self.f.source_asset))
        self.assertEqual(doc['engine_source_asset'], e.asset_record(self.f.engine_source_asset))
        self.assertEqual(doc['profiles']['single']['receipt']['sha256'], sha(self.f.receipt_path.read_bytes()))
        self.assertEqual(doc['identities']['profile_sha256'], sha(self.f.source['profile.json']))

    def test_only_real_commit_tree_pairs(self):
        for changes in ({'commit': 'f' * 40}, {'tree': 'e' * 40}, {'commit': 'HEAD'}, {'commit': self.f.tree}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.f.seal(**changes)

    def test_template_bytes_and_approval_commit_resolve(self):
        path = self.f.template / 'TEMPLATE.json'
        doc = json.loads(path.read_bytes())
        doc['approved_commit'] = 'f' * 40
        path.write_bytes(encoded(doc))
        with self.assertRaises(ValueError):
            self.f.seal()

    def test_template_mutation_rejected(self):
        (self.f.template / 'launch.sh').write_bytes(b'tampered')
        with self.assertRaises(ValueError):
            self.f.seal()

    def test_missing_duplicate_receipts_rejected(self):
        for receipts in ([], [self.f.receipt_path, self.f.receipt_path]):
            with self.subTest(receipts=receipts), self.assertRaises(ValueError):
                self.f.seal(receipts=receipts)

    def test_strings_are_not_receipts(self):
        self.f.receipt_path.write_bytes(encoded({'doctor': 'FAIL', 'capability': 'PASS', 'exactness': '15/15'}))
        with self.assertRaises(ValueError):
            self.f.seal()

    def test_manifest_release_model_and_profile_drift(self):
        for name in ('source.manifest.json', 'release.lock.json', 'profile.json', 'model.lock.json'):
            with self.subTest(name=name):
                bundle = copy.deepcopy(self.f.bundle)
                bundle[name] += b' '
                self.f.tar(self.f.archive, bundle)
                with self.assertRaises(ValueError):
                    self.f.seal()

    def _reseal_with_release(self, **changes):
        release = copy.deepcopy(self.f.release)
        native = changes.pop('native', {})
        release.update(changes)
        release['native_version_fields'].update(native)
        bundle = copy.deepcopy(self.f.bundle)
        bundle['release.lock.json'] = encoded(release)
        self.f.tar(self.f.archive, bundle)
        return release

    def test_release_version_must_be_the_native_engine_version_not_the_promotion_number(self):
        # The promotion number (1 for approve-<id>-1) used to be demanded here, while the tester's installer
        # demands release.version == native engine version: no release could pass both.
        self._reseal_with_release(version='1')
        with self.assertRaisesRegex(ValueError, 'native engine version'):
            self.f.seal()

    def test_release_version_that_differs_from_the_native_version_is_refused(self):
        self._reseal_with_release(version='0.1.2', native={'version': '0.1.3'})
        with self.assertRaisesRegex(ValueError, 'native engine version'):
            self.f.seal()

    def test_trusted_registry_is_not_self_authorized(self):
        path, doc, registry = self.f.published_fixture()
        self.assertEqual(e.verify_assets(path, registry=registry, approved_templates=self.f.approved)[0], doc)
        with self.assertRaises(ValueError):
            e.verify_assets(path, registry={'schema': 'rexl3.approval-registry.v1', 'approvals': {}}, approved_templates=self.f.approved)
        doc['approved_by'] = 'forged'
        path.write_bytes(encoded(doc))
        with self.assertRaises(ValueError):
            e.verify_assets(path, registry=registry, approved_templates=self.f.approved)

    def test_durable_cli_no_publish_and_exact_retry(self):
        output = Path(self.tmp.name) / 'durable'
        command = [sys.executable, str(Path(a.__file__).resolve()), '--runtime-id', 'synthetic-demo',
            '--target-commit', self.f.commit, '--tree', self.f.tree, '--repo', str(self.f.repo), '--version', '1',
            '--template-dir', str(self.f.template), '--archive', str(self.f.archive), '--source-asset', str(self.f.source_asset),
            '--engine-source-asset', str(self.f.engine_source_asset),
            '--receipt', str(self.f.receipt_path), '--evidence-root', str(self.f.assets), '--approved-by', 'synthetic',
            '--out-dir', str(output), '--no-publish']
        missing_engine = list(command)
        index = missing_engine.index('--engine-source-asset')
        del missing_engine[index:index + 2]
        refused = subprocess.run(missing_engine, capture_output=True, text=True)
        self.assertEqual(refused.returncode, 2)
        self.assertIn('--engine-source-asset', refused.stderr)
        self.assertFalse(output.exists())
        run = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        sealed = (output / 'approval.json').read_bytes()
        run = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        self.assertEqual((output / 'approval.json').read_bytes(), sealed)
        self.assertFalse((self.f.repo / 'approval.json').exists())
        self.assertIn('--target ' + self.f.commit, run.stdout)
        self.assertIn('evidence.tar.gz', run.stdout)
        self.assertIn(self.f.engine_source_asset.name, run.stdout)
        self.assertEqual((output / self.f.engine_source_asset.name).read_bytes(), self.f.engine_source_asset.read_bytes())

    def test_transport_metadata_is_strict_not_float_or_boolean(self):
        path, doc, registry = self.f.published_fixture()
        doc['bundle']['members']['profile.json']['mode'] = 420.0
        path.write_bytes(encoded(doc))
        registry['approvals']['synthetic-demo/v1']['approval_sha256'] = sha(path.read_bytes())
        with self.assertRaises(ValueError):
            e.verify_assets(path, registry=registry, approved_templates=self.f.approved)


class PerProfilePackTest(unittest.TestCase):
    """One runtime entry, two profiles: each profile pins its own lock role and runtime id (dev #159)."""

    ALT = dict(pack='alt', runtime_id='synthetic-alt')

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def fixture(self, sub='alt', **kwargs):
        root = Path(self.tmp.name) / sub
        root.mkdir()
        return SyntheticSeal(str(root), alt_profile=kwargs.pop('alt_profile', self.ALT), **kwargs)

    def rebind(self, fixture, path, approval, registry, key='synthetic-demo/v1'):
        path.write_bytes(encoded(approval))
        registry['approvals'][key]['approval_sha256'] = sha(path.read_bytes())

    def test_each_profile_seals_and_verifies_under_its_own_runtime_id_and_pack(self):
        fixture = self.fixture()
        path, approval, registry = fixture.published_fixture()
        identities = {name: approval['profiles'][name]['identity'] for name in ('single', 'alt')}
        self.assertEqual((identities['single']['runtime_id'], identities['single']['pack']), ('synthetic-demo', 'target'))
        self.assertEqual((identities['alt']['runtime_id'], identities['alt']['pack']), ('synthetic-alt', 'alt'))
        self.assertEqual(approval['runtime_id'], 'synthetic-demo')
        self.assertEqual(e.verify_assets(path, registry=registry, approved_templates=fixture.approved)[0], approval)

    def test_the_prod_family_model_registry_key_verifies_a_per_profile_approval(self):
        fixture = self.fixture()
        path, approval, registry = fixture.published_fixture_model('qwen3.8', 'qwen3.8-27b')
        verified, _ = e.verify_assets(path, registry=registry, approved_templates=fixture.approved,
                                      key='qwen3.8/qwen3.8-27b/v1')
        self.assertEqual(verified, approval)

    def test_an_identity_bound_to_the_top_level_runtime_id_or_target_pack_is_drift(self):
        for field, value in (('runtime_id', 'synthetic-demo'), ('pack', 'target')):
            with self.subTest(field=field):
                fixture = self.fixture(field)
                path, approval, registry = fixture.published_fixture()
                approval['profiles']['alt']['identity'][field] = value
                self.rebind(fixture, path, approval, registry)
                with self.assertRaisesRegex(ValueError, 'approved profile identity drift'):
                    e.verify_assets(path, registry=registry, approved_templates=fixture.approved)

    def test_a_receipt_for_another_runtime_id_does_not_seal(self):
        fixture = self.fixture()
        receipt = e.load_json(fixture.alt_receipt_path)
        receipt['runtime_id'] = 'synthetic-demo'
        fixture.alt_receipt_path.write_bytes(encoded(receipt))
        with self.assertRaises(ValueError):
            fixture.seal()

    def test_a_profile_naming_a_role_the_lock_lacks_or_the_draft_role_is_refused(self):
        cases = {
            'role missing from the lock': dict(alt_profile=self.ALT, edit_lock=lambda packs: packs.pop('alt')),
            'draft role as a serving pack': dict(alt_profile=dict(pack='draft', runtime_id='synthetic-alt')),
        }
        for name, kwargs in cases.items():
            with self.subTest(case=name):
                fixture = self.fixture(name.replace(' ', '-'), **kwargs)
                with self.assertRaisesRegex(ValueError, 'unknown pack role'):
                    fixture.seal()

    def test_a_lock_without_a_target_pack_is_refused(self):
        fixture = self.fixture('no-target', edit_lock=lambda packs: packs.pop('target'))
        with self.assertRaisesRegex(ValueError, 'model lock needs a target pack'):
            fixture.seal()

    def test_an_invalid_runtime_id_override_is_refused(self):
        fixture = self.fixture('bad-id', alt_profile=dict(pack='alt', runtime_id='Not An Id'))
        with self.assertRaisesRegex(ValueError, 'invalid runtime id override'):
            fixture.seal()


class VisionOffApprovalTest(unittest.TestCase):
    """Sealing and trusted verification for a release that ships with images=false."""

    def fixture(self, images=False, **kwargs):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        return SyntheticSeal(tmp.name, images=images, **kwargs)

    def test_vision_off_seal_and_trusted_verification_need_refusal_evidence(self):
        fixture = self.fixture()
        path, doc, registry = fixture.published_fixture()
        self.assertIs(doc['profiles']['single']['identity']['capabilities']['images'], False)
        self.assertEqual(e.verify_assets(path, registry=registry, approved_templates=fixture.approved)[0], doc)

    def test_vision_off_seal_rejects_positive_or_missing_image_evidence(self):
        for mode in ('supported_case', 'text_only', 'unmeasured_refusal'):
            fixture = self.fixture()
            raw_path = fixture.assets / 'vision.json'
            raw = e.load_json(raw_path)
            if mode == 'supported_case':
                raw['cases'][0].update(expectation='supported')
            elif mode == 'text_only':
                raw['cases'][0].update(modality='text')
            else:
                raw['cases'][0].pop('result')
            raw_path.write_bytes(encoded(raw))
            fixture.receipt['gates']['vision']['evidence'][0]['sha256'] = sha(raw_path.read_bytes())
            fixture.receipt_path.write_bytes(encoded(fixture.receipt))
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                fixture.seal()

    def test_vision_off_receipt_for_a_vision_on_profile_is_rejected(self):
        # Refusal-only evidence cannot seal a profile that claims images=true.
        fixture = self.fixture(images=True)
        raw_path = fixture.assets / 'vision.json'
        raw = e.load_json(raw_path)
        raw['cases'][0].update(expectation='refused', name='image_refused')
        raw_path.write_bytes(encoded(raw))
        fixture.receipt['gates']['vision']['evidence'][0]['sha256'] = sha(raw_path.read_bytes())
        fixture.receipt_path.write_bytes(encoded(fixture.receipt))
        with self.assertRaisesRegex(ValueError, 'positive supported image evidence required'):
            fixture.seal()
