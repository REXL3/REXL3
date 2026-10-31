"""assemble_release_notes.py: fragment parsing, template rendering, CHANGELOG entry, refusals."""
import tempfile
import unittest
from pathlib import Path

from tools import assemble_release_notes as n
from tools import release_evidence as e
from tools.test_release_evidence import write, encoded

VERSION = 'v0.1.0'
HEX40 = 'a' * 40
HEX64 = 'b' * 64


def manifest(version=VERSION):
    return {'schema': 'rexl3.release-manifest.v1', 'version': version,
            'engine': {'dev_commit': HEX40, 'dev_tree': HEX40, 'engine_release_tag': 'engine-v0.1.2-4208d848',
                       'artifacts': {'rexl3-serve.tar.gz': HEX64}},
            'models': [{'family': 'qwen3.8', 'model': 'qwen3.8-27b', 'version': 'v1',
                        'runtime_id': 'qwen38-27b-dflash2-exl3-4.00', 'approval_sha256': HEX64}],
            'notes': 'first release'}


class NotesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.template = self.root / 'docs/templates/RELEASE_NOTES.md'
        self.template.parent.mkdir(parents=True)
        self.template.write_text('# REXL3 v{{VERSION}} release notes\n\n## Highlights\n\n{{HIGHLIGHTS}}\n\n'
                                 '## New features and improvements\n\n{{NEW_FEATURES}}\n\n'
                                 '## Supported models\n\n{{SUPPORTED_MODELS}}\n\n## Assets and checksums\n\n{{ASSETS}}\n\n'
                                 '## Provenance\n\n{{PROVENANCE}}\n')
        self.fragments = self.root / 'changes/unreleased'
        self.fragments.mkdir(parents=True)
        write(self.root, 'changes/unreleased/slug.md', b'## Highlights\n- First release.\n## New features and improvements\n- Faster.\n')
        write(self.root, f'releases/{VERSION}/manifest.json', encoded(manifest()))

    def test_assembles_notes_and_changelog(self):
        code = n.assemble(str(self.root), VERSION)
        self.assertEqual(code, 0)
        notes = (self.root / f'releases/{VERSION}/NOTES.md').read_text()
        self.assertIn('First release.', notes)
        self.assertIn('Faster.', notes)
        self.assertIn('qwen3.8/qwen3.8-27b v1', notes)
        self.assertIn('rexl3-serve.tar.gz', notes)
        self.assertIn(HEX40, notes)
        changelog = (self.root / 'CHANGELOG.md').read_text()
        self.assertIn(f'[{VERSION}]', changelog)
        self.assertIn('First release.', changelog)
        self.assertIn('- Models: qwen3.8/qwen3.8-27b v1.', changelog)
        self.assertNotIn('vv1', changelog)

    def test_repeat_assembly_is_idempotent(self):
        n.assemble(str(self.root), VERSION)
        before = (self.root / f'releases/{VERSION}/NOTES.md').read_bytes()
        self.assertEqual(n.assemble(str(self.root), VERSION), 0)
        self.assertEqual((self.root / f'releases/{VERSION}/NOTES.md').read_bytes(), before)
        # a different entry for the same version is refused, never overwritten
        (self.root / 'CHANGELOG.md').write_text('# Changelog\n\n## [v0.1.0] - 2026-10-26\n\ndifferent\n')
        with self.assertRaisesRegex(ValueError, 'different entry'):
            n.assemble(str(self.root), VERSION)

    def test_unknown_heading_is_refused(self):
        write(self.root, 'changes/unreleased/bad.md', b'## Highlights\n- ok.\n## Random\n- no.\n')
        with self.assertRaisesRegex(ValueError, 'unknown heading'):
            n.assemble(str(self.root), VERSION)

    def test_empty_fragment_is_refused(self):
        write(self.root, 'changes/unreleased/empty.md', b'')
        with self.assertRaisesRegex(ValueError, 'empty fragment'):
            n.assemble(str(self.root), VERSION)

    def test_fragment_without_heading_is_refused(self):
        write(self.root, 'changes/unreleased/nohead.md', b'just prose\n')
        with self.assertRaisesRegex(ValueError, 'no recognized ## heading'):
            n.assemble(str(self.root), VERSION)

    def test_duplicate_slug_is_refused(self):
        write(self.root, 'changes/unreleased/slug.md', b'## Highlights\n- one.\n')
        write(self.root, 'changes/unreleased/slug-2.md', b'## Highlights\n- two.\n')
        n.assemble(str(self.root), VERSION)

    def test_manifest_schema_is_enforced(self):
        bad = manifest()
        bad['engine']['dev_commit'] = 'short'
        write(self.root, f'releases/{VERSION}/manifest.json', encoded(bad))
        with self.assertRaises(ValueError):
            n.assemble(str(self.root), VERSION)


if __name__ == '__main__':
    unittest.main()
