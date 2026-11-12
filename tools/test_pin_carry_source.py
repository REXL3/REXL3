"""CPU-only synthetic seals: tools/pin_carry_source.py derives carry pins only from verified sealed evidence."""
import json
import tempfile
import unittest
from pathlib import Path

from tools import pin_carry_source as pin
from tools import release_evidence as e
from tools.test_approve_profile import SyntheticSeal
from tools.test_release_evidence import encoded, sha


class PinCarrySource(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.f = SyntheticSeal(tmp.name)
        self.path, self.doc, self.registry = self.f.published_fixture_model('synthetic', 'synthetic-demo')
        self.key = 'synthetic/synthetic-demo/v1'

    def derive(self):
        return pin.derive_pins(self.f.assets, registry=self.registry, key=self.key, approved_templates=self.f.approved)

    def test_pins_come_from_the_verified_approval(self):
        pins = self.derive()
        self.assertEqual(pins['binary_sha256'], self.doc['identities']['binary_sha256'])
        self.assertEqual(pins['profiles'], {'single': {'receipt': {'sha256': self.doc['profiles']['single']['receipt']['sha256']}}})
        entry = pin.pinned_entry(self.registry['approvals'][self.key], pins)
        self.assertEqual(entry['approval_sha256'], self.registry['approvals'][self.key]['approval_sha256'])
        self.assertEqual(pin.pinned_entry(entry, pins), entry)  # idempotent

    def test_existing_pins_are_never_changed(self):
        pins = self.derive()
        for field, value in (('binary_sha256', 'e' * 64),
                             ('profiles', {'single': {'receipt': {'sha256': 'e' * 64}}})):
            with self.subTest(field):
                entry = dict(self.registry['approvals'][self.key], **{field: value})
                with self.assertRaisesRegex(ValueError, 'already pins another'):
                    pin.pinned_entry(entry, pins)

    def test_unverified_evidence_yields_no_pins(self):
        # Approval bytes that are not the registry's, and a receipt that no longer matches the approval, are refused.
        self.registry['approvals'][self.key]['approval_sha256'] = 'e' * 64
        with self.assertRaisesRegex(ValueError, 'untrusted approval bytes'):
            self.derive()
        self.registry['approvals'][self.key]['approval_sha256'] = sha(self.path.read_bytes())
        evidence = self.f.assets / 'evidence.tar.gz'
        evidence.write_bytes(evidence.read_bytes()[:-8] + b'\0' * 8)
        with self.assertRaises(ValueError):
            self.derive()


if __name__ == '__main__':
    unittest.main()
