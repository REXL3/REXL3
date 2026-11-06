"""release_manifest.py: the carried-forward model entry contract (decision D37)."""
import unittest

from tools.release_manifest import validate_manifest

VERSION = 'v0.1.1'
HEX40 = '0' * 40
HEX64 = 'd' * 64
ENGINE_TAG = 'engine-v0.1.1-11111111'
DEV_COMMIT = '11111111' * 5  # first 8 hex must equal ENGINE_TAG's commit suffix


def base_manifest():
    return {'schema': 'rexl3.release-manifest.v1', 'version': VERSION,
            'engine': {'dev_commit': DEV_COMMIT, 'dev_tree': 'b' * 40,
                       'engine_release_tag': ENGINE_TAG,
                       'artifacts': {'rexl3-serve.tar.gz': HEX64}},
            'models': [{'family': 'qwen3.8', 'model': 'qwen3.8-27b', 'version': 'v1',
                        'runtime_id': 'synthetic-demo', 'approval_sha256': HEX64}],
            'notes': 'carry-forward release'}


def carried_manifest(tier='T0', **changes):
    doc = base_manifest()
    doc['models'][0].update({
        'carried_forward': True,
        'qualified_engine_release_tag': ENGINE_TAG,
        'qualification_tier': tier,
        'scope_diff_base': HEX40,
    })
    doc['models'][0].update(changes)
    return doc


class ManifestSchemaTest(unittest.TestCase):
    def test_plain_manifest_passes_unchanged(self):
        self.assertEqual(validate_manifest(base_manifest(), VERSION), base_manifest())

    def test_carried_forward_t0_is_accepted_and_t1_t2_are_rejected(self):
        # Only T0 is a manifest carry-forward (the prior sealed approval travels as-is); T1/T2
        # re-seal a NEW approval on the new engine build, so they are not manifest carry-forwards.
        validate_manifest(carried_manifest('T0'), VERSION)
        for tier in ('T1', 'T2'):
            with self.subTest(tier=tier), self.assertRaisesRegex(ValueError, 'invalid qualification tier'):
                validate_manifest(carried_manifest(tier), VERSION)

    def test_missing_qualified_engine_release_tag_is_rejected(self):
        doc = carried_manifest()
        del doc['models'][0]['qualified_engine_release_tag']
        with self.assertRaisesRegex(ValueError, 'incomplete carried-forward model entry'):
            validate_manifest(doc, VERSION)

    def test_missing_qualification_tier_is_rejected(self):
        doc = carried_manifest()
        del doc['models'][0]['qualification_tier']
        with self.assertRaisesRegex(ValueError, 'incomplete carried-forward model entry'):
            validate_manifest(doc, VERSION)

    def test_missing_scope_diff_base_is_rejected(self):
        doc = carried_manifest()
        del doc['models'][0]['scope_diff_base']
        with self.assertRaisesRegex(ValueError, 'incomplete carried-forward model entry'):
            validate_manifest(doc, VERSION)

    def test_t3_carried_forward_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'invalid qualification tier'):
            validate_manifest(carried_manifest('T3'), VERSION)

    def test_malformed_qualified_engine_release_tag_is_rejected(self):
        for tag in ('engine-v0.1.1', 'engine-v0.1.1-1111111', 'engine-v0.1.1-1111111g', 'engine-v0.1.1-11111111-extra'):
            with self.subTest(tag=tag), self.assertRaisesRegex(ValueError, 'invalid qualified engine release tag'):
                validate_manifest(carried_manifest(qualified_engine_release_tag=tag), VERSION)

    def test_carried_forward_false_with_carry_forward_fields_is_rejected(self):
        doc = carried_manifest()
        doc['models'][0]['carried_forward'] = False
        with self.assertRaisesRegex(ValueError, 'carry-forward fields require carried_forward: true'):
            validate_manifest(doc, VERSION)

    def test_carried_forward_null_is_rejected(self):
        doc = carried_manifest()
        doc['models'][0]['carried_forward'] = None
        with self.assertRaisesRegex(ValueError, 'carried_forward must be a boolean'):
            validate_manifest(doc, VERSION)

    def test_carried_forward_false_without_carry_fields_is_accepted(self):
        doc = base_manifest()
        doc['models'][0]['carried_forward'] = False
        self.assertEqual(validate_manifest(doc, VERSION), doc)

    def test_carry_forward_fields_without_carried_forward_are_rejected(self):
        doc = base_manifest()
        doc['models'][0]['qualified_engine_release_tag'] = ENGINE_TAG
        doc['models'][0]['qualification_tier'] = 'T0'
        doc['models'][0]['scope_diff_base'] = HEX40
        with self.assertRaisesRegex(ValueError, 'carry-forward fields require carried_forward: true'):
            validate_manifest(doc, VERSION)

    def test_carried_forward_non_boolean_is_rejected(self):
        for value in ('yes', 1, 0, []):
            with self.subTest(value=value):
                doc = carried_manifest()
                doc['models'][0]['carried_forward'] = value
                with self.assertRaises(ValueError):
                    validate_manifest(doc, VERSION)

    def test_invalid_scope_diff_base_is_rejected(self):
        for base in ('abc', '1' * 39, '1' * 41, None):
            with self.subTest(base=base), self.assertRaises(ValueError):
                validate_manifest(carried_manifest(scope_diff_base=base), VERSION)

    def test_duplicate_family_model_entries_are_rejected(self):
        for extra in ({'version': 'v2'}, {'carried_forward': True,
                                           'qualified_engine_release_tag': 'engine-v0.1.0-22222222',
                                           'qualification_tier': 'T0', 'scope_diff_base': HEX40}):
            with self.subTest(extra=extra):
                doc = base_manifest()
                doc['models'].append(dict(doc['models'][0], **extra))
                with self.assertRaisesRegex(ValueError, r'duplicate \(family, model\) entry'):
                    validate_manifest(doc, VERSION)

    def test_engine_dev_commit_prefix_must_match_the_release_tag_commit(self):
        doc = base_manifest()
        doc['engine']['dev_commit'] = 'a' * 40
        with self.assertRaisesRegex(ValueError, 'dev_commit prefix must match the engine release tag commit'):
            validate_manifest(doc, VERSION)


if __name__ == '__main__':
    unittest.main()
