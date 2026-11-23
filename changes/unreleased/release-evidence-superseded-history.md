## Bug fixes

- Fixed the promotion verifier refusing an approval whose qualification receipt keeps a superseded attempt. When the qualifier reruns a row because its probe changed, it keeps the earlier attempt and its files in the sealed evidence. The verifier now checks that history (each attempt's bytes, the archive, and that a rerun uses a different probe and passes) and expects those files in the evidence asset, as the sealer does. Receipts without a superseded attempt verify as before. (PR #190)
