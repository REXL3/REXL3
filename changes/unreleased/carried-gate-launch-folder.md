## Bug fixes

- Production verification now accepts a carried gate whose source run was launched from another folder. This only applies when the receipt pins the source run's launch observation and both launches are the same profile choice once absolute folder paths are taken out; any other launch difference is still refused. Bit-identity records also accept the reviewed earlier runtime id. This matches the development sealer (REXL3/REXL3-dev#343, #344), so requalified receipts composed there verify in production. PR #135.
