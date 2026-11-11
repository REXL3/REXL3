## Bug fixes

- The template allow-list (`runtimes/templates/APPROVED.json`) now accepts a list of approved digests per template version as well as a single digest. Template `v1` lists both the digest the existing trees were sealed against and the digest the v0.1.1 approvals were sealed against, so promotion, carry-source pinning and verification accept the new approvals without re-sealing the existing trees. An unlisted version or digest is still refused.
