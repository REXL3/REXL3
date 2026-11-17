## Bug fixes

- Fixed the release-cut verification to fetch a newly uploaded draft by its release ID before publication, avoiding GitHub's 404 response for draft lookups by tag. This keeps the maintainer release ceremony from stopping after assets have uploaded successfully. Validated against a simulated API; no live release was created or modified. (PR #171)
