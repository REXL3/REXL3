# Security policy

## Reporting a vulnerability

Please report a vulnerability **privately**. Use GitHub's private vulnerability reporting: open this repository's **Security** tab and choose **Report a vulnerability**. If that option is not shown, contact a maintainer (`@am423` or `@vcruz305`) through a private channel and ask for one. Do not open a public issue or pull request for it, and do not post a proof of concept in public.

Say which runtime and version you used (`runtimes/<id>/v<N>`, and the release tag `prod/<id>/v<N>`), what you did, and what happened.

In scope:

- a way to get a runtime tree, a release asset or an `approval.json` accepted by `verify` or by `tools/promote_runtime.py` without a matching maintainer-reviewed approval (a bypass of the registry, the template allow-list, the hash pin, the receipt checks or the tar and path checks);
- a weakness in `.github/workflows/` that lets pull-request content run with more than read access, execute code from the pull request, or merge itself;
- a launcher or script in a released pack that reads or writes outside its documented state directory, runs privileged code it should not, or leaks prompts, responses, environment values or credentials into `diagnostics`;
- a request that crashes or wedges a served model, reads memory or files it should not, or bypasses a documented limit.

## Supported versions

| Version | Supported |
| --- | --- |
| `main` | yes: fixes land here first |
| `runtimes/dsv41-flash-1017-sage-1.62/v1` | yes, until the next version. A fix ships as a new version; a released version is never edited in place. |

There are no backport branches. A fix is a new promoted version, and the old one is withdrawn as described in `docs/PROMOTION.md`.

## Threat model

A served pack is an inference server meant for a trusted network behind a reverse proxy. The server has **no authentication and no TLS** built in: anyone who can reach the port can use the model and consume the GPU. The `v1` profile binds `127.0.0.1` and the launcher refuses a non-loopback single-node address; keep it that way, or put your own proxy in front that terminates TLS, authenticates clients and enforces header, body, connection and time limits. Never expose a cluster, router or admin port beyond a private network.

The model weights are fetched by the launcher from the repository and revision pinned in `model.lock.json` and checked against the recorded file hashes. A pack is read by trusted code and is not hardened against a hostile weight author: do not change the lock to load a model you do not trust.

What this repository's pipeline does protect: a runtime tree in git, and the release that carries the same bytes, must match a maintainer-reviewed approval. Hashes prove the bytes did not change; they do not prove that a benchmark ran. The qualification evidence is produced and reviewed in the dev repository (`docs/PROMOTION.md`). The pipeline needs no signing key and no secret, and `verify` has read-only access.
