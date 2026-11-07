# TLS fixtures of the HTTPS image-ingress tests

`cert.pem` and `key.pem` are a **generated, self-signed test pair**, and `key.pem` is **not a secret**: it protects nothing and is
published on purpose.

- An RSA-2048 key and a SHA-256 certificate. Subject and issuer are both `CN=ingress.test`; the only subject alternative name is the
  DNS name `ingress.test` (a reserved, non-resolvable name). The certificate is valid from 2026-10-05 to 2036-10-02.
- The tests in `crates/server/src/vision_ingress.rs` (the `#[cfg(test)]` module only) include both files with `include_bytes!`: they start a
  local TLS listener that presents this pair, and trust `cert.pem` as the only root of the client under test, to exercise the
  HTTPS-only, validated-dial image ingress without a network or a real certificate authority.
- Nothing in the engine, the server binary or any deployment uses this pair. Do not reuse it, do not install it in a trust store,
  and do not generate a production key into this directory.

To replace it (for example when it expires), generate a new self-signed pair for `ingress.test`
(`openssl req -x509 -newkey rsa:2048 -sha256 -nodes -subj /CN=ingress.test -addext subjectAltName=DNS:ingress.test -days 3650 -keyout key.pem -out cert.pem`)
and run the server tests. The repository's release checks accept exactly one private key in the tree, this one, and only while this file sits
beside it.
