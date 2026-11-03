## Bug fixes

- The release ceremony (`tools/cut_release.py --publish`) now verifies an approval whose bundle leaves
  the server and cluster executables to the dev engine release. It downloads the executables the approval
  names from that engine release, which must be the one the release manifest names, and checks their
  exact name, size, digest and bytes, and that the engine tag names the approved commit. Before this fix,
  every such approval, which includes every v0.1.0 model, was refused as "engine binary files are
  required". Each model's dev release is now also staged in its own directory, so a release with several
  models no longer stops at the second model on a "destination conflict" between the models' receipts.
  (PR #58)
