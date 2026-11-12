## Bug fixes

- Production verification now accepts a carried qualification gate whose source is a complete, all-pass bundle of an earlier engine build that was never sealed, when the evidence also holds a bit-identity record showing the new build answers that bundle binary's requests with identical outputs, a full-window request included. Before, a carried gate needed a registry-pinned sealed approval of the earlier build. Exactness and prefix-cache checks still always re-run. The release tools also gain a one-time pin tool for registry entries sealed before carry-forward existed. PR #116.
