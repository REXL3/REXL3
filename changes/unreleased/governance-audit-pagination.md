## Bug fixes

- Fixed the governance audit's merged-PR scan stopping early. The pulls listing is sorted by last update, but the scan stopped on the merge date of a page's last PR, so a PR merged long ago and updated recently could end the scan before a later page that still held PRs merged in the 14-day window. It now stops on the update date. Covered by a new unit test.
