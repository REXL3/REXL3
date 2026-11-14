## Bug fixes

- Corrected the multi-model promotion command reference to record the protected main base once and reuse it for every promotion onto one release branch. The former second-promotion command used the first promotion's `HEAD`, which the CLI correctly refuses because it is not on protected main; a regression test now covers two successful promotions with the original base and the old command's refusal.
