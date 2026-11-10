## Bug fixes

- Production verification now carries the GLM-5.3-Flash v0.1.0 qualification gates across two reviewed renames that changed no serving input: the single and two-node entries were split into their own runtime ids, and both were renamed to the `glm53-flash-sage-2.25` and `glm53-flash-tp2-sage-2.25` ids with the pack repository moved to `r0b0tlab/GLM-5.3-Flash-EXL3-SAGE-2.25bpw` (same revision, same files). Every other identity key stays strict, and the pack records must still match exactly. Replaces #113.
