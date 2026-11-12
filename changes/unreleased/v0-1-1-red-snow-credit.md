## New features and improvements

- The RED-SNOW-5.3-Flash two-node profile (#127) and the GLM engine pieces it uses (P2 prefill for every expert format of that pack, the pool-page prompt cache, the modelled activation reserve, the embedding table read in place) were built as our own implementation with @justincasey's patch and measurements in #95 as the reference.
