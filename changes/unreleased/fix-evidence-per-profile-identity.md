## Bug fixes

- Production verification now accepts a recipe whose profiles each pin their own pack and runtime id,
  the layout dev uses for a model with a single-node and a two-node profile. Every receipt identity
  now names its `pack` role. A profile may name a lock role other than `draft`, and the model lock
  must carry a `target` pack and every role a profile names. Before, `verify` and the promotion tools
  derived every profile's identity from the approval's top-level runtime id and refused such a recipe
  as "approved profile identity drift". Approvals sealed before dev added the `pack` identity key no
  longer verify; none is registered in production. (PR #52)
