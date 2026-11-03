## Known issues

- GLM-5.3-Flash on one node (`glm53-flash-single`) keeps 2 prefix-cache entries at its 262,144-token window, so the cache holds one long conversation at a time. A second long conversation evicts the first, and the next request of the first one recomputes its prompt. Replies are unchanged.
- GLM-5.3-Flash's 1,048,576-token window needs the two-node profile; one GB10 holds 262,144 tokens with the required features on.
- GLM-5.3-Flash's two-node profile listens on the leader node's network address with no API key and no request log. Run it only on a private, trusted network, or behind a firewall or an SSH tunnel.
- GLM-5.3-Flash's thinking has no off switch; scale it with `reasoning_effort`.
