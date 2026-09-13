## Future Work
- **Context-Aware Semantic Caching**:
  - Resolve anaphora and deictic references (e.g. queries like *"Repeat this again"*, *"do that with python"*, *"retry"*).
  - Raw isolated queries containing deictic terms must either bypass semantic caching or be expanded against prior conversation context (e.g. incorporating previous turn's tool execution state/signature into the composite cache key) to prevent catastrophic false-positive cache hits across unrelated conversational sessions.