## Future Work
- ~~Optimize context and token usage by improving prompt construction and memory retrieval.~~ **Completed** (Implemented stable prefix ordering via `system_instruction`, working memory active-turns capping at 8 turns, entry truncation, and proactive 1500-token compaction threshold).
- ~~Improve tool categorization and retrieval, so only relevant tools are given to LLM for a query while also ensuring that generalized tools are properly utilized.~~ **Completed** (Implemented `ToolRetriever` in `core/tool_retriever.py` with ChromaDB semantic search, dynamic thresholding at 8 tools, and core utility guarantees).
- Improve latency by caching tool pipelines and execution results for similar queries.
