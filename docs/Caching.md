# MAVIS Caching Strategies
## Objectives
- Cache reusable DAG pipelines
- Cache results of previous executions
## Basic Structure
- Caching will be handled by **ChromaDB**, which natively supports vector storage and similarity search.
- Queries will be embedded and stored in ChromaDB collections, with metadata containing the associated pipeline ID and execution results.

## Pipeline Caching Strategy
- Let the interpreter output if a pipeline is cacheable by asking it whether it can be reused in the following manner: generalized or specialized.
- On checking the cache, generalized pipelines will be preferred over specialized ones for reusability. 
- **Parameter Extraction**: When a generalized pipeline is retrieved from the cache, a lightweight extraction step will identify and extract the necessary parameters from the new query to feed into the pipeline.

## Pipeline Output Caching Strategy
- Let the interpreter output if the pipeline output can be reused for very similar queries and for how long(TTL) in days, minutes, hours, etc. This will especially be helpful for separating out queries whose results can change frequently (such as current time, weather, etc.) and whose results remain constant (such as concepts, definitions, etc.) for the foreseeable future, such that it makes sense to cache the results and not run the pipeline again.
- **Cache Eviction**: A background job will handle cache eviction, enforcing the TTL. The system will use an **LRU (Least Recently Used)** strategy to manage the overall cache size when storage limits are reached.

## Caching Flow
1. When a query comes, it is embedded and checked against the ChromaDB collection using vector similarity search to find the top-k matches.
2. A tiered threshold approach is used for the similarity score:
   - **Similarity > 0.95**: Automatic cache hit. The cached result is served immediately only if its TTL is still valid. If the TTL has expired, the cached pipeline is executed.
   - **Similarity between 0.85 and 0.95**: The top-k queries (along with pipeline generalizability and result TTL) are passed to an LLM-driven module to verify the cache hit. If verified, the cache is reused; if rejected, execution hands off to the interpreter.
   - **Similarity < 0.85**: Automatic cache miss. Execution hands off to the interpreter to generate a pipeline as normal.