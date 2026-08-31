# Bugs and Issues
## Ctrl+C and exit command does not work at all
- ~~Pressing Ctrl+C and exit doesn't properly close MAVIS and asks again what can I do for you?~~ **Fixed** (Updated interactive prompt loop to break on `exit`, `quit`, and `KeyboardInterrupt`/`EOFError`. Hardened `_stop_workers()` and shutdown handlers against `BaseException` to terminate workers cleanly within 2s without tracebacks).

## Tool addition restarts the entire MAVIS subprocess
- ~~After adding a new tool, MAVIS restarts completely so that the tool can be used. This results in 2 issues: the user has to give the command once again and the memory workers and task schedulers are also restarted.~~ **Fixed** (Removed unnecessary restart prompt after tool synthesis. Tools are executed in isolated child processes via `core/run_tool.py` which dynamically import newly created tools from disk, allowing execution to proceed immediately without restarting MAVIS or background schedulers).


## Memory is not specialized
- ~~ToolBuilder is not concerned with memory related to interpreter and vice-versa. There should be a way to demarcate different topics in memory, such as tools, tasks, user-profile, etc. Possibly, a publisher-subscriber model will be best with each component being a subscriber to relevant memory topics.~~ **Fixed** (Implemented topic namespaces in `MemoryStore` for `interpreter`, `toolbuilder`, `debugger`, and `tasks`. Added multi-collection ChromaDB isolation, cross-namespace peer retrieval via `extra_namespaces`, and specialized pattern/fix writes for ToolBuilder and Debugger).

## High Dependence on Gemini
- ~~MAVIS only uses Gemini as a backbone and should be able to support using any LLM as a backbone. To achieve this, LLM calls must be abstracted to a LLM class (something like Langchain client) for this purpose.~~ **Fixed** (Introduced `core/llm` unified provider layer with `BaseLLMClient`, `GeminiClient`, `OpenAICompatClient`, and `OllamaClient`. MAVIS now loads provider settings from `mavis_config.json` and supports local LLMs without code changes).

## Scheduler and program self-exited after single command
- ~~After executing any command, the scheduler stops and MAVIS self-exits.~~ **Fixed** (Updated `interpret_command()` in `main.py` to explicitly return `True` on all continuation branches and `False` only on exit/quit, preventing the `if not interpret_command(command): break` loop condition from prematurely terminating on falsy `None`).

## Sandboxed tools time out on unlisted or greylisted commands
- ~~Executing shell commands or tools requiring user approval inside `core/run_tool.py` times out after 30s.~~ **Fixed** (Implemented bidirectional IPC between `core/run_tool.py` and `main.py` using `sys.__stdout__`/`sys.__stdin__` and `selectors`. Child processes now delegate approval prompts to MAVIS's main terminal UI. Also enhanced `preflight_scan` to perform deep parameter scanning on shell commands and forward session allowances to tool subprocesses).

## Cognitive/semantic tasks fail in ToolBuilder retry loop
- ~~Queries requiring cognitive processing (summaries, transformations, factual lookups) caused ToolBuilder to synthesize Python files calling raw HTTP endpoints, which failed pytest assertions and exhausted 3 retries.~~ **Fixed** (Implemented heterogeneous DAG with `subagent` nodes, `AgentBuilder`, `AgentTester` with LLM-as-a-Judge binary `passed`/`failed` verdicts, `AgentDebugger` with `memories/agent_debugger/`, built-in `semantic_transform` primitive, and terminal `Answerer` module).

## Tool debugging retries failed due to sys.modules caching and return contract mismatch
- ~~Automated tool debugging repeated the initial failure across all 3 retries and failed pytest assertions.~~ **Fixed** (Evicted `tools.{func_name}` from `sys.modules` in `ToolTester` before running tests so debugged files are reloaded fresh from disk. Established that `(status: int, output: Any)` is an internal execution contract: `core/run_tool.py` and `execute_pipeline` strictly enforce status checking and error handling, while `ToolTester` verifies the 2-tuple return structure without leaking execution plumbing into the Interpreter's high-level planning).
- ~~ChromaDB `$gte` query error on string dates and unhandled retrieval exceptions caused silent interactive loop exit in `main.py`.~~ **Fixed** (In `memory_store.py`, updated `_query_chroma` to filter TTL date cutoffs in Python over metadata dates instead of passing string `$gte` to ChromaDB. Protected `sync_tools` in `tool_retriever.py` and wrapped `interpret_command` in `main.py` with exception handling and logging so unexpected errors never silently terminate the session).
- ~~Multi-file summarization pipeline mismatch in `read_and_concatenate_files`.~~ **Fixed** (Hardened `read_and_concatenate_files` to cleanly accept raw string/markdown JSON/newline inputs and parse them into a file list. Updated Interpreter system prompt to ensure multi-file content gathered by file reading/concatenation flows downstream into cognitive agents like `semantic_transform` for semantic summarization).