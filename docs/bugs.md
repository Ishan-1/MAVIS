# Bugs and Issues
## Ctrl+C and exit command does not work at all
- ~~Pressing Ctrl+C and exit doesn't properly close MAVIS and asks again what can I do for you?~~ **Fixed** (Updated interactive prompt loop to break on `exit`, `quit`, and `KeyboardInterrupt`/`EOFError`. Hardened `_stop_workers()` and shutdown handlers against `BaseException` to terminate workers cleanly within 2s without tracebacks).

## Tool addition restarts the entire MAVIS subprocess
- ~~After adding a new tool, MAVIS restarts completely so that the tool can be used. This results in 2 issues: the user has to give the command once again and the memory workers and task schedulers are also restarted.~~ **Fixed** (Removed unnecessary restart prompt after tool synthesis. Tools are executed in isolated child processes via `core/run_tool.py` which dynamically import newly created tools from disk, allowing execution to proceed immediately without restarting MAVIS or background schedulers).


## Memory is not specialized
- ~~ToolBuilder is not concerned with memory related to interpreter and vice-versa. There should be a way to demarcate different topics in memory, such as tools, tasks, user-profile, etc. Possibly, a publisher-subscriber model will be best with each component being a subscriber to relevant memory topics.~~ **Fixed** (Implemented topic namespaces in `MemoryStore` for `interpreter`, `toolbuilder`, `debugger`, and `tasks`. Added multi-collection ChromaDB isolation, cross-namespace peer retrieval via `extra_namespaces`, and specialized pattern/fix writes for ToolBuilder and Debugger).

## High Dependence on Gemini
- ~~MAVIS only uses Gemini as a backbone and should be able to support using any LLM as a backbone. To achieve this, LLM calls must be abstracted to a LLM class (something like Langchain client) for this purpose.~~ **Fixed** (Introduced `core/llm` unified provider layer with `BaseLLMClient`, `GeminiClient`, `OpenAICompatClient`, and `OllamaClient`. MAVIS now loads provider settings from `mavis_config.json` and supports local LLMs without code changes).