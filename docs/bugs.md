# Bugs and Issues
## Ctrl+C and exit command does not work at all
- Pressing Ctrl+C and exit doesn't properly close MAVIS and asks again what can I do for you?

## Tool addition restarts the entire MAVIS subprocess
- After adding a new tool, MAVIS restarts completely so that the tool can be used. This results in 2 issues: the user has to give the command once again and the memory workers and task schedulers are also restarted.


## Memory is not specialized
- ToolBuilder is not concerned with memory related to interpreter and vice-versa. There should be a way to demarcate different topics in memory, such as tools, tasks, user-profile, etc. Possibly, a publisher-subscriber model will be best with each component being a
subscriber to relevant memory topics.

