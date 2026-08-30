# tasks/
# Package for autonomous background task modules.
# Each task module should expose a zero-argument `run()` function
# and be registered with the TaskRunner in main.py.
#
# Example task module (tasks/my_task.py):
#
#   from helpers import log_it
#
#   def run():
#       # ... do work ...
#       log_it("my_task ran successfully.", "my_task")
#
# Example registration in main.py:
#
#   from tasks.my_task import run as my_task_run
#   runner.register(my_task_run, interval_minutes=30, task_name="my_task")
