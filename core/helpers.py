# Helper functions for logging and monitoring
import os

_MAV_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def log_it(message, entity_name):
    # Log the message to entity log file and main log file
    log_file = os.path.join(_MAV_ROOT, "logs", f"{entity_name}.log")
    main_log = os.path.join(_MAV_ROOT, "logs", "main.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, 'a') as file:
        file.write(f"{message}\n")
    with open(main_log, 'a') as file:
        file.write(f"{entity_name}: {message}\n")
