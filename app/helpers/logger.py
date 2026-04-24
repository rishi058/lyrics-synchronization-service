import os
import time

class CustomLogger:
    log_file = "debug.log"
    log_file_path = os.path.join(os.getcwd(), log_file)

    @staticmethod
    def log(message: str):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] \n {message}\n"
        with open(CustomLogger.log_file_path, "a", encoding="utf-8") as f:
            f.write(log_entry)