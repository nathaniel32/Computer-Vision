import os
import config

class Logging:
    def __init__(self, file_name="log.txt"):
        self.file_path = os.path.join(config.LOG_DIR, file_name)

    def clear(self) -> None:
        with open(self.file_path, "w") as f:
            f.write("")

    def _write_to_file(self, message: str) -> None:
        with open(self.file_path, "a") as f:
            f.write(message + "\n")

    def info(self, *args: object) -> None:
        args_str = " ".join(str(arg) for arg in args) if args else ""
        if args_str:
            print(args_str)
            self._write_to_file(args_str)

    def warning(self, *args: object) -> None:
        args_str = " ".join(str(arg) for arg in args) if args else ""
        if args_str:
            msg = f"Warning: {args_str}"
            print(msg)
            self._write_to_file(msg)

    def error(self, *args: object) -> None:
        args_str = " ".join(str(arg) for arg in args) if args else ""
        if args_str:
            msg = f"Error: {args_str}"
            print(msg)
            self._write_to_file(msg)

logger = Logging()