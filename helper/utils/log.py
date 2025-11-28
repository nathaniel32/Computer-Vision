import os

class Logging:
    def __init__(self, log_dir, file_name="log.txt"):
        self.file_path = os.path.join(log_dir, file_name)
        os.makedirs(log_dir, exist_ok=True)

    def _write_to_file(self, message: str) -> None:
        with open(self.file_path, "a", encoding="utf-8") as f:
            f.write(message + "\n")
    
    def clear(self) -> None:
        with open(self.file_path, "w") as f:
            f.write("")

    def print(self, *args: object) -> None:
        args_str = " ".join(str(arg) for arg in args) if args else ""
        if args_str:
            print(args_str)

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