import os
import pc_color_seg.configs as configs

class Logging:
    def __init__(self, config:configs.BaseConfig):
        self.config = config

    def _write_to_file(self, message: str) -> None:
        os.makedirs(os.path.dirname(self.config.save_log_path), exist_ok=True)
        with open(self.config.save_log_path, "a", encoding="utf-8") as f:
            f.write(message + "\n")
    
    def clear(self) -> None:
        os.makedirs(os.path.dirname(self.config.save_log_path), exist_ok=True)
        with open(self.config.save_log_path, "w") as f:
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