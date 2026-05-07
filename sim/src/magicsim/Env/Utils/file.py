import sys


class Logger:
    def __init__(self, name: str, logger, log_file: str = None):
        self.name = name
        self.logger = logger
        self.logger.add(
            sys.stdout,
            colorize=True,
            format="<green>{time}</green> <level>{message}</level>",
        )
        if log_file is not None:
            self.logger.add(log_file)

    def info(self, *message):
        message = " ".join([str(m) for m in message])
        self.logger.info(f"{self.name}: {message}")

    def debug(self, *message):
        message = " ".join([str(m) for m in message])
        self.logger.debug(f"{self.name}: {message}")

    def warning(self, *message):
        message = " ".join([str(m) for m in message])
        self.logger.warning(f"{self.name}: {message}")

    def error(self, *message):
        message = " ".join([str(m) for m in message])
        self.logger.error(f"{self.name}: {message}")
