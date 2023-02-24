import logging
import inspect
from pathlib import Path

class SCFormatter(logging.Formatter):
    """Log Formatter. Passed during creation of logger"""
    def __init__(self):
        self.header_formatter = logging.Formatter("\u001b[32m%(asctime)s\u001b[0m - \u001b[34m%(name_last)15s\u001b[0m -", datefmt='%Y-%m-%d,%H:%M:%S')
        self.colors = {
            'DEBUG': '\u001b[38;5;245m',
            'INFO': '\u001b[37m',
            'WARNING': '\u001b[33m',
            'ERROR': '\u001b[31m',
            'CRITICAL': '\u001b[31;1m'
        }
        self.msg_formatter    = logging.Formatter("%(message)s")

    def format(self, record):
        color = self.colors[record.levelname]
        level = logging.Formatter(color + "%(levelname)8s").format(record)
        record.name_last = record.name.rsplit('.', 1)[-1]
        stack_depth = len(inspect.stack(0))-10
        header = self.header_formatter.format(record)
        msg = '[' + self.msg_formatter.format(record) + ']'
        output = "{header} {level} {stack_indentation} {msg}{escape_color}".format(
            header=header, 
            level=level,
            stack_indentation='.'*(stack_depth*2),
            msg=msg,
            escape_color='\u001b[0m')
        return output

def get_logger(logname='mlopsaws', logfile=None, log_level=logging.INFO):
    formatter = SCFormatter()
    if logfile is None:
        logger = logging.getLogger(logname)
        logger.setLevel(log_level)
    else:
        logger = logging.getLogger('mlopsaws' + f".{Path(logfile).stem}")
        filehandler = logging.FileHandler(logfile, 'w')
        assert Path(logfile).parent.exists(), f"{logfile} is not a valid log file directory"
        filehandler.setFormatter(formatter)
        filehandler.setLevel(log_level)
        logger.addHandler(filehandler)

    # Prevent logging from propagating to the root logger
    logger.propagate = False
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    return logger