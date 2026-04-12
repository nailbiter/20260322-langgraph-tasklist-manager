# adapted from https://github.com/nailbiter/alex_leontiev_toolbox_python/blob/c5f5b4de2c18669e8e2d4394070bdf32abca0ca7/alex_leontiev_toolbox_python/utils/logging_helpers/__init__.py
import typing
import logging
import functools
import sys


@functools.singledispatch
def make_log_format(x) -> str:
    raise NotImplementedError(x)


@make_log_format.register
def _(l: list) -> str:
    return " - ".join([f"%({x})s" for x in l])


@make_log_format.register
def _(s: str) -> str:
    return s


_LOG_LEVELS_URGENCY = {
    k: i for i, k in enumerate(["DEBUG", "INFO", "WARNING", "ERROR"])
}


def get_configured_logger(
    name: str,
    level: typing.Literal["DEBUG", "INFO", "WARNING"] = "DEBUG",
    log_format=make_log_format(
        [
            "levelname",
            "name",
            "asctime",
            "message",
        ]
    ),
    is_pre_clean: bool = True,
    is_propagate: bool = False,
    log_to_file: typing.Optional[str] = None,
    file_log_level: typing.Literal["DEBUG", "INFO", "WARNING"] = "DEBUG",
    file_log_format: typing.Optional[str] = None,
    file_mode: str = "w",
) -> logging.Logger:
    app_logger = logging.getLogger(name)

    if not is_propagate:
        app_logger.propagate = False

    # --- Step 2: Set the logging level for YOUR logger ---
    # This logger will now process any message of DEBUG severity or higher.
    app_logger.setLevel(
        getattr(
            logging,
            min(
                [level] + ([] if log_to_file is None else [file_log_level]),
                key=_LOG_LEVELS_URGENCY.get,
            ),
        )
    )

    if is_pre_clean:
        # while len(app_logger.handlers) > 0:
        #     h = app_logger.handlers[0]
        #     # dbg.debug('removing handler %s'%str(h))
        #     app_logger.removeHandler(h)
        #     # dbg.debug('%d more to go'%len(testLogger.handlers))
        app_logger.handlers.clear()

    handlers = []

    # --- Step 3: Create a StreamHandler to output to stderr for YOUR logger ---
    # This handler will specifically handle messages from 'app_logger'.
    app_console_handler = logging.StreamHandler(
        sys.stderr
    )  # or just logging.StreamHandler()
    # You can also set a level on the handler if you want it to be more restrictive
    # than the logger itself, but typically you want it to respect the logger's level.
    # app_console_handler.setLevel(logging.DEBUG)
    # --- Step 4: Create a Formatter for better message layout (Optional but recommended) ---
    formatter = logging.Formatter(log_format)
    app_console_handler.setFormatter(formatter)
    app_console_handler.setLevel(getattr(logging, level))
    handlers.append(app_console_handler)

    if log_to_file is not None:
        handler = logging.FileHandler(log_to_file, mode=file_mode)
        # Create a formatter and set it for the handler
        formatter = logging.Formatter(
            log_format if file_log_format is None else file_log_format
        )
        handler.setLevel(getattr(logging, file_log_level))
        handler.setFormatter(formatter)
        handlers.append(handler)

    # --- Step 5: Add the configured handler to YOUR logger ---
    for h in handlers:
        app_logger.addHandler(h)
    return app_logger
