import logging

def setup_logger(log_level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(message)s",
    )
