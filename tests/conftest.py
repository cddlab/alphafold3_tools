import pytest
from loguru import logger


@pytest.fixture
def warnings_log():
    """Collects the messages logged at WARNING level or above."""
    messages: list[str] = []
    handler_id = logger.add(lambda message: messages.append(message.record["message"]), level="WARNING")
    yield messages
    logger.remove(handler_id)
