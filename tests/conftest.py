import pytest
from dotenv import load_dotenv

# Load environment variables BEFORE pytest starts collecting tests
load_dotenv()


@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"
