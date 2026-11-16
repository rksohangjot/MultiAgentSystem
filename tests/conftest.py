"""Load .env before test modules are imported so GROQ_API_KEY skip checks see it."""
from dotenv import load_dotenv

load_dotenv()
