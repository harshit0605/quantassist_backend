from dotenv import load_dotenv
import os

def get_api_key(api_key_name):
    """Load API key from .env file."""
    load_dotenv()  # Load environment variables from .env file
    return os.getenv(api_key_name)

# Example usage
if __name__ == "__main__":
    API_KEY = get_api_key()  # Retrieve API key from .env file
