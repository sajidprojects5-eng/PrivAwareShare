
from typing import Tuple

def extract_location_demo() -> Tuple[float, float]:
    """Return dummy coordinates for demo."""
    return 17.3850, 78.4867  # e.g., Hyderabad

def map_location_to_context(lat: float, lon: float) -> str:
    if lat == 0.0 and lon == 0.0:
        return "unknown"
    return "home"
