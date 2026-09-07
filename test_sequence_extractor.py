import sys
import os

# Add project root to sys.path if not there
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph.sequence_extractor import extract_workflow_sequence

text = """
1. Users must be able to register a new account.
2. Users must be able to log in to an existing account.
3. Users must be able to browse available restaurants.
4. Users must be able to search for specific food items.
5. Users must be able to add items to the shopping cart.
"""

nodes = [
  {"id": "user", "type": "Actor", "name": "User"},
  {"id": "register_account", "type": "Feature", "name": "Register Account"},
  {"id": "login_account", "type": "Feature", "name": "Login Account"},
  {"id": "browse_restaurants", "type": "Feature", "name": "Browse Restaurants"},
  {"id": "search_food_items", "type": "Feature", "name": "Search Food Items"},
  {"id": "add_to_cart", "type": "Feature", "name": "Add to Cart"}
]

print("Testing extract_workflow_sequence...")
result = extract_workflow_sequence(text, nodes)
print("Result:")
print(result)
