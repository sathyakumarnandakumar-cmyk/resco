import neptune
import requests
import base64
import json

# First, let's decode the API token to see what info we can extract
try:
    token = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxNjU2Y2Q4OC04YWVkLTQ3ZmYtYTVlNC1mZDRlYjk4OGQ5MWIifQ=="
    decoded = base64.b64decode(token + '==')
    token_info = json.loads(decoded.decode('utf-8'))
    
    print("=== API Token Information ===")
    print(f"API Address: {token_info.get('api_address')}")
    print(f"API URL: {token_info.get('api_url')}")
    print(f"API Key: {token_info.get('api_key')}")
    
    # Try to get user info from Neptune API directly
    api_key = token_info.get('api_key')
    api_url = token_info.get('api_url')
    
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    
    # Try to get current user info
    try:
        response = requests.get(f'{api_url}/api/backend/v1/user/current', headers=headers)
        if response.status_code == 200:
            user_info = response.json()
            print("\n=== User Account Information ===")
            print(f"Email: {user_info.get('email', 'N/A')}")
            print(f"Username: {user_info.get('username', 'N/A')}")
            print(f"First Name: {user_info.get('firstName', 'N/A')}")
            print(f"Last Name: {user_info.get('lastName', 'N/A')}")
            print(f"Organization: {user_info.get('organization', {}).get('name', 'N/A')}")
        else:
            print(f"\nFailed to get user info. Status: {response.status_code}")
            print(f"Response: {response.text}")
    except Exception as api_error:
        print(f"Error calling Neptune API: {api_error}")
    
    # From the project URL, we can see the workspace is "tensorcell-sathya"
    print("\n=== Project Information (from URL) ===")
    print("Workspace: tensorcell-sathya")
    print("Project: TensorCell")
    print("Full URL: https://app.neptune.ai/tensorcell-sathya/TensorCell/")
    
except Exception as e:
    print(f"Error processing token: {e}")