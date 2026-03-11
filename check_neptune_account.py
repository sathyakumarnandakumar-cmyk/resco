import neptune

try:
    # Initialize Neptune to get account information
    project = neptune.init_project(
        project="tensorcell-sathya/TensorCell", 
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxNjU2Y2Q4OC04YWVkLTQ3ZmYtYTVlNC1mZDRlYjk4OGQ5MWIifQ==",
        mode="read-only"
    )
    
    # Get project information
    project_info = project.fetch()
    print("=== Neptune Account Information ===")
    print(f"Project: {project_info.get('sys/name', 'N/A')}")
    print(f"Workspace: {project_info.get('sys/workspace', 'N/A')}")
    print(f"Owner: {project_info.get('sys/owner', 'N/A')}")
    print(f"Created: {project_info.get('sys/creation_time', 'N/A')}")
    print(f"Project ID: {project_info.get('sys/id', 'N/A')}")
    
    project.stop()
    
except Exception as e:
    print(f"Error getting account information: {e}")
    
    # Alternative: Try to get user info from API token directly
    try:
        import base64
        import json
        
        # Decode the JWT token to get user information
        token = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxNjU2Y2Q4OC04YWVkLTQ3ZmYtYTVlNC1mZDRlYjk4OGQ5MWIifQ=="
        decoded = base64.b64decode(token + '==')  # Add padding if needed
        token_info = json.loads(decoded.decode('utf-8'))
        
        print("=== Token Information ===")
        print(f"API Address: {token_info.get('api_address', 'N/A')}")
        print(f"API URL: {token_info.get('api_url', 'N/A')}")
        print(f"API Key: {token_info.get('api_key', 'N/A')}")
        
    except Exception as token_error:
        print(f"Could not decode token: {token_error}")