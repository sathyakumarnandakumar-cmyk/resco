import neptune
import requests

try:
    # Initialize Neptune management client
    project = neptune.init_project(
        project="tensorcell-sathya/TensorCell", 
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxNjU2Y2Q4OC04YWVkLTQ3ZmYtYTVlNC1mZDRlYjk4OGQ5MWIifQ==",
        mode="read-only"
    )
    
    print("=== Neptune Account Information ===")
    print("✅ Successfully connected to Neptune")
    print(f"Project URL: https://app.neptune.ai/tensorcell-sathya/TensorCell/")
    print(f"Workspace: tensorcell-sathya")
    print(f"Project Name: TensorCell")
    
    # The workspace name "tensorcell-sathya" suggests this might be associated with 
    # an account named "sathya" under the "tensorcell" organization
    
    project.stop()
    
    print("\n=== Account Details (Inferred) ===")
    print("• Workspace indicates this is likely associated with user 'sathya'")
    print("• Organization/Team: 'tensorcell'") 
    print("• To get the exact email, you would need to:")
    print("  - Log into Neptune UI at https://app.neptune.ai")
    print("  - Go to Account Settings")
    print("  - Check Profile section")
    
    print("\n=== API Token Info ===")
    print("• Token is valid and working")
    print("• Has read access to tensorcell-sathya/TensorCell project")
    print("• API Key: 1656cd88-8aed-47ff-a5e4-fd4eb988d91b")
    
except Exception as e:
    print(f"Error: {e}")