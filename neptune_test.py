import neptune

try:
    # This tries to initialize a connection in 'read-only' mode 
    # so you don't accidentally create a fake run in your team's project.
    project = neptune.init_project(
        project="tensorcell-sathya/TensorCell", 
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxNjU2Y2Q4OC04YWVkLTQ3ZmYtYTVlNC1mZDRlYjk4OGQ5MWIifQ==",
        mode="read-only"
    )
    print("✅ Connection successful! Your key is working.")
    project.stop()
except Exception as e:
    print(f"❌ Connection failed. Error: {e}")



    #project="tensorcell-sathya/TensorCell",
    #api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxNjU2Y2Q4OC04YWVkLTQ3ZmYtYTVlNC1mZDRlYjk4OGQ5MWIifQ==",