import optuna

# 1. Define the objective function you want to minimize (or maximize)
def objective(trial):
    # Suggest a value for 'x' between -10 and 10
    x = trial.suggest_float("x", -10, 10)
    
    # Return the result of the function
    return (x - 2) ** 2

if __name__ == "__main__":
    # 2. Create a study object
    study = optuna.create_study(direction="minimize")
    
    # 3. Optimize the study, running 100 trials
    study.optimize(objective, n_trials=100)

    # 4. Print the results
    print(f"Best value: {study.best_value}")
    print(f"Best parameters: {study.best_params}")