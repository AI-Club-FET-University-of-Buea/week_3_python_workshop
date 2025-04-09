def calculate_mse(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()

def calculate_rmse(y_true, y_pred):
    return calculate_mse(y_true, y_pred) ** 0.5

def calculate_r2_score(y_true, y_pred):
    ss_total = ((y_true - y_true.mean()) ** 2).sum()
    ss_residual = ((y_true - y_pred) ** 2).sum()
    return 1 - (ss_residual / ss_total)

def print_evaluation_metrics(y_true, y_pred):
    mse = calculate_mse(y_true, y_pred)
    rmse = calculate_rmse(y_true, y_pred)
    r2 = calculate_r2_score(y_true, y_pred)
    
    print(f'Mean Squared Error: {mse:.4f}')
    print(f'Root Mean Squared Error: {rmse:.4f}')
    print(f'R^2 Score: {r2:.4f}')