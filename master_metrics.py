import numpy as np

def calculate_metrics(actual, predicted, mape_threshold=5.0):
    """
    Calculate master evaluation metrics for GHI predictions.
    
    Parameters:
    - actual: array-like of measured ground truth values
    - predicted: array-like of predicted values
    - mape_threshold: minimum actual value to consider for MAPE calculation (to avoid division by zero)
    
    Returns:
    - dictionary containing RMSE, nRMSE, MAE, and MAPE
    """
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    # Ensure same length
    if len(actual) != len(predicted):
        raise ValueError(f"Actual (len {len(actual)}) and predicted (len {len(predicted)}) arrays must have the same length.")
        
    rmse = np.sqrt(np.mean((predicted - actual) ** 2))
    mae = np.mean(np.abs(predicted - actual))
    
    mean_actual = np.mean(actual)
    nrmse = (rmse / mean_actual) * 100 if mean_actual != 0 else 0
    
    mape_mask = actual >= mape_threshold
    if np.sum(mape_mask) > 0:
        mape = np.mean(np.abs((predicted[mape_mask] - actual[mape_mask]) / actual[mape_mask])) * 100
    else:
        mape = 0
        
    return {
        'RMSE': rmse,
        'nRMSE': nrmse,
        'MAE': mae,
        'MAPE': mape,
        'N': int(len(actual)),
    }

def print_metrics(metrics_dict, title="EVALUATION METRICS", unit="W/m²"):
    """
    Utility to beautifully print the calculated metrics dictionary.
    """
    nrmse = metrics_dict.get("nRMSE", metrics_dict.get("nRMSE_pct", 0))
    mae = metrics_dict.get("MAE", 0)
    mape = metrics_dict.get("MAPE", metrics_dict.get("MAPE_pct", 0))
    print(f"\n--- {title} ---")
    print(f"RMSE  : {metrics_dict['RMSE']:.2f} {unit}")
    print(f"nRMSE : {nrmse:.2f} %")
    print(f"MAE   : {mae:.2f} {unit}")
    print(f"MAPE  : {mape:.2f} %")
    if "N" in metrics_dict:
        print(f"N     : {metrics_dict['N']}")

# Simple test block (only runs if executed directly)
if __name__ == "__main__":
    # Dummy data
    test_actual = [100, 200, 300, 400, 500]
    test_pred = [110, 190, 315, 380, 525]
    res = calculate_metrics(test_actual, test_pred)
    print_metrics(res, title="TEST METRICS OUTPUT")
