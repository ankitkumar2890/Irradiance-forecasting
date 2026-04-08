import os
import matplotlib.pyplot as plt
import numpy as np

def set_style():
    """Set the master plot style."""
    plt.style.use('seaborn-v0_8-darkgrid')

def plot_time_series(time_array, actual, predicted, title="Actual vs Predicted GHI", save_path=None):
    """
    Creates a standard line graph over time comparing actual vs predicted.
    
    Parameters:
    - time_array: datetime array or generic sequence for the X-axis
    - actual: array-like of ground truth values
    - predicted: array-like of predicted values
    - title: String title for the plot
    - save_path: Absolute path to save the generated image
    """
    set_style()
    plt.figure(figsize=(15, 6))
    
    plt.plot(time_array, actual, label='Measured (Ground Truth)', color='dodgerblue', linewidth=2, alpha=0.8)
    plt.plot(time_array, predicted, label='Predicted GHI', color='coral', linewidth=2, linestyle='dashed')
    
    plt.title(title, fontsize=16, weight='bold')
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Irradiance (W/m²)', fontsize=12)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Time series plot saved to: {save_path}")
    else:
        plt.show()
    plt.close()

def plot_4panel_evaluation(actual, predicted, hour_array=None, title="Model Evaluation Analysis", save_path=None):
    """
    Generates the comprehensive 4-panel analysis grid used for clear sky verification.
    Includes: Scatter, Error Histogram, Error vs Measured, Error vs Time of Day.
    
    Parameters:
    - actual: array of true values
    - predicted: array of predicted values
    - hour_array: optional array of hour of day (float 0-24) mapped to the data points.
    - title: Main figure title
    - save_path: Absolute path to save the generated image
    """
    actual = np.array(actual)
    predicted = np.array(predicted)
    errors = predicted - actual
    
    set_style()
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(title, fontsize=20, weight='bold', y=0.98)
    
    # 1. Scatter Plot: Predicted vs Actual
    ax1 = plt.subplot(2, 2, 1)
    ax1.scatter(actual, predicted, alpha=0.6, color='dodgerblue', edgecolor='k', s=60)
    
    max_val = max(np.max(actual) if len(actual) else 0, np.max(predicted) if len(predicted) else 0)
    min_val = min(np.min(actual) if len(actual) else 0, np.min(predicted) if len(predicted) else 0)
    
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction (y=x)')
    ax1.set_title('Predicted vs Measured Ground Truth', fontsize=14, weight='bold')
    ax1.set_xlabel('Measured GHI (W/m²)', fontsize=12)
    ax1.set_ylabel('Predicted GHI (W/m²)', fontsize=12)
    ax1.legend()

    # 2. Histogram of Errors
    ax2 = plt.subplot(2, 2, 2)
    ax2.hist(errors, bins=30, color='coral', edgecolor='black', alpha=0.8)
    ax2.axvline(0, color='r', linestyle='--', linewidth=2, label='Zero Error')
    if len(errors) > 0:
        mean_err = np.mean(errors)
        ax2.axvline(mean_err, color='k', linestyle='-', linewidth=2, label=f"Mean Error ({mean_err:.1f} W/m²)")
    ax2.set_title('Error Distribution (Predicted - Measured)', fontsize=14, weight='bold')
    ax2.set_xlabel('Error (W/m²)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.legend()

    # 3. Error vs Initial Measured Value
    ax3 = plt.subplot(2, 2, 3)
    ax3.scatter(actual, errors, alpha=0.6, color='purple', edgecolor='w', s=60)
    ax3.axhline(0, color='r', linestyle='dotted', linewidth=2)
    ax3.set_title('Error Magnitude vs Incoming Radiation', fontsize=14, weight='bold')
    ax3.set_xlabel('Measured Ground Truth GHI (W/m²)', fontsize=12)
    ax3.set_ylabel('Error (W/m²)', fontsize=12)

    # 4. Error vs Time of Day
    ax4 = plt.subplot(2, 2, 4)
    if hour_array is not None and len(hour_array) == len(actual):
        scatter = ax4.scatter(hour_array, errors, c=actual, cmap='viridis', alpha=0.8, s=60, edgecolor='w')
        ax4.axhline(0, color='r', linestyle='dotted', linewidth=2)
        ax4.set_title('Prediction Bias by Time of Day', fontsize=14, weight='bold')
        ax4.set_xlabel('Hour of Day', fontsize=12)
        ax4.set_ylabel('Error (W/m²)', fontsize=12)
        cbar = plt.colorbar(scatter, ax=ax4)
        cbar.set_label('Measured GHI (W/m²)')
    else:
        ax4.text(0.5, 0.5, 'Time of Day Mapping\nNot Provided or Mismatched', horizontalalignment='center', verticalalignment='center', fontsize=14, color='grey')
        ax4.set_axis_off()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"4-Panel Evaluation Plot saved to: {save_path}")
    else:
        plt.show()
    plt.close()
