import matplotlib.pyplot as plt
import pandas as pd
import config

def plot_training_stats(train_losses, val_losses, train_accuracies, val_accuracies):
    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label="Train Accuracy")
    plt.plot(val_accuracies, label="Validation Accuracy")
    plt.title("Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_test_prediction(point_cloud, true_labels, pred_labels):
    df_true = pd.DataFrame({
        'x': point_cloud[:, 0], 
        'y': point_cloud[:, 1], 
        'z': point_cloud[:, 2], 
        'label': true_labels
    })
    
    df_pred = pd.DataFrame({
        'x': point_cloud[:, 0], 
        'y': point_cloud[:, 1], 
        'z': point_cloud[:, 2], 
        'label': pred_labels
    })
    
    fig = plt.figure(figsize=(15, 6))
    
    # True
    ax1 = fig.add_subplot(121, projection='3d')
    for idx, _class in enumerate(config.CLASSES):
        c_df = df_true[df_true['label'] == idx]
        if len(c_df) > 0:
            ax1.scatter(c_df['x'], c_df['y'], c_df['z'], c=[_class['color']] * len(c_df), label=_class['label'], alpha=0.5)
    ax1.set_title('True')
    ax1.legend()
    
    # Predicted
    ax2 = fig.add_subplot(122, projection='3d')
    for idx, _class in enumerate(config.CLASSES):
        c_df = df_pred[df_pred['label'] == idx]
        if len(c_df) > 0:
            ax2.scatter(c_df['x'], c_df['y'], c_df['z'], c=[_class['color']] * len(c_df), label=_class['label'], alpha=0.5)
    ax2.set_title('Predicted')
    ax2.legend()
    
    plt.show()