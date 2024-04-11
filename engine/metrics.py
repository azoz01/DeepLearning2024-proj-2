def accuracy(y_true, y_pred):
    return (y_true == y_pred).float().mean().item()
