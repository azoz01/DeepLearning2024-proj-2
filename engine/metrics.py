from sklearn.metrics import balanced_accuracy_score

def balanced_accuracy(y_true, y_pred):
    return balanced_accuracy_score(y_true.cpu().numpy(), y_pred.cpu().numpy())

def accuracy(y_true, y_pred):
    return (y_true == y_pred).float().mean().item()
