import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, cohen_kappa_score, confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve, average_precision_score


def plot_probability_histograms(model_name, predicted_probabilities, axis, row, encoder):
    """
    Plots array of predicted probabilities for test/validation data on each class

    Parameters:
    -----------
    model_name : str
        Name of model, used for formatting
    predicted_probabilities : NumPy Array
        Array of predicited probabilities, each column will be plotted in histogram,
        each column represents probabiltiies for each unique class
    axis : Matplotlib axis
        Specific axis of matplotlib figure the histogram will be plotted
    row : int
        Row of defined matplotlib figure for each model
    colors: Matplotlib object
        Matplotlib object of colors made to distinguish classes
    encoder: Sci-Kit LabelEncoder
        Encodes and decodes y-value labels from a numpy array

    Returns:
    --------
    None
    """

    num_classes = predicted_probabilities.shape[1]
    colors = plt.cm.rainbow(np.linspace(0, 1, 3))

    # Iterate over each class
    for m in range(num_classes):
        class_probs = predicted_probabilities[:, m]

        # Plot histogram
        axis[row, m].hist(class_probs, bins=100, range=(0.0, 1.0), edgecolor='black', alpha=0.7, color=colors[m])
        axis[row, m].set_xlabel('Probability')
        axis[row, m].set_ylabel('Frequency')
        axis[row, m].set_title(f'Probabilities: {encoder.inverse_transform([m])[0]}s, {model_name}')


def plot_confusion_matrix(conf_mat, axis, name, class_li):
    """
    Plot sk-learn confusion matrix in matplotlib

    Takes conf_matrix, plots in specific axis depending on model,
    adds correct formatting and style

    Parameters:
    -----------
    conf_mat : Sklearn Confusion Matrix (np.array)
        Confusion matrix that will be plotted in matplotlib
    axis : Matplotlib axis
        Specific column of matplotlib figure the confusion matrix is plotted
    name : str
        Name of model, used for formatting
    class_li : list
        List of unique classes being plotted in Confusion Matrix

    Returns:
    --------
    None
    """

    # Initialize formatting
    im = axis.imshow(conf_mat, cmap=plt.cm.coolwarm)
    axis.set_title(name)
    axis.set_xlabel('Predicted Labels')
    axis.set_ylabel('True Labels')
    axis.set_xticks(np.arange(len(class_li)))
    axis.set_yticks(np.arange(len(class_li)))
    axis.set_xticklabels(class_li)
    axis.set_yticklabels(class_li)
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Display values in each cell of the confusion matrix
    for i in range(len(class_li)):
        for j in range(len(class_li)):
            axis.text(j, i, conf_mat[i, j], ha="center", va="center", color="white")


def plot_rocauc_pr_rec(model_name, ax_roc, ax_pr, y_true,y_pred_proba, class_li, encoder):
    """
    Plot ROC-AUC and Precision Recall curves for each class

    Takes true y-values, classifiers predicted probabilities for all classes,
    plots calculated ROC-AUC and Precision-Recall

    Parameters:
    -----------
    model_name : str
        Name of model, used for formatting
    ax_roc : Matplotlib subplot
        Specific subplot of matplotlib figure the ROC-AUC will be plotted
    ax_pr : Matplotlib subplot
        Specific subplot of matplotlib figure the Precision-Recall will be plotted
    y_true : NumPy Array
        Array of true y-values, used for calculating ROC-AUC and Precision-Recall
    y_pred_proba : NumPy Array
        Array of y predicited probabilities, used for calculating ROC-AUC and Precision-Recall
    class_li : list
        List of unique classes
    encoder: Sci-Kit LabelEncoder
        Encodes and decodes y-value labels from a numpy array

    Returns:
    --------
    None
    """
    
    # Plot ROC-AUC multiclass, each classifier
    for m in range(len(class_li)):
        class_name = encoder.inverse_transform([m])[0]

        # Calculates ROC-AUC, plots curve
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, m], pos_label=m)
        roc_auc = auc(fpr, tpr)
        ax_roc.plot(fpr, tpr, lw=2, label='Class {} (AUC = {:.2f})'.format(class_name, roc_auc))

        # Calculates and Plots Precision and Recall
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba[:, m], pos_label=m)
        ax_pr.plot(recall, precision, lw=2, label='Class {}'.format(class_name))

    # Formatting ROC-AUC
    ax_roc.plot([0, 1], [0, 1], linestyle='--', lw=2, color='black', label='Random')
    ax_roc.set_xlim([-0.05, 1.05])
    ax_roc.set_ylim([-0.05, 1.05])
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title('ROC-AUC Curve - {}'.format(model_name))
    ax_roc.legend(loc="lower right")

    # Formatting Precision-Recall
    ax_pr.set_xlim([-0.05, 1.05])
    ax_pr.set_ylim([-0.05, 1.05])
    ax_pr.set_xlabel('Recall')
    ax_pr.set_ylabel('Precision')
    ax_pr.set_title('Precision-Recall Curve - {}'.format(model_name))
    ax_pr.legend(loc="lower left")