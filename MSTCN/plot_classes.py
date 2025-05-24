import numpy as np
import matplotlib.pyplot as plt

def plot_label_distribution(train_counts, test_counts, title="Label Distribution"):
    """
    Plots a grouped bar chart of train vs test label counts.

    Args:
        train_counts (list or np.ndarray): length-35 list of train set counts for classes 0…34
        test_counts  (list or np.ndarray): length-35 list of test set counts  for classes 0…34
        title (str): plot title
    """
    classes = np.arange(len(train_counts))  # 0,1,…,34
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(classes - width/2, train_counts, width, label='Train')
    ax.bar(classes + width/2, test_counts,  width, label='Test')

    ax.set_xlabel('Class')
    ax.set_ylabel('Number of Samples')
    ax.set_title(title)
    ax.set_xticks(classes)
    ax.set_xticklabels(classes, rotation=90)
    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # replace the [...] with your actual counts
    train_counts = [
        944, 2420, 2430,    0, 1461, 14517,  551,  8714,   0,  5688,
       9966,    0, 5711, 11892, 15765,     0,     0, 11653, 28312,   888,
       5568,    0, 706,     0,    0,    80, 11211,   147,   861,     0,
        551,    0, 1174,  7055,  8083
    ]
    test_counts = [
        120, 1053,    0,    0,  461,  4579,  732,  3458,    0,  4703,
       8418,    0, 1578,  3846,  4851,    0,    0,  1925, 19453,    0,
          0,    0,    0,    0,    0,    0,  1993,    0,    0,  168,
        356,    0,   72,  4495,  4432
    ]

    plot_label_distribution(train_counts, test_counts,
                            title="Train vs Test Label Distribution")
