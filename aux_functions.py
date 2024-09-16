import numpy as np
import matplotlib.pyplot as plt

def plot_label_distribution(y_train, y_val, y_test):
    labels = range(10)  # 0 to 9
    train_counts = [np.sum(y_train == i) for i in labels]
    val_counts = [np.sum(y_val == i) for i in labels]
    test_counts = [np.sum(y_test == i) for i in labels]
    
    train_percentages = [count / len(y_train) * 100 for count in train_counts]
    val_percentages = [count / len(y_val) * 100 for count in val_counts]
    test_percentages = [count / len(y_test) * 100 for count in test_counts]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width, train_percentages, width, label='Train')
    rects2 = ax.bar(x, val_percentages, width, label='Validation')
    rects3 = ax.bar(x + width, test_percentages, width, label='Test')

    ax.set_ylabel('Percentage')
    ax.set_xlabel('Digit')
    ax.set_title('Label Distribution in Train, Validation, and Test Sets')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    fig.tight_layout()
    plt.show()