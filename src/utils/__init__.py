from imgcat import imgcat
import matplotlib.pyplot as plt


def output_plot(path):
    plt.savefig(path)
    print()
    print(f"Plot saved to: {path}")
    print()
    imgcat(open(path))
    print()
