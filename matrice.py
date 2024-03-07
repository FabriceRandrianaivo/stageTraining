import numpy as np
import matplotlib.pyplot as plt

dataset = {f"experience{i}": np.random.randn(100) for i in range(4)}

def graphique(dataset: dict):
    print(dataset)
    plt.figure()
    for i in dataset:
        plt.subplot(2, 2, 1)
        plt.plot(dataset["experience"+{i}])
        plt.title("graphique0")
    plt.show()

graphique(dataset)
