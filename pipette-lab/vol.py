import matplotlib.pyplot as plt
import pandas 

data = pandas.read_csv("pipette-lab/vol.csv", header=None)

data = data.transpose()

labels = ['.5ml Target', '.2ml Target']



plt.boxplot(data - [.5,.2], patch_artist=True, tick_labels=labels, medianprops=dict(color="orange", linewidth=2))
plt.plot([.5,2.5],[0,0], label = "target", color="red")
plt.ylabel("Difference from target (ml)")
plt.title("Volume Comparison")
plt.legend()
plt.show()