import matplotlib.pyplot as plt
import pandas 

data = pandas.read_csv("pipette-lab/tool.csv", header=None)

data = data.transpose()

labels = ['water', '20% gly']

fig, ax1 = plt.subplots()

# ax2 = ax1.twinx();



ax1.boxplot([data[0]/.5, data[1]/.5260], patch_artist=True, tick_labels=labels, medianprops=dict(color="orange", linewidth=2))
# ax1.boxplot(, patch_artist=True, tick_labels=[labels[1]], medianprops=dict(color="orange", linewidth=2))
ax1.plot([.5,2.5],[1,1], label = "target", color="red")
plt.ylabel("Proportion of target")
plt.title("Liquid Comparison")
plt.legend()
plt.show()