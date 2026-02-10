import matplotlib.pyplot as plt
import pandas 

data = pandas.read_csv("pipette-lab/tool.csv", header=None)

data = data.transpose()

labels = ['man', 'aid']



plt.boxplot(data, patch_artist=True, tick_labels=labels, medianprops=dict(color="orange", linewidth=2))
plt.plot([.5,2.5],[.5,.5], label = "target", color="red")
plt.legend()
plt.title("Tool Comparison")
plt.ylabel("Output (ml)")
plt.show()