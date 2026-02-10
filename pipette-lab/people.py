import matplotlib.pyplot as plt
import pandas 

data = pandas.read_csv("pipette-lab/people.csv", header=None)

data = data.transpose()

print(data[0])

labels = ['Dylan', 'Luke', 'Joseph']
colors = ['lightblue', 'lightgreen', 'lightcoral']



plt.boxplot(data, patch_artist=True, tick_labels=labels, medianprops=dict(color="orange", linewidth=2))
plt.plot([.5,3.5],[.5,.5], label = "target", color="red")
plt.ylabel("Output (ml)")
plt.title("People Comparison")
plt.legend()
plt.show()