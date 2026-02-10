import matplotlib.pyplot as plt

data_1 = [
    [390, 1351, 390, 1351],
    [372, 490, 372, 490],
    [386, 504, 386, 504],
    [393, 1354, 393, 1354],
    [389, 1350, 389, 1350],
    [365, 1302, 365, 1302]
]
data_2 = [
    [],
    [285, 315, 285, 315],
    [],
    [306, 1023, 306, 1023],
    [302, 1019, 302, 1019],
    [278, 308, 278, 308]
]



# print(data[0])

labels = ['A1', 'A2', 'A3', 'A5', 'A6', 'A7']

plt.boxplot(data_1, patch_artist=True, tick_labels=labels, orientation='horizontal', medianprops=dict(color="orange", linewidth=0))
plt.ylabel("Strand")
plt.xlabel("Expected Binding Positions")
plt.title("First Primer Pair")
plt.xlim(0, 1400)
# plt.legend()
plt.show()

plt.boxplot(data_2, patch_artist=True, tick_labels=labels, orientation='horizontal', medianprops=dict(color="orange", linewidth=0))
plt.ylabel("Strand")
plt.xlabel("Expected Binding Positions")
plt.title("Second Primer Pair")
plt.xlim(0, 1400)
# plt.legend()
plt.show()