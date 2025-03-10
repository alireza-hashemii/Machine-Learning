import numpy as np
import matplotlib.pyplot as plt 


# calculating gini impurity metric for 
# thousand numbers evenly spaced from 0 to 1
pos_fraction = np.linspace(0.00, 1.00, 1000)
gini = 1 - pos_fraction**2 - (1-pos_fraction) **2

# plot
# plt.plot(pos_fraction, gini)
# plt.ylim(0, 1)
# plt.xlabel('Positive fraction')
# plt.ylabel('Gini impurity')
# plt.show()



# gini impurity calculation
def gini_impurity(labels):
    if len(labels) == 0:
        return 0
    # count the occurences of each label
    counts = np.unique(labels, return_counts=True)[1]
    fractions = counts / float(len(labels))
    return 1 - np.sum(fractions ** 2)



    
res = gini_impurity([1, 1, 0, 1, 0])
print(f'{res:.4f}')
