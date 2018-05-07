import pickle
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
#"Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4", "METEOR", "ROUGE_L"
fields, scores = pickle.load( open( "figure.pkl", "rb" ) )
num_index = min(8, len(scores[0]))
num_plots = len(fields)
#colors = [plt.cm.spectral(i) for i in np.linspace(0, 1, num_plots)]
colors = ['red', 'green', 'blue', 'black','purple','orange']
fig = plt.figure()
ax = plt.subplot(111)
ax.set_prop_cycle(cycler('color', colors))
x = np.arange(num_index)
for i in range(1, num_plots):
    if i == 5:
        ax.plot(x, np.asarray(scores[i][:num_index])* 100)
    else:
        ax.plot(x, np.asarray(scores[i][:num_index]))
ax.legend(fields[1:], loc='best').draggable()
plt.title(r"Evalutation Results")
plt.grid()
plt.show()
