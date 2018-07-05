import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_context("paper", rc={'font.sans-serif': 'Arial',
                             'font.size': 12})

df_green = pd.read_csv('~/Resources/Experiments/drstfn-13/drstfn-green/train/history.csv')
df_red = pd.read_csv('~/Resources/Experiments/drstfn-13/drstfn-red/train/history.csv')
df_nir = pd.read_csv('~/Resources/Experiments/drstfn-13/drstfn-nir/train/history.csv')

df_green = df_green.head(28)
df_red = df_red.head(28)
df_nir = df_nir.head(28)

epoch = df_green['epoch']
metrics = ('r2', 'val_r2')

labels = ('Green', 'Red', 'NIR')
colors = ('green', 'red', 'orange')
linestyles = ('-', '--')

fig, ax = plt.subplots()
for metric, linestyle in zip(metrics, linestyles):
    score = (df_green[metric], df_red[metric], df_nir[metric])
    for i in range(3):
        ax.plot(epoch + 1, score[i], label=labels[i], color=colors[i],
                linestyle=linestyle)

ax.set_xlabel('Epoch', fontsize=13)
ax.set_ylabel(r'$R^2$', fontsize=13)

ax.tick_params(axis='both', which='both', labelsize=11)
ax.set_xticks(range(0, epoch.size + 1, 10))
ax.set_ylim([0.5, 0.9])
ax.grid(True, color=(0.95, 0.95, 0.95))

for i in range(2):
    ax.plot([], [], color='black', linestyle=linestyles[i])
ax.grid(True)
lines = ax.get_lines()
color_legend = ax.legend(handles=[lines[i] for i in range(3)], labels=labels,
                         loc=4, bbox_to_anchor=(0.98, 0.05), fontsize=11, frameon=False)
line_legend = ax.legend(handles=[lines[i] for i in range(-2, 0)], labels=('Training', 'Validation'),
                        loc=4, bbox_to_anchor=(0.76, 0.085), fontsize=11, frameon=False)
ax.add_artist(color_legend)
ax.add_artist(line_legend)
ax.set_title('Fitted Curve', fontsize=15, fontweight='bold')

plt.savefig('fit.eps')
plt.close()
