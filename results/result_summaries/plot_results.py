import matplotlib.pyplot as plt
import numpy as np


# path in computer and clusters
path_comp_moumita = "/media/moumita/Research/Files/University_Colorado/Work/work4/Spells_data_results/results/CNN/imp_results/graph/"
path_comp_brandon = ""
path_cluster = "/projects/mosa2108/spells/"

path = path_comp_moumita


x = np.array([1, 2, 3])
x_labels = ['class 1', 'class 2', 'class 3']

# all are kept in order as class1, class 2, class 3
precision_3layer_uwnd = [.87,.72,.83]
precision_3layer_vwnd = [.79,.73,.84]
precision_3layer_hgt = [.89,.74,.77]
precision_3layer_comb = [.91,.78,.85]

recall_3layer_uwnd = [.77,.9,.72]
recall_3layer_vwnd = [.82,.78,.74]
recall_3layer_hgt = [.79,.77,.79]
recall_3layer_comb = [.86,.85,.81]

fscore_3layer_uwnd = [.82,.8,.77]
fscore_3layer_vwnd = [.8,.75,.79]
fscore_3layer_hgt = [.84,.75,.77]
fscore_3layer_comb = [.88,.81,.83]

accuracy_3layer = [.79, .78, .78, .85]  # in order uwnd, vwnd, hgt, comb
correct_uwnd = [612,0,569]
correct_vwnd = [648,0,585]
correct_hgt = [624, 0, 629]
correct_comb = [679,0,644]

x_labels_accuracy =['uwnd', 'vwnd','hgt','uwnd+vwnd+hgt']
comp_1layers_3hist = [.87,.75,.85,0,.85,.85,.75,0,.86,.8,.8,0,.81] # all results for comb-- precision, recall, fscore, accuracy of all classes respectively
comp_3layers_3hist = [.91,.78,.85,0,.86,.85,.81,0,.88,.81,.83,0,.85]

#### Plot graph.................................................................
# plt.figure()
# plt.plot(x,precision_3layer_uwnd, marker = 'x', linestyle = '--', linewidth = '0.5', color = 'g', label= 'uwnd')
# plt.plot(x,precision_3layer_vwnd, marker = 'x', linestyle = '--', linewidth = '0.5',color = 'c', label= 'vwnd')
# plt.plot(x,precision_3layer_hgt, marker = 'x', linestyle = '--', linewidth = '0.5',color = 'brown', label= 'hgt')
# plt.plot(x,precision_3layer_comb, marker = 'X', linestyle = '--', color = 'b', label= 'uwnd+vwnd+hgt')
# plt.legend()
# plt.grid(axis='y', linestyle = ':', linewidth = '0.5')
#
# plt.xlabel('Daily rainfall classes', fontweight='bold')
# plt.ylabel('Precision', fontweight='bold')
# plt.xticks(x, x_labels, rotation=0)
# plt.title('Classification at lead 3 (pressure layers: 3, history of features: 3)')
# plt.tight_layout()
# plt.savefig(path+'precision_comp.png')
# plt.show()


#### Bar graph.................................................................
# plt.figure()
# plt.bar([1,2,3,4], accuracy_3layer, color=('g','c','brown','b'), width=0.6)
# plt.grid(axis='y', linestyle = ':', linewidth = '0.5')
# plt.xlabel('Variables', fontweight='bold')
# plt.ylabel('Overall Accuracy', fontweight='bold')
# plt.xticks([1,2,3,4], x_labels_accuracy, rotation=0)
# plt.title('Classification at lead 3 (pressure layers: 3, history of features: 3)')
# plt.tight_layout()
# plt.savefig(path+'accuracy_comp.png')
# plt.show()

# #### Group Bar graph.................................................................
# plt.figure()
#
# # set width of bar
# barWidth = 0.25
# # Set position of bar on X axis
# r1 = np.arange(len(correct_uwnd))
# r2 = [x + barWidth for x in r1]
# r3 = [x + barWidth for x in r2]
# r4 = [x + barWidth for x in r3]
#
#
# plt.bar(r1, correct_uwnd, color='g', width=barWidth, edgecolor='white', label='uwnd')
# plt.bar(r2, correct_vwnd, color='c', width=barWidth, edgecolor='white', label='vwnd')
# plt.bar(r3, correct_hgt, color='brown', width=barWidth, edgecolor='white', label='hgt')
# plt.bar(r4, correct_comb, color='b', width=barWidth, edgecolor='white', label='uwnd+vwnd+hgt')
#
# plt.grid(axis='y', linestyle = ':', linewidth = '0.5')
# plt.ylabel('Correctly classified samples', fontweight='bold')
# # Add xticks on the middle of the group bars
# plt.xlabel('Rainfall classes', fontweight='bold')
# plt.xticks([r + barWidth for r in range(len(correct_uwnd))], ['class 1', '', 'class 3'])
#
# # Create legend & Show graphic
# plt.legend()
# plt.title('Classification at lead 3 (pressure layers: 3, history of features: 3)')
# plt.tight_layout()
# plt.savefig(path+'correctly classified_comp.png')
# plt.show()


# #### Group Bar graph 2.................................................................
plt.figure()

# set width of bar
barWidth = 0.25
# Set position of bar on X axis
r1 = np.arange(len(comp_1layers_3hist))
r2 = [x + barWidth for x in r1]

plt.bar(r1, comp_1layers_3hist, color='gray', width=barWidth, edgecolor='white', label='pressure layer: 1')
plt.bar(r2, comp_3layers_3hist, color='b', width=barWidth, edgecolor='white', label='pressure layer: 3')

plt.grid(axis='y', linestyle = ':', linewidth = '0.5')
plt.ylabel('Classification measures', fontweight='bold')
# Add xticks on the middle of the group bars
plt.xlabel('precision                 recall                        f-score                 accuracy', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(comp_1layers_3hist))], ['class 1', 'class 2', 'class 3','', 'class 1', 'class 2', 'class 3','','class 1', 'class 2', 'class 3','','overall'], rotation = 45)

# Create legend & Show graphic
plt.legend(loc = 4)
plt.title('Classification at lead 3 (uwnd+vwnd+hgt)')
plt.tight_layout()
plt.savefig(path+'pressurelayers_comp.png')
plt.show()