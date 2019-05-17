from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
def plot_cm(confusion_matrix, label_tuple):
    cmap = matplotlib.cm.Greens
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)  
    ms = plt.matshow(confusion_matrix, cmap=cmap, norm=norm)
    plt.colorbar(ms)
    plt.ylabel(u'True Label',fontsize=16)
    plt.xlabel(u'Predicted Label',fontsize=16,labelpad = 14)

    ax=plt.gca()
    ax.set_xticks(np.linspace(0,4,5))
    ax.set_yticks(np.linspace(0,4,5)) 
    ax.set_xticklabels( label_tuple)
    ax.set_yticklabels( label_tuple)

    for i in range(len(confusion_matrix)):
        for j in range(len(confusion_matrix[i])): 
            plt.text(i, j, "%0.2f" %(confusion_matrix[j][i],), color='black', fontsize=18, va='center', ha='center')
    plt.gcf().set_size_inches(7,5)
    plt.savefig('CM.eps')
    plt.show()
    

if __name__ == '__main__':
    confusion_matrix=[[0.30, 0.20, 0.17, 0.13, 0.20],[0.06, 0.67, 0.21, 0.04, 0.02],[0.10, 0.14, 0.57, 0.13, 0.06],[0.04, 0.08, 0.12, 0.69,  0.07],[0.10, 0.07, 0.25, 0.12, 0.46]]
    label_tuple = ('Disgusted', 'Scared', 'Sad', 'Neutral' ,'Positive')
    plot_cm(confusion_matrix, label_tuple)
