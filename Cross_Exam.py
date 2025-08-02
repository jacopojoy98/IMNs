import numpy as np
import os 
import matplotlib.pyplot as plt


def experiment(Results_dir):
    Experiment_list = os.listdir(Results_dir)
    Measures = ["Accuracy", "Precision", "Recall"]
    percentage = 0
    percentages = [0,10,20,30,40,50,60,70,80,90]
    percentages = [0,20,40,60,80,100]
    for Experiment in Experiment_list:
        C = 0
        current_experiment = os.path.join(Results_dir, Experiment)
        os.chdir(current_experiment)
        fig, ax = plt.subplots(1,3)
        for j, measure in enumerate(Measures):
            for percentage in percentages:
                if not os.path.isfile(measure+str(percentage)+".txt"):
                    C = 1
            if C:
                continue
            array = []
            stds = []
            for percentage in percentages:
                values = np.loadtxt(measure+str(percentage)+".txt")
                if measure == "Accuracy":
                    value = np.mean(values[-100:])
                    std = np.std(values[-100:])
                if measure == "Precision" or measure == "Recall":
                    values = np.mean(values, axis = 1)
                    value = np.mean(values[-100:])
                    std = np.std(values[-100:])
                array.append(value)
                stds.append(std)
            ax[j].errorbar(percentages, array, stds)
            ax[j].set_title(measure)
            fig.set_size_inches((36,10))
            plt.savefig("/home/jcolombini/Purpose/Labeler/DBG/IMG/"+Experiment+".png")

def folded_experiment(Experiment):
    Res_dir = "/home/jcolombini/Purpose/Labeler/Results/Labeler_results"
    Measures = ["Accuracy", "Precision", "Recall"]
    percentage = 0
    percentages = [0,10,20,30,40,50,60,70,80,90]
    percentages = [0,20]
    os.chdir(os.path.join(Res_dir,Experiment))
    fig, ax = plt.subplots(1,3)
    for j, measure in enumerate(Measures):
        C = 0
        for percentage in percentages:
            for fold in range(10):
                if not os.path.isfile(measure+str(percentage)+"f"+str(fold)+".txt"):
                    C = 1
        if C:
            continue
        array = []
        stds = []
        for percentage in percentages:
            m=0
            e=0
            for fold in range(10):
                values = np.loadtxt(measure+str(percentage)+"f"+str(fold)+".txt")
                if measure == "Accuracy":
                    value = np.mean(values[-100:])
                    std = np.std(values[-100:])
                if measure == "Precision" or measure == "Recall":
                    values = np.mean(values, axis = 1)
                    value = np.mean(values[-100:])
                    std = np.std(values[-100:])
                m+=value
                e+=std
            array.append(m/10)
            stds.append(e/10)
        ax[j].errorbar(percentages, array, stds)
        ax[j].set_title(measure)
        fig.set_size_inches((36,10))
        plt.savefig("/home/jcolombini/Purpose/Labeler/DBG/IMG/"+Experiment+".png")


if __name__ == "__main__":
    # for experime in os.listdir("/home/jcolombini/Purpose/Labeler/Results/Labeler_results"):
    experime="a_NFMLP2025-08-02-16-21E=700_LR=0.0001_NL=13"
    folded_experiment(experime)
    