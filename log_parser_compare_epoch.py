import csv
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab


params = {'legend.fontsize': 'x-large',
          'figure.figsize': (23, 8),
         'axes.labelsize': 20,
         'axes.titlesize':'x-large',
         'axes.labelweight':'bold',
         'xtick.labelsize': 18,
         'ytick.labelsize': 18}

pylab.rcParams.update(params)
# first dict keys : gpu_and_dataset
# second dict keys : epoch
# third dict keys : batch_size
# forth dict keys : "gpu_util", "mem_util", "iter"
logs = {}

with open("out_dcgan_preprocess.csv", "r") as f:
    reader = csv.reader(f)

    for row in reader:
        if len(row) < 1:
            continue
        dataset_name = row[3].split("-")
        dataset = dataset_name[0]
        job, epoch, batch_size, gpu_util, iteration = row[2].replace(".yaml",""), int(row[4]), int(row[5]), float(row[6]), int(row[9])
        gpu_and_dataset = row[1].replace(" ","_") + "_" + job + "_" + dataset
        mem_util = float(row[7]) / (1024*1024*1024)       # change unit to GB

        if gpu_and_dataset not in logs.keys():
            logs[gpu_and_dataset] = {}
        
        if epoch not in logs[gpu_and_dataset].keys():
            logs[gpu_and_dataset][epoch] = {}
        
        if batch_size not in logs[gpu_and_dataset][epoch].keys():
            logs[gpu_and_dataset][epoch][batch_size] = {"gpu_util": [], "mem_util" :[], "iter" : []}
        
        logs[gpu_and_dataset][epoch][batch_size]["gpu_util"].append(gpu_util)
        logs[gpu_and_dataset][epoch][batch_size]["mem_util"].append(mem_util)
        logs[gpu_and_dataset][epoch][batch_size]["iter"].append(iteration/10)

# draw graphs
for gpu_and_dataset in logs.keys():
    print("gpu_and_dataset : ", gpu_and_dataset)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(23, 8))
    plt.subplots_adjust(left=0.1,
                bottom=0.1, 
                right=0.95, 
                top=0.9, 
                wspace=0.2)

    # fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(10, 5))
    # plt.subplots_adjust(left=0.12,
    #             bottom=0.15, 
    #             right=0.97, 
    #             top=0.95, 
    #             wspace=0.25)

    x_axis = list(logs[gpu_and_dataset][0].keys())
    x_axis.sort()
    epochs = list(logs[gpu_and_dataset].keys())
    epochs.sort()
    x_labels = [str(i) for i in x_axis]
    
    color = ['bo-', 'ro-', 'go-']
    label = ['epoch_1', 'epoch_2', 'epoch_3']
    for epoch in epochs:
        y_axis_gpu_util, y_axis_mem_util, y_axis_iter = [], [], []
        y_axis_gpu_util_min, y_axis_mem_util_min, y_axis_iter_min = [], [], []
        y_axis_gpu_util_max, y_axis_mem_util_max, y_axis_iter_max = [], [], []
        
        for batch_size in x_axis:
            y_axis_gpu_util.append( np.mean(logs[gpu_and_dataset][epoch][batch_size]["gpu_util"]) )
            y_axis_gpu_util_min.append( np.min(logs[gpu_and_dataset][epoch][batch_size]["gpu_util"]) )
            y_axis_gpu_util_max.append( np.max(logs[gpu_and_dataset][epoch][batch_size]["gpu_util"]) )
            y_axis_mem_util.append( np.mean(logs[gpu_and_dataset][epoch][batch_size]["mem_util"]) )
            y_axis_mem_util_min.append( np.min(logs[gpu_and_dataset][epoch][batch_size]["mem_util"]) )
            y_axis_mem_util_max.append( np.max(logs[gpu_and_dataset][epoch][batch_size]["mem_util"]) )
            y_axis_iter.append( np.mean(logs[gpu_and_dataset][epoch][batch_size]["iter"]) )
            y_axis_iter_min.append( np.min(logs[gpu_and_dataset][epoch][batch_size]["iter"]) )
            y_axis_iter_max.append( np.max(logs[gpu_and_dataset][epoch][batch_size]["iter"]) )
        
        yerr_max = tuple(np.array(y_axis_iter_max) - np.array(y_axis_iter))
        yerr_min = tuple(np.array(y_axis_iter) - np.array(y_axis_iter_min))
        yerr = [yerr_min, yerr_max]
        # print(yerr)

        # make graph
        ax1.plot(range(len(x_axis)), y_axis_gpu_util,color[epoch], linewidth=2, label=label[epoch])
        ax2.plot(range(len(x_axis)), y_axis_mem_util,color[epoch], linewidth=2, label=label[epoch])
        ax3.plot(range(len(x_axis)), y_axis_iter, color[epoch], linewidth=2, label=label[epoch])
    ax1.legend()
    ax2.legend()
    ax3.legend()

    ax1.set_xlabel("Batch size")
    ax1.set_ylabel("Gpu utilization (%)")
    ax1.set_xticks(range(len(x_axis)))
    ax1.set_xticklabels(x_labels)
    ax1.set_ylim([0, 100])

    ax2.set_xlabel("batch size")
    ax2.set_ylabel("memory utilization (GiB)")
    ax2.set_xticks(range(len(x_axis)))
    ax2.set_xticklabels(x_labels)
    ax2.set_ylim([0, 25])

    ax3.set_xlabel("Batch size")
    ax3.set_ylabel("Iterations")
    ax3.set_xticks(range(len(x_axis)))
    ax3.set_xticklabels(x_labels)
    ax3.set_ylim([np.min(y_axis_iter_min), np.max(y_axis_iter_max)])

    plt.savefig("graphs_dataset/{}.png".format(gpu_and_dataset))
    plt.close(fig)
