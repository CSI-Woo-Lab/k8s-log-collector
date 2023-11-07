import csv
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab


params = {'legend.fontsize': 'x-large',
          'figure.figsize': (15, 5),
         'axes.labelsize': 20,
         'axes.titlesize':'x-large',
         'axes.labelweight':'bold',
         'xtick.labelsize': 18,
         'ytick.labelsize': 18}

pylab.rcParams.update(params)
# first dict keys : node_and_gpu
# second dict keys : job
# third dict keys : batch_size
# forth dict keys : "gpu_util", "mem_util", "iter"
logs = {}

with open("out.csv", "r") as f:
    reader = csv.reader(f)

    for row in reader:
        if len(row) < 1:
            continue
        node_name_split = row[0].split("-")
        node_name = node_name_split[0]
        node_and_gpu = node_name + "_" + row[1].replace(" ","_")
        job, batch_size, gpu_util, iteration = row[2], int(row[3]), float(row[4]), int(row[7])
        mem_util = float(row[6]) / (1024*1024*1024)       # change unit to GB

        if node_and_gpu not in logs.keys():
            logs[node_and_gpu] = {}
        
        if job not in logs[node_and_gpu].keys():
            logs[node_and_gpu][job] = {}
        
        if batch_size not in logs[node_and_gpu][job].keys():
            logs[node_and_gpu][job][batch_size] = {"gpu_util": [], "mem_util" :[], "iter" : []}
        
        logs[node_and_gpu][job][batch_size]["gpu_util"].append(gpu_util)
        logs[node_and_gpu][job][batch_size]["mem_util"].append(mem_util)
        logs[node_and_gpu][job][batch_size]["iter"].append(iteration)

# draw graphs
for node_and_gpu in logs.keys():
    for job in logs[node_and_gpu].keys():
        # fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        # plt.subplots_adjust(left=0.1,
        #             bottom=0.1, 
        #             right=0.95, 
        #             top=0.9, 
        #             wspace=0.2)

        fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(10, 5))
        plt.subplots_adjust(left=0.12,
                    bottom=0.15, 
                    right=0.97, 
                    top=0.95, 
                    wspace=0.25)

        x_axis = list(logs[node_and_gpu][job].keys())
        x_axis.sort()
        x_labels = [str(i) for i in x_axis]
        y_axis_gpu_util, y_axis_mem_util, y_axis_iter = [], [], []
        y_axis_gpu_util_min, y_axis_mem_util_min, y_axis_iter_min = [], [], []
        y_axis_gpu_util_max, y_axis_mem_util_max, y_axis_iter_max = [], [], []

        for batch_size in x_axis:
            y_axis_gpu_util.append( np.mean(logs[node_and_gpu][job][batch_size]["gpu_util"]) )
            y_axis_gpu_util_min.append( np.min(logs[node_and_gpu][job][batch_size]["gpu_util"]) )
            y_axis_gpu_util_max.append( np.max(logs[node_and_gpu][job][batch_size]["gpu_util"]) )
            y_axis_mem_util.append( np.mean(logs[node_and_gpu][job][batch_size]["mem_util"]) )
            y_axis_mem_util_min.append( np.min(logs[node_and_gpu][job][batch_size]["mem_util"]) )
            y_axis_mem_util_max.append( np.max(logs[node_and_gpu][job][batch_size]["mem_util"]) )
            y_axis_iter.append( np.mean(logs[node_and_gpu][job][batch_size]["iter"]) )
            y_axis_iter_min.append( np.min(logs[node_and_gpu][job][batch_size]["iter"]) )
            y_axis_iter_max.append( np.max(logs[node_and_gpu][job][batch_size]["iter"]) )

        ax1.plot(range(len(x_axis)), y_axis_gpu_util,'bo-', linewidth=2)
        # ax2.plot(range(len(x_axis)), y_axis_mem_util,'bo-', linewidth=2)
        ax3.plot(range(len(x_axis)), y_axis_iter, 'bo-', linewidth=2)

        ax1.set_xlabel("Batch size")
        ax1.set_ylabel("Gpu utilization (%)")
        ax1.set_xticks(range(len(x_axis)))
        ax1.set_xticklabels(x_labels)
        ax1.set_ylim([0, 100])

        # ax2.set_xlabel("batch size")
        # ax2.set_ylabel("memory utilization (GiB)")
        # ax2.set_xticks(range(len(x_axis)))
        # ax2.set_xticklabels(x_labels)
        # ax2.set_ylim([0, 25])

        ax3.set_xlabel("Batch size")
        ax3.set_ylabel("Iterations")
        ax3.set_xticks(range(len(x_axis)))
        ax3.set_xticklabels(x_labels)
        ax3.set_ylim([0, 300])

        plt.savefig("graphs/{}_{}.png".format(node_and_gpu, job.strip(".yaml")))
        plt.close(fig)




print(logs)