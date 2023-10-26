import csv
import numpy as np
import matplotlib.pyplot as plt

# first dict keys : node_and_gpu
# second dict keys : job
# third dict keys : batch_size
# forth dict keys : "gpu_util", "mem_util", "iter"
logs = {}

with open("out.csv", "r") as f:
    reader = csv.reader(f)

    for row in reader:
        node_and_gpu = row[0] + "_" + row[1].replace(" ","_")
        job, batch_size, gpu_util, iteration = row[2], int(row[3]), float(row[4]), int(row[7])
        mem_util = float(row[6]) / 1000000.0        # change unit to GB

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
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        x_axis = list(logs[node_and_gpu][job].keys())
        x_axis.sort()
        y_axis_gpu_util, y_axis_mem_util, y_axis_iter = [], [], []
        for batch_size in x_axis:
            y_axis_gpu_util.append( np.mean(logs[node_and_gpu][job][batch_size]["gpu_util"]) )
            y_axis_mem_util.append( np.mean(logs[node_and_gpu][job][batch_size]["mem_util"]) )
            y_axis_iter.append( np.mean(logs[node_and_gpu][job][batch_size]["iter"]) )

        ax1.plot(x_axis, y_axis_gpu_util)
        ax2.plot(x_axis, y_axis_mem_util)
        ax3.plot(x_axis, y_axis_iter)

        plt.savefig("graphs/{}_{}.png".format(node_and_gpu, job))



# print(logs)