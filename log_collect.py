import subprocess
import os
import time
import csv
import threading
import random

import zmq
import yaml

INIT = 1
HEADER = ['server', 'job', 'gpu_name', 'gpu_memory', 'gpu_memory_percent', 'gpu_util', 'batch_size', 'current_iteration']

def init_schedule(process_list):
    schedule_thread = threading.Thread(target=start_schedule, args=(process_list,), daemon=True)
    schedule_thread.start()

def start_schedule(process_list):
    import os
    import signal
    import time
    while True:
        for job in process_list:
            while True:
                out = job.stdout.readline()
                if out == "ready\n":
                    print('out:', out)
                    break
        for job in process_list:
            try:
                job.stdin.write("start\n")
                job.stdin.close()
            except:    
                pass
            print('job_start')
        # print("break complete!!")
        break
        

number_of_jobs = 2

node_list = ["gpu01", "gpu01", "gpu02", "gpu03"]
# model_list = [ "./vae", "./vision_transformer", "./Yolov8", "./mnist", "./mnist_rnn"]

model_list = ["vision_transformer", "Yolov8"]
Yolov8_model = ["yolov8n.pt", "yolov8m.pt", "yolov8l.pt", "yolov8s.pt", "yolov8x.pt"]
model_batch_dict = {
    # "./vae":[128, 256, 512, 1024, 2048, 4096], 
    "vision_transformer":[4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], 
    "yolov8n.pt":[4, 8, 16],
    "yolov8l.pt":[2, 4, 8],
    "yolov8m.pt":[4, 8, 16],
    "yolov8s.pt":[4, 8, 16],
    "yolov8x.pt":[2, 4, 8],
    # "./mnist":[32, 64, 128, 256, 512],
    # "./mnist_rnn":[32, 64, 128, 256, 512],
}



# "./vae",
# "./Yolov8/yolov8m", "./Yolov8/yolov8n", "./vae", "./vision_transformer",
# model_list = ["./Yolov8/yolov8m"]
run_list = ["run_diff_batchsize.sh", "run_diff_batch-size.sh", "run_diff_model_name.sh", "run_diff_batch-size.sh"]
run_file = "run.sh"
# model_list = ["./vae"]
# process_list = []

# while True:


# f = open('./out.csv', 'a')
# wr = csv.writer(f)
# wr.writerow(HEADER)
# f.close()
for i in range(10000):
    file = './out.csv'
    f = open('./out.csv', 'a')
    wr = csv.writer(f)
    wr.writerow('')
    wr.writerow(HEADER)
    f.close()
    run_model_list = []
    batch_size_list = []
    yolo_model_list = []
    for j in range(number_of_jobs):
        model = random.choice(model_list)
        run_model_list.append(model)
        if model == 'Yolov8':
            model = random.choice(Yolov8_model)
            yolo_model_list.append(model)
        else :
            yolo_model_list.append('')
        
        batch_size = random.choice(model_batch_dict[model])  
        batch_size_list.append(batch_size)
    for t in range(3):
        process_list = []
        # f = open(file, 'a')
        # wr = csv.writer(f)
        # wr.writerow('')
        # f.close()
        time.sleep(1)
        process_list.append(subprocess.Popen('srun --nodelist={} --gres=gpu:gpu1:1 ./models/{}/{} {} {}'.format(node_list[0], run_model_list[0], run_file, batch_size_list[0], yolo_model_list[0]), \
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, shell=True, text=True))
        for job in range(1, number_of_jobs):
            process_list.append(subprocess.Popen('srun --nodelist={} --gres=gpu:gpu0:1 ./models/{}/{} {} {}'.format(node_list[job], run_model_list[job], run_file, batch_size_list[job], yolo_model_list[job]), \
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, shell=True, text=True))
            # p2 = subprocess.Popen('srun --nodelist=gpu02 --gres=gpu:1 ./vae/run.sh ', stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, shell=True, text=True)

        for job in process_list:
            while True:
                out = job.stdout.readline()
                if out == "ready\n":
                    break
                # elif len(out) == 0:
                #     print("error!!")
                #     break 
            print('out :', out)
        for job in process_list:
            try:
                job.stdin.write("start\n")
                job.stdin.close()
            except:    
                pass
            print('job_start')

        for job in process_list:
            job.wait()

        f = open(file, 'a')
        wr = csv.writer(f)
        while True:
            for job in process_list:
                out = job.stdout.readline()    
                print(out)
                if "gpu" in out:
                    out = out.strip("\n")
                    log_list = out.split(',')
                    wr.writerow(log_list)
            if out == "done\n":
                break
        f.close()

print('done')

##################gpu 이름, gpu사용량, 메모리 사용량, iteration개수, batchsize, job 유형 .txt / .out파일로#############




