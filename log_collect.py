import subprocess
import os
import time

INIT = 1

def init_schedule(process_list):
    import threading
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
        time.sleep(1)
        for job in process_list:
            try:
                job.stdin.write("start\n")
                job.stdin.close()
            except:    
                pass
            print('job_start')
        

number_of_jobs = 2
node_list = ["gpu01", "gpu02", "gpu01", "gpu02"]
model_list = ["./vision_transformer", "./vae","./vision_transformer", "./vae"]
# process_list = []

# while True:
process_list = []
f = open('./out.txt', 'a')
f.write("################# start ###################\n")
f.close()
for job in range(number_of_jobs):
    process_list.append(subprocess.Popen('srun --nodelist={} --gres=gpu:1 {}/run_diff_batchsize.sh'.format(node_list[job], model_list[0]), \
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, shell=True, text=True))
    # p2 = subprocess.Popen('srun --nodelist=gpu02 --gres=gpu:1 ./vae/run.sh ', stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, shell=True, text=True)

# for job in process_list:
#     init_schedule(job)
init_schedule(process_list)
    # while True:
    #     out = job.stdout.readline()
    #     # print(out == "ready\n")
    #     if out =="ready\n":
    #         break


# for job in process_list:
#     job.stdin.write("start\n")
#     job.stdin.close()

# for job in process_list:
#     out = job.stdout.readline()
#     print(out)
# p1.communicate(input="start")
# p2.communicate(input="start")

file = './out.csv'

while True:
    f = open(file, 'a')
    for job in process_list:
        out = job.stdout.readline()
        # print(out)        
        if ":" in out:
            


            f.write(out)
    f.close()
    if out == "done\n":
        break

for job in process_list:
    job.wait()

print('done')
time.sleep(1)
    # for model in model_list:
    #     file = './{}.txt'.format(model_list[i])
    #     f = open(file, 'r')
    #     f.readline()


##################gpu 이름, gpu사용량, 메모리 사용량, iteration개수, batchsize, job 유형 .txt / .out파일로#############




