import subprocess
import os

number_of_jobs = 2
node_list = ["gpu01", "gpu02", "gpu01", "gpu02"]
model_shell_list = ["./vision_transformer/run.sh"]
process_list = []
for job in range(number_of_jobs):
    process_list.append(subprocess.Popen('srun --nodelist={} --gres=gpu:1 {}'.format(node_list[job], model_shell_list[0]), \
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, shell=True, text=True))
    # p2 = subprocess.Popen('srun --nodelist=gpu02 --gres=gpu:1 ./vae/run.sh ', stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, shell=True, text=True)

for job in process_list:
    while True:
        out = job.stdout.readline()
        # print(out == "ready\n")
        if out =="ready\n":
            break


for job in process_list:
    job.stdin.write("start\n")
    job.stdin.close()

# for job in process_list:
#     out = job.stdout.readline()
#     print(out)
# p1.communicate(input="start")
# p2.communicate(input="start")

for job in process_list:
    job.wait()

print('done')

##################gpu 이름, gpu사용량, 메모리 사용량, iteration개수, batchsize, job 유형 .txt / .out파일로#############




