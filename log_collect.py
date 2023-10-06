import subprocess
import os


p1 = subprocess.Popen('srun --nodelist=gpu01 --gres=gpu:1 ./vision_transformer/run.sh', stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, shell=True, text=True)
p2 = subprocess.Popen('srun --nodelist=gpu02 --gres=gpu:1 ./vae/run.sh ', stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, shell=True, text=True)

p1.wait(timeout=5)
p2.wait(timeout=5)

p1.communicate(input="start")
p2.communicate(input="start")


##################gpu 이름, gpu사용량, 메모리 사용량, iteration개수, batchsize, job 유형 .txt / .out파일로#############




