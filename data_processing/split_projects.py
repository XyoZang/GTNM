import os
import random

# train_project_num = 9000
# valid_project_num = 200
# test_project_num = 1022

# projects = os.listdir("data/")


projects = open("data/repos.txt",'r').readlines()
n = len(projects)
print('total projects: {}'.format(n))
random.shuffle(projects)

wf = open("data/train_projects.txt", "w")
for x in projects[:9000]:
    wf.write(x)
wf.close()

wf = open("data/eval_projects.txt", "w")
for x in projects[9000:9200]:
    wf.write(x)
wf.close()

wf = open("data/test_projects.txt", "w")
for x in projects[9200:]:
    wf.write(x)
wf.close()

train_projects = open("data/train_projects.txt", "r").readlines()
wf = open("data/small_train_projects.txt", "w")
for x in train_projects[:4000]:
    wf.write(x)
wf.close()
