from random import shuffle
import json
file1=open("SMALL-CORPUS-WITH-TOPICS.txt", 'r')
lines=file1.readlines()
file1.close()
abs_t = []
for line in lines:
    abs_t.append(line)
shuffle(abs_t)
total = len(abs_t)
dev = total//10
train  = total - dev - dev
i = 0
file1=open("data/dev.dat", 'w')
for i in range(dev):
    file1.writelines(abs_t[i])
file1.close()
file1=open("data/test.dat", 'w')
for i in range(dev, 2 * dev):
    file1.writelines(abs_t[i])
file1.close()
file1=open("data/train.dat", 'w')
for i in range(2 * dev, total):
    file1.writelines(abs_t[i])
file1.close()
