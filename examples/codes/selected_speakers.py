import glob
import sys
import os

infile = sys.argv[1]
datfile = sys.argv[2]

input = open(datfile, "r")
data = input.read().split("\n")[:-1]
input.close()

SPK = ["p"+d.split(" ")[0] for d in data]
print(SPK)
KEEP = []

input = open(infile, "r")
data = input.read().split("\n")[:-1]
input.close()

outfile = infile+".selected"

for d in data:
    speaker = d.split("_")[0]
#    print(speaker)
    if speaker in SPK:
        KEEP.append(d)

output = open(outfile, "w")
outstring = "\n".join(KEEP)
output.write(outstring)
output.close()
