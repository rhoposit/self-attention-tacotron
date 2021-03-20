import sys
import glob


sdir = "/home/smg/v-j-williams/workspace/external_modified/data/sys5_txt_source"
tdir = "/home/smg/v-j-williams/workspace/external_modified/data/sys5_txt_target"

sfiles = glob.glob(sdir+"/*.tfrecord")
tfiles = glob.glob(tdir+"/*.tfrecord")


trainfile = "train.csv"
input = open(trainfile, "r")
traindata = input.read().split("\n")[:-1]

validfile = "validation.csv"
input = open(validfile, "r")
validdata = input.read().split("\n")[:-1]

testfile = "test.csv"
input = open(testfile, "r")
testdata = input.read().split("\n")[:-1]


TRAIN, VALID, TEST = [], [], []
for f in traindata:
    sf = sdir+"/"+f+".source.tfrecord"
    tf = tdir+"/"+f+".target.tfrecord"
    if sf in sfiles:
        TRAIN.append(f)
    else:
        print("missing", f)


for f in validdata:
    sf = sdir+"/"+f+".source.tfrecord"
    tf = tdir+"/"+f+".target.tfrecord"
    if sf in sfiles:
        VALID.append(f)
    else:
        print("missing", f)


for f in testdata:
    sf = sdir+"/"+f+".source.tfrecord"
    tf = tdir+"/"+f+".target.tfrecord"
    if sf in sfiles:
        TEST.append(f)
    else:
        print("missing", f)

print("train:", len(TRAIN))
print("valid:", len(VALID))
print("test:", len(TEST))


outfile = "train_new.csv"
output = open(outfile, "w")
outstring = "\n".join(TRAIN)
output.write(outstring)
output.close()

outfile = "valid_new.csv"
output = open(outfile, "w")
outstring = "\n".join(VALID)
output.write(outstring)
output.close()

outfile = "test_new.csv"
output = open(outfile, "w")
outstring = "\n".join(TEST)
output.write(outstring)
output.close()
