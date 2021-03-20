import sys, os, glob


infile = sys.argv[1]
outfile = infile+".revised"
existsfile = "vctk_files.txt"

input = open(infile, "r")
tacodata = input.read().split("\n")
input.close()

input = open(existsfile, "r")
existsdata = input.read().split("\n")
input.close()

existsdata = [item.split(".")[0] for item in existsdata]
tacodata_revised = list(set(tacodata) & set(existsdata))

if len(tacodata_revised) < len(tacodata):
    diff = len(tacodata) - len(tacodata_revised)
    print("had to remove:", diff)

outstring = "\n".join(tacodata_revised)
output = open(outfile, "w")
output.write(outstring)
output.close()
