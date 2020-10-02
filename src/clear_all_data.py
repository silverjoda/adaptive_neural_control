import os.path
import os
import pathlib
AGENTS = True
TB = True

# everything = pathlib.Path().glob("algos/**")
# print(list(everything))
# exit()

filelist = [f for f in os.walk(os.path.join(os.path.dirname(os.path.realpath(__file__)), "algos/"))]
for root, directories, filenames in filelist:
    #os.remove(f)
    if root.endswith("agents") or root.endswith("agents_cp") or root.endswith("tb"):
        dirs = list(os.path.join(root, dir) for dir in directories if len(directories) > 0)
        files = list(os.path.join(root, files) for files in filenames if len(filenames) > 0)

        for f in files:
            print(f)
        for d in dirs:
            print(d)



