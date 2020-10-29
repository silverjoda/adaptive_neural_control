import os
import shutil

DELETE_AGENTS = True
DELETE_TB = True

filelist = [f for f in os.walk(os.path.join(os.path.dirname(os.path.realpath(__file__)), "algos/"))]
for root, directories, filenames in filelist:
    if ((root.endswith("agents") or root.endswith("agents_cp")) or root.endswith("agents_best") and DELETE_AGENTS) or (root.endswith("tb") and DELETE_TB):
        dirs = list(os.path.join(root, dir) for dir in directories if len(directories) > 0)
        files = list(os.path.join(root, files) for files in filenames if len(filenames) > 0)

        for f in files:
            os.remove(f)
        for d in dirs:
            shutil.rmtree(d)



