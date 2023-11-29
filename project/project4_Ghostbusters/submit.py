# submit.py
# -------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to ShanghaiTech University, including a link 
# to https://i-techx.github.io/iTechX/courses?course_code=CS181


import tarfile
from projectParams import STUDENT_CODE_DEFAULT

def tar(fname, files):
    with tarfile.open(fname, "w") as tar:
        for f in files:
            tar.add(f)

tar("tracking.tar", STUDENT_CODE_DEFAULT.split(","))
print("""Successfully generated tar file. Please submit the generated tar file to
Autolab to receive credit.""")