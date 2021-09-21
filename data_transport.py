"""把数据放到一个文件夹下的脚本"""
import os
import shutil

path="D:\高速运动关键点检测\data\数据标注\outdir"
dst="D:\高速运动关键点检测\data\Train"
f_dirs=os.listdir(path)
print(f_dirs)
i=111
for f_dir in f_dirs:
    files=os.listdir(os.path.join(path,f_dir))
    for file in files:
        if ".json" in file:
            filename=str(i)+'_'+file
            abs=os.path.join(path,f_dir,filename)
            os.rename(os.path.join(path, f_dir, file), abs)
            print(abs)
            shutil.copy(abs,dst)
            i+=1
