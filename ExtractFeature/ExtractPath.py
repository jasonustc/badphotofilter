#! /usr/bin/env python

import os
import sys
import pdb
import shutil

rootDir = r'D:\BadPhoto.DataSet\DataSet.big\allphotos'
#rootDir = r'test'

outputDir = r'imgPath.txt'

def extract(rootDir,outputDir):
    if not os.path.isdir(rootDir):
        print 'wrong image path!'
        return
    dirs=os.listdir(rootDir)
    if os.path.isfile(outputDir):
        os.remove(outputDir)
    outputfile=open(outputDir,'w')
    for dir in dirs:
        #os.system('echo {0} >> {1}'.format(dirs, output_file))        
        path = os.path.join(rootDir,dir)
        print path
        if path[-4:].lower()=='.jpg':
            outputfile.writelines(path+"\n");
extract(rootDir,outputDir)
