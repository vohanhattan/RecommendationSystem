import sys
import urllib
from csv import reader
import os.path
from urllib.request import urlretrieve

csv_filename = sys.argv[1]

with open(csv_filename+".csv".format(csv_filename), 'r',encoding="ISO-8859-1") as csv_file:
    for line in reader(csv_file):
        if os.path.isfile("posters/" + line[0] + ".jpg"):
            print ("Image skipped for {0}".format(line[0]))
        else:
            if line[5] != '' and line[0] != "imdbId":
                try:
                    urlretrieve(line[5], "posters/" + line[0] + ".jpg")
                    print ("Image saved for {0}".format(line[0]))
                except Exception:
                    pass
            else:
                print ("No result for {0}".format(line[0]))