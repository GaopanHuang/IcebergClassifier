import numpy as np
import csv
import json

with open("./data/train.json",'r') as load_f:
  load_dict = json.load(load_f)

csvfile = open('./data/train.csv', 'w')
write = csv.writer(csvfile)

for i in range(len(load_dict)):
  if load_dict[i]['inc_angle'] == 'na':
    con = np.hstack((load_dict[i]['is_iceberg'],
        0,load_dict[i]['band_1'],load_dict[i]['band_2']))
  else:
    con = np.hstack((load_dict[i]['is_iceberg'],
        load_dict[i]['inc_angle'],load_dict[i]['band_1'],load_dict[i]['band_2']))
  write.writerow(con)
  print i
csvfile.close()
