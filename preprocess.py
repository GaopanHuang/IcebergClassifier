import numpy as np
import time
import json
import matplotlib.pyplot as plt


with open("./data/train.json",'r') as load_f:
  load_dict = json.load(load_f)

fig = plt.figure(0)

i = 0
t = -15
while (i < 1604/4):
  plt.figure(figsize=(100, 50))
  hh = np.array(load_dict[i]['band_1']).reshape((75,75))
  hv = np.array(load_dict[i]['band_2']).reshape((75,75))
  hht = hh[:,:]>t
  hvt = hv[:,:]>t

  plt.subplot(221)
  plt.title("%s: %d  (%s)" % (load_dict[i]['id'],load_dict[i]['is_iceberg'], 
      load_dict[i]['inc_angle']))
  plt.imshow(hh)
  plt.subplot(222); plt.imshow(hv)
  plt.subplot(223); plt.imshow(hht)
  plt.subplot(224); plt.imshow(hvt)
  plt.show()
  i += 1