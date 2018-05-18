import cv2
import numpy as np
from matplotlib import pyplot as plt
import glob
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 12
fig_size[1] = 9
plt.rcParams["figure.figsize"] = fig_size
np.set_printoptions(threshold=np.inf)
from natsort import natsorted
from PIL import Image, ImageEnhance, ImageStat

Pixel = []
index = []
count = 0

for idx, filename in enumerate(natsorted(glob.glob(r'D:\Wilharm\magnet philine\08.05.2018\frames\*.jpg'))):
    if idx > 0:

        img = Image.open(filename).convert('LA')
        stat = ImageStat.Stat(img)
        if stat.mean[0] < 24:
            print("black")
        if stat.mean[0] > 24:
            index.append(idx)
            print(filename)
            if stat.mean[0] == 85.77908859252929:
                pass
            if stat.mean[0] < 85.77908859252929:
                bright = ImageEnhance.Brightness(img)
                f = 85.77908859252929 / stat.mean[0]
                img = bright.enhance(f)
            if stat.mean[0] > 85.77908859252929:
                bright = ImageEnhance.Brightness(img)
                f = 85.77908859252929 / stat.mean[0]
                img = bright.enhance(f)
            img = img.rotate(0)
            img = img.crop((300, 100, 500, 300)) # l,o,r,u
            #img.show()
            threshold = 115
            im = img.point(lambda p: p > threshold and 255)


            im.save(r"D:\Wilharm\magnet philine\08.05.2018\Binary\frame%d.png" % count)  # save frame as JPEG file
            count += 1

            #im.show()
            Pixel.append(len(np.asarray(im)[np.asarray(im) == 255]))
    #if idx > 2000:
        #break


data_T = [26.9, 27.4, 27.9, 28.3, 28.9, 29.4, 29.9, 30.4, 30.8, 31.4, 31.9, 32.4, 32.9, 33.4,
          33.8, 34.3, 34.7, 35.1, 35.5, 35.9, 36.3, 36.5, 36.9, 37.2, 37.6, 37.8, 38.1, 38.3,
          38.6, 38.7, 39.1, 39.4, 39.6, 39.8, 39.8, 40.1, 40.2, 40.4, 40.6, 40.6, 40.8, 40.9,
          41.1, 41.2, 41.2, 41.4, 41.7, 41.7, 42.0, 42.0, 42.2, 42.4, 42.3, 42.4, 42.3, 42.0,
          42.0, 41.7, 41.4, 41.4, 41.5, 40.9, 40.5, 40.3, 40.0, 39.7, 39.3, 39.1, 38.9, 38.6,
          38.3, 38.0, 37.7, 37.5, 37.2, 36.8, 36.5, 36.3, 36.0, 35.6, 35.4, 35.2, 34.9, 34.7,
          33.3, 33.2, 33.1, 33.0, 32.9, 32.8, 32.7, 32.6, 32.5, 32.4, 32.3, 32.2, 32.1, 31.9,
          31.8, 31.7, 31.6, 31.5, 31.4, 31.3, 31.2, 31.1, 31.0, 31.0, 30.8, 30.8, 30.7, 30.6,
          30.5, 30.4, 30.3, 30.2, 30.1, 30.0, 30.0, 29.9, 29.8, 29.7, 29.6, 29.5, 29.4, 29.3,
          29.3, 29.2, 29.1, 29.0, 29.0, 28.9, 28.8, 28.7, 28.6, 28.6, 28.5, 28.5, 28.4, 28.3,
          28.2, 28.2, 28.1, 28.1, 28.0, 27.9, 27.8, 27.7, 27.6, 27.6, 27.5, 27.4, 27.4, 27.3,
          27.2]



#plt.figure(1)

#plt.subplot(211)
#plt.plot(Pixel)
#plt.xlabel('frame number', size = 20)
#plt.ylabel('pixel area of gel', color='k', size = 20)

#f = np.divide(len(Pixel), len(T))

#plt.subplot(212)
#plt.plot(np.multiply(x, f/2), T[index])
#plt.xlabel('frame number', size = 20)
#plt.ylabel('Temperature[°C]', color='k', size = 20)

plt.plot(Pixel)

plt.show()

