from ray_optics_criteria_ITMO import calc_loss
import time



t0=time.time()
path2model= '/home/oleg/hackathons/Optics-Hackathon/test.roa' # сюда надо вставлять полный путь (или я криворукий, можете у вас получиться)
# так же скорее всего этот файл лучше занести в gitignore, если это так то в будующем уберу
loss=calc_loss(path2model) # так же кажеться что эту функцию лучше не менять и скорее всего по ней и строиться список лидеров.
# менять мы должны только test.roa
elapsed_time=time.time()-t0
print(f'{loss=}, {elapsed_time=} sec')