from matplotlib.colors import LogNorm, PowerNorm, Normalize
from rayoptics.util.misc_math import normalize
from contextlib import redirect_stdout
from deap import base, creator, tools, algorithms
from rayoptics.environment import *
import numpy as np
import random
isdark = False
import re
import io
from scipy.optimize import minimize

from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


def calc_loss_mute(model):
    efl_for_loss=5                      #mm
    fD_for_loss=2.1
    total_length_for_loss=7.0             #mm
    radius_enclosed_energy_for_loss=50    #micron
    perc_max_enclosed_energy_for_loss=80    #%
    perc_min_enclosed_energy_for_loss=50    #%
    min_thickness_for_loss=0.1              #mm
    min_thickness_air_for_loss=0.0            #mm
    number_of_field=5
    number_of_wavelength=2

    def funct_loss_enclosed_energy(enclosed_energy,perc_max_enclosed_energy_for_loss,perc_min_enclosed_energy_for_loss):
        if enclosed_energy<perc_max_enclosed_energy_for_loss:
            if enclosed_energy<perc_min_enclosed_energy_for_loss:
                loss_enclosed_energy=1e3
            else:
                loss_enclosed_energy=(perc_max_enclosed_energy_for_loss-enclosed_energy)
        else:
            loss_enclosed_energy=0
        return loss_enclosed_energy

    def get_thichness(sm):
        f = io.StringIO()
        with redirect_stdout(f):
            sm.list_model()
        s = f.getvalue()
        rows = re.split(r"\n", s)
        thickness_list = []
        thickness_material_list=[]
        thickness_air_list=[]
        for row in rows[1:-1]:
            row = re.sub(r'\s+',r'!', row)
            values = re.split(r"!", row)
            if values[4]!='air' and values[4]!='1':
                thickness_material_list.append(float(values[3]))
            if values[4]=='air' and values[4]!='1':
                thickness_air_list.append(float(values[3]))
            thickness_list.append(float(values[3]))      #3 - thickness, 2 - curvature, 4 - type of material
        number_of_surfaces=len(rows)-2
        return thickness_list, thickness_material_list, thickness_air_list, number_of_surfaces

    opm = model#open_model(f'{path2model}', info=True)

    sm = opm['seq_model']
    osp = opm['optical_spec']
    pm = opm['parax_model']
    em = opm['ele_model']
    pt = opm['part_tree']
    ar = opm['analysis_results']

    pm.__dict__

    efl=pm.opt_model['analysis_results']['parax_data'].fod.efl
    fD=pm.opt_model['analysis_results']['parax_data'].fod.fno


    ax_ray, pr_ray, fod = ar['parax_data']
    u_last = ax_ray[-1][mc.slp]
    central_wv = opm.nm_to_sys_units(sm.central_wavelength())
    n_last = pm.sys[-1][mc.indx]
    to_df = compute_third_order(opm)

    tr_df=to_df.apply(to.seidel_to_transverse_aberration, axis='columns', args=(n_last,u_last))
    distortion=tr_df.to_numpy()[-1,5]

    field=0
    psf = SpotDiagramFigure(opm)
    test_psf = psf.axis_data_array[field][0][0][0]
    test_psf[:,1]=test_psf[:,1]-np.mean(test_psf[:,1])


    fld, wvl, foc = osp.lookup_fld_wvl_focus(0)
    efl=pm.opt_model['analysis_results']['parax_data'].fod.efl
    opm.update_model()

    if abs(efl-efl_for_loss)>0.25:
        loss_focus=1e2*(efl-efl_for_loss)**2
    else:
        loss_focus=0

    if abs(fD)>=fD_for_loss:
        loss_FD=5*1e4*(fD-fD_for_loss)**2
    else:
        loss_FD=0


    thickness_list,thickness_material_list,thickness_air_list, number_of_surfaces=get_thichness(sm)
    #print(thickness_list)
    total_length=np.sum(thickness_list[1:])

    min_thickness=np.min(thickness_material_list)
    min_thickness_air=np.min(thickness_air_list)
    if (total_length-total_length_for_loss)>0:
        loss_total_length=1e4*(total_length-total_length_for_loss)**2
    else:
        loss_total_length=0

    if min_thickness<min_thickness_for_loss:
        loss_min_thickness=1e6*(min_thickness-min_thickness_for_loss)**2
    else:
        loss_min_thickness=0

    if min_thickness_air<min_thickness_air_for_loss:
        loss_min_thickness_air=8e4*(min_thickness_air-min_thickness_air_for_loss)**2
    else:
        loss_min_thickness_air=0


    loss_enclosed_energy_all=0
    loss_rms_all=0
    temp=0
    for idx_field in range(number_of_field):
        for idx_wavelength in range(number_of_wavelength):
            test_psf = psf.axis_data_array[idx_field][0][0][idx_wavelength]
            test_psf[:,1]=test_psf[:,1]-np.mean(test_psf[:,1])
            r_psf=np.sort(np.sqrt(test_psf[:,0]**2+test_psf[:,1]**2))
            enclosed_energy=100*np.sum(r_psf<=radius_enclosed_energy_for_loss/1e3)/len(test_psf[:,0])
            loss_enclosed_energy=funct_loss_enclosed_energy(enclosed_energy,perc_max_enclosed_energy_for_loss,perc_min_enclosed_energy_for_loss)
            loss_enclosed_energy_all=loss_enclosed_energy_all+loss_enclosed_energy

            dl=int(np.floor(len(test_psf[:,0])*perc_max_enclosed_energy_for_loss/100))
            loss_rms=np.sqrt(np.sum((1e3*r_psf[:dl])**2)/dl)
            loss_rms_all=loss_rms_all+loss_rms

            temp=temp+1
    loss_enclosed_energy_all=loss_enclosed_energy_all/temp
    loss_rms_all=loss_rms_all/temp
    loss=loss_focus+loss_FD+loss_total_length+loss_min_thickness+loss_min_thickness_air+loss_enclosed_energy_all+loss_rms_all
    return(loss)


def evaluate_system(system):
    try:
        opm = OpticalModel() # create new model

        sm = opm['seq_model']
        osp = opm['optical_spec']
        pm = opm['parax_model']
        em = opm['ele_model']
        pt = opm['part_tree']
        ar = opm['analysis_results']

        opm.system_spec.title = 'Test Model'
        opm.system_spec.dimensions = 'mm'

        osp['pupil'] = PupilSpec(osp, value=2.5)
        osp['fov'] = FieldSpec(osp, key=['object', 'angle'], is_relative=False, flds=[0., 5., 10., 15., 20.])
        osp['wvls'] = WvlSpec([(470, 1.0), (650, 1.0)], ref_wl=0)

        sm.do_apertures = False
        sm.gaps[0].thi=1e10

        # 1 surface - lens
        curvature_1 = 0.2747823174694503
        t_1 = system[0] # 1
        k_1 = system[1] # 1
        medium_1_1 = 1.54 * k_1 + 1.67 * (1 - k_1) # 1.540
        medium_2_1 = 75.0 * k_1 + 39.0 * (1 - k_1) # 75.0
        
        sd_1 = system[2] # 0.12499000001
        r_1 = system[3] # 3.639244363353831
        coefs_1 = [0.0, 0.009109298409282469, -0.03374649200850791, 0.01797256809388843, -0.0050513483804677005, 0.0, 0.0, 0.0]

        sm.add_surface([curvature_1, t_1, medium_1_1, medium_2_1], sd=sd_1)
        sm.ifcs[sm.cur_surface].profile = EvenPolynomial(r=r_1, coefs=coefs_1)

        sm.set_stop()

        # 2 surface - air
        curvature_2 = 0.13556582944950138
        t_2 = system[4] # 0.5
        sd_2 = system[5] # 1.4607610829755018
        r_2 = system[6] # 7.476060108611791
        coefs_2 = [0.0, -0.002874728268075267, -0.03373322938525211, 0.004205227876537139, -0.0001705765222318475, 0.0, 0.0, 0.0]

        sm.add_surface([curvature_2, t_2], sd=sd_2)
        sm.ifcs[sm.cur_surface].profile = EvenPolynomial(r=r_2,
                                coefs=coefs_2)
        sm.ifcs[sm.cur_surface].profile.sd = sd_2


        # 3 surface - lens
        curvature_3 = -0.055209803982245384
        t_3 = system[7] # 1
        k_2 = system[8] # 0
        medium_1_2 = 1.54 * k_2 + 1.67 * (1 - k_2) # 1.670
        medium_2_2 = 75.0 * k_2 + 39.0 * (1 - k_2) # 39.0

        sd_3 = system[9] # 1.4312127337246845
        r_3 = system[10] # -17.751725057339173
        coefs_3 = [0.0, -0.0231369463217776, 0.011956554928461116, -0.017782670650182023, 0.004077846642272649, 0.0, 0.0, 0.0]

        sm.add_surface([curvature_3, t_3, medium_1_2, medium_2_2], sd=sd_3)
        sm.ifcs[sm.cur_surface].profile = EvenPolynomial(r=r_3,
                                coefs=coefs_3)
        sm.ifcs[sm.cur_surface].profile.sd = sd_3

        # 4 surface - air
        curvature_4 = system[11] #-0.2568788474926888
        t_4 = system[12] # 4.215892884493065
        sd_4 = system[13] # 1.608772352457493

        sm.add_surface([curvature_4, t_4], sd=sd_4)
        sm.ifcs[sm.cur_surface].profile.sd = sd_4

        opm.update_model()

        return calc_loss_mute(opm)
    
    except: return 1000000


def save_system(system, do_draw=True, path='result.roa'):
    
    opm = OpticalModel() # create new model

    sm = opm['seq_model']
    osp = opm['optical_spec']
    pm = opm['parax_model']
    em = opm['ele_model']
    pt = opm['part_tree']
    ar = opm['analysis_results']

    opm.system_spec.title = 'Test Model'
    opm.system_spec.dimensions = 'mm'

    osp['pupil'] = PupilSpec(osp, value=2.5)
    osp['fov'] = FieldSpec(osp, key=['object', 'angle'], is_relative=False, flds=[0., 5., 10., 15., 20.])
    osp['wvls'] = WvlSpec([(470, 1.0), (650, 1.0)], ref_wl=0)

    sm.do_apertures = False
    sm.gaps[0].thi=1e10

    # 1 surface - lens
    curvature_1 = 0.2747823174694503
    t_1 = system[0] # 1
    k_1 = system[1] # 1
    medium_1_1 = 1.54 * k_1 + 1.67 * (1 - k_1) # 1.540
    medium_2_1 = 75.0 * k_1 + 39.0 * (1 - k_1) # 75.0
    
    sd_1 = system[2] # 0.12499000001
    r_1 = system[3] # 3.639244363353831
    coefs_1 = [0.0, 0.009109298409282469, -0.03374649200850791, 0.01797256809388843, -0.0050513483804677005, 0.0, 0.0, 0.0]

    sm.add_surface([curvature_1, t_1, medium_1_1, medium_2_1], sd=sd_1)
    sm.ifcs[sm.cur_surface].profile = EvenPolynomial(r=r_1, coefs=coefs_1)

    sm.set_stop()

    # 2 surface - air
    curvature_2 = 0.13556582944950138
    t_2 = system[4] # 0.5
    sd_2 = system[5] # 1.4607610829755018
    r_2 = system[6] # 7.476060108611791
    coefs_2 = [0.0, -0.002874728268075267, -0.03373322938525211, 0.004205227876537139, -0.0001705765222318475, 0.0, 0.0, 0.0]

    sm.add_surface([curvature_2, t_2], sd=sd_2)
    sm.ifcs[sm.cur_surface].profile = EvenPolynomial(r=r_2,
                            coefs=coefs_2)
    sm.ifcs[sm.cur_surface].profile.sd = sd_2


    # 3 surface - lens
    curvature_3 = -0.055209803982245384
    t_3 = system[7] # 1
    k_2 = system[8] # 0
    medium_1_2 = 1.54 * k_2 + 1.67 * (1 - k_2) # 1.670
    medium_2_2 = 75.0 * k_2 + 39.0 * (1 - k_2) # 39.0

    sd_3 = system[9] # 1.4312127337246845
    r_3 = system[10] # -17.751725057339173
    coefs_3 = [0.0, -0.0231369463217776, 0.011956554928461116, -0.017782670650182023, 0.004077846642272649, 0.0, 0.0, 0.0]

    sm.add_surface([curvature_3, t_3, medium_1_2, medium_2_2], sd=sd_3)
    sm.ifcs[sm.cur_surface].profile = EvenPolynomial(r=r_3,
                            coefs=coefs_3)
    sm.ifcs[sm.cur_surface].profile.sd = sd_3

    # 4 surface - air
    curvature_4 = system[11] #-0.2568788474926888
    t_4 = system[12] # 4.215892884493065
    sd_4 = system[13] # 1.608772352457493

    sm.add_surface([curvature_4, t_4], sd=sd_4)
    sm.ifcs[sm.cur_surface].profile.sd = sd_4

    opm.update_model()

    if do_draw:
        isdark = False
        # 1 plot
        layout_plt0 = plt.figure(FigureClass=InteractiveLayout, opt_model=opm,do_draw_rays=True, do_paraxial_layout=False,is_dark=isdark).plot()
        plt.show()
        '''
        # 2 plot
        spot_plt = plt.figure(FigureClass=SpotDiagramFigure, opt_model=opm,
                      scale_type=Fit.All_Same, dpi=200, is_dark=isdark).plot()
        plt.show()
        '''
        # 3 plot
        to_pkg = compute_third_order(opm)
        fig, ax = plt.subplots()
        ax.set_xlabel('Surface')
        ax.set_ylabel('third order aberration')
        ax.set_title('Surface by surface third order aberrations')
        to_pkg.plot.bar(ax=ax, rot=0)
        ax.grid(True)
        fig.tight_layout()

    opm.save_model(path)


def parameter_constraint(system, i):
    # Здесь проверьте значения параметров и верните True, если они удовлетворяют ограничениям, иначе False
    if i in [0, 4, 7, 12]:                          # t
        if system[i] > 0.005 and system[i] < 2.0:
            return True
        else:
            return False  

    if i in [1, 8]:                                 # k
        if system[i] > 0.0 and system[i] < 1.0:
            return True
        else:
            return False

    if i in [2, 5, 9, 13]:                          # sd
        if system[i] > 0.05 and system[i] < 10.0:
            return True
        else:
            return False
        
    if i in [3, 6, 10, 11]:                          # r
        if system[i] > 0.1 and system[i] < 20.0 or system[i] > -20.0 and system[i] < -0.1:
            return True
        else:
            return False



# Начальное приближение для параметров системы
initial_guess = [1.0, 1.0, 0.12499000001, 3.639244363353831, 0.5, 1.4607610829755018, 7.476060108611791, 1.0, 0.0, 1.4312127337246845, -17.751725057339173, -0.2568788474926888, 4.215892884493065, 1.608772352457493]

# Параметры оптимизации
optimization_params = {'method': 'BFGS', 'tol': 1e-6, 'options': {'maxiter': 10, 'disp': True}}

print(evaluate_system(initial_guess))

# начальный шаг поиска для каждого параметра
# t, k, sd, r, t, sd, r, t, k, sd, r, c, t, sd
init_steps = [0.2, 0.2, 0.1, 2, 0.2, 0.1, 2, 0.2, 0.2, 0.1, 2, 2, 0.2, 0.1]

# новая система
new_system = initial_guess.copy()

# во сколько раз будем уменьшать шаг поиска каждый раз
step_decrease_factor = 2.0

'''
for i in tqdm(range(len(initial_guess))):
    
    step_decreased = 0

    while step_decreased < 50:

        best_loss = evaluate_system(new_system)
        cur_step = init_steps[i]
        
        pos_system = new_system.copy()
        neg_system = new_system.copy()

        pos_system[i] += cur_step
        neg_system[i] -= cur_step

        if parameter_constraint(pos_system, i):
            pos_loss = evaluate_system(pos_system)
        else:
            pos_loss = best_loss + 1.0

        if pos_loss < best_loss:
            new_system = pos_system.copy()
            best_loss = pos_loss.copy()
        else:
            if parameter_constraint(neg_system, i):
                neg_loss = evaluate_system(neg_system)
            else:
                neg_loss = best_loss + 1.0
            
            if neg_loss < best_loss:
                new_system = neg_system.copy()
                best_loss = neg_loss.copy()
            else:
                init_steps[i] /= step_decrease_factor
                step_decreased += 1


print(new_system)

print(evaluate_system(new_system))

'''

save_system([1.217980256450714, 0.9996882556514192, 0.12499000001, 3.63924436335376, 0.49872483183280575, 1.4607610829755018, 7.4715849675421575, 1.0000000000011426, 3.637978807091713e-13, 1.4312127337246845, -17.751725057339186, -0.2568788474926959, 4.215892884493065, 1.608772352457493], path='0.6750401902049818.roa')