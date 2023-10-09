from matplotlib.colors import LogNorm, PowerNorm, Normalize
from rayoptics.util.misc_math import normalize
from contextlib import redirect_stdout
from deap import base, creator, tools
from rayoptics.environment import *
from deap import base, algorithms
from deap import tools
import numpy as np
import random
isdark = False
import re
import io

def calc_loss_mute(path2model):
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

    opm = path2model#open_model(f'{path2model}', info=True)

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
    #sm.list_model()
    #sm.list_surfaces()
    efl=pm.opt_model['analysis_results']['parax_data'].fod.efl

    #pm.first_order_data()
    opm.update_model()

    # total_length=0
    # min_thickness=0.15
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

def generate_random_system(fine_tune_system, is_first_lens):
    """1) реализовать определение границ внутри этой системы 
    2) добавить аргумент,который определяет что линза последняя, для нее не будет 
    генирироваться толщина эта функция будет вызываться внутри фунции создающую  модель"""
    system = []

    if fine_tune_system:
        #Генерация параметров для линзы
        k = random.uniform(0.4, 0.6)           # k - соотношение смешивания материалов: k = 0.2 => 0.2 материала #1 и 0.8 материала #2
        t = random.uniform(0.8, 1.0)
        sd_1 = random.uniform(1.5, 1.7)        # sd - semi-diameter профиля
        r_1 = random.uniform(1.4, 3.4)       

        # Генерация параметров для слоя воздуха
        sd_2 = random.uniform(1.1, 1.2)        # - 4-ый параметр в списке system!
        r_2 = random.uniform(9.0, 11.0)
        t_prev_air = random.uniform(0.8, 1.0)

    else:
        k = random.uniform(0.001, 1.0)  
        t = random.uniform(0.001, 1.0)
        sd_1 = random.uniform(1.1, 1.9)            
        sd_2 = random.uniform(1.1, 1.9) 
        t_prev_air = random.uniform(0.001, 1.0)
        
        k_2 = random.uniform(0.001, 1.0)  
        t_1 = random.uniform(0.001, 1.0)
        sd_3 = random.uniform(1.6, 2.9)            
        sd_4 = random.uniform(1.6, 2.9) 
        t_prev_air_1 = random.uniform(0.001, 1.0)
        
        # Генерация радиуса кривизны
        # для линзы и слоя воздуха они 
        # должны быть одного знака и каждый 
        # не может быть меньше 0.6 по модулю
        
        if random.uniform(0.0, 1.0) >= 0.5 or is_first_lens:
            r_1 = random.uniform(1.01, 10.0)       # r - радиус кривизны (замещает величину кривизны "curvature")
            r_2 = random.uniform(1.0, 10.0)
            r_3 = random.uniform(-10.0, -1.01)
            r_4 = random.uniform(-10.0, -1.01)
    
    if is_first_lens:
        system.extend([
            k, t, sd_1, r_1, sd_2, r_2, t_prev_air_1,
            k_2, t_1, sd_3, r_3, sd_4, r_4
            ])
    else:
        system.extend([k, t, sd_1, r_1, sd_2, r_2, t_prev_air]) # если линза не первая, генерируем ширину слоя воздуха после предыдущей линзы до текущей

    return system




def evaluate_system(system):
    # Здесь вызывайте вашу функцию calc_loss(), передавая параметры системы
    # и получая метрики качества изображения, например, Encircled Energy и Spot RMS.
    # Затем объедините метрики в одну целевую функцию для оптимизации.
    # Например:
    #encircled_energy = calc_encircled_energy(system)
    #spot_rms = calc_spot_rms(system)

    # Здесь можно взвешать метрики, если необходимо
    #weighted_score = 0.7 * encircled_energy + 0.3 * spot_rms

    opm = OpticalModel() # create new model
    sm = opm['seq_model']
    osp = opm['optical_spec']
    opm.system_spec.title = 'Test Model'
    opm.system_spec.dimensions = 'mm'

    osp['pupil'] = PupilSpec(osp, value=2.5)
    osp['fov'] = FieldSpec(osp, key=['object', 'angle'], is_relative=False, flds=[0., 5., 10., 15., 20.])
    osp['wvls'] = WvlSpec([(470, 1.0), (650, 1.0)], ref_wl=1)

    sm.do_apertures = False
    sm.gaps[0].thi=1e10

    # 1 surface - lens    
    k_1 = system[0]
    t_1 = system[1]                    # 0.0001 < t < 1.0

    medium_1_1 = 1.54 * k_1 + 1.67 * (1 - k_1)
    medium_2_1 = 75.0 * k_1 + 39.0 * (1 - k_1)
    sd_1 = system[2]
    r_1 = system[3]
    curvature_1 = r_1 # according to doc this parameter is ovverwritten by radius 'r'
    coefs_1 = [0.0, 0.009109298409282469, -0.03374649200850791, 0.01797256809388843, -0.0050513483804677005, 0.0, 0.0, 0.0] #[0.0, -0.0231369463217776, 0.011956554928461116, -0.017782670650182023, 0.004077846642272649, 0.0, 0.0, 0.0]

    sm.add_surface([curvature_1, t_1, medium_1_1, medium_2_1], sd=sd_1)
    sm.ifcs[sm.cur_surface].profile = EvenPolynomial(r=r_1, coefs=coefs_1)

    sm.set_stop()
        # 2 surface - air
    sd_2 = system[4]
    r_2 = system[5]
    t_2=system[6]
    curvature_2 = r_2 #-0.2568888474926888
    coefs_2 = [0.0, -0.002874728268075267, -0.03373322938525211, 0.004205227876537139, -0.0001705765222318475, 0.0, 0.0, 0.0] #[0., 0., -1.131e-1, -7.863e-2, 1.094e-1, 6.228e-3, -2.216e-2, -5.89e-3, 4.123e-3, 1.041e-3]

    sm.add_surface([curvature_2, t_2], sd=sd_2)
    sm.ifcs[sm.cur_surface].profile = EvenPolynomial(r=r_2,coefs=coefs_2)
    
        # 3 surface - lens    
    k_3 = system[7]
    t_3 = system[8] # 0.0001 < t < 1.0

    medium_1_3 = 1.54 * k_3 + 1.67 * (1 - k_3)
    medium_2_3 = 75.0 * k_3 + 39.0 * (1 - k_3)
    sd_3 = system[9]
    r_3 = system[10]
    curvature_3 = r_3 # according to doc this parameter is ovverwritten by radius 'r'
    coefs_3 = [0.0, -0.0231369463217776, 0.011956554928461116, -0.017782670650182023, 0.004077846642272649, 0.0, 0.0, 0.0]

    sm.add_surface([curvature_3, t_3, medium_1_3, medium_2_3], sd=sd_3)
    sm.ifcs[sm.cur_surface].profile = EvenPolynomial(r=r_3, coefs=coefs_3)

    # 4 surface - air
    sd_4 = system[11]
    t_4 = 6.8 - (t_1 + t_2 + t_3)
    r_4 = system[12]
    curvature_4 = r_4
    coefs_4 = [0., 0., -1.131e-1, -7.863e-2, 1.094e-1, 6.228e-3, -2.216e-2, -5.89e-3, 4.123e-3, 1.041e-3]
    sm.add_surface([curvature_4, t_4], sd=sd_4)
    sm.ifcs[sm.cur_surface].profile = EvenPolynomial(r=r_4,coefs=coefs_4)


    opm.update_model()
    return [calc_loss_mute(opm)]

test=generate_random_system(0,1)
result = evaluate_system(test)  # Здесь вызываем функцию evaluate_system
print(result)

#Creating obgects
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)


toolbox = base.Toolbox()
toolbox.register("attribute", generate_random_system, fine_tune_system=0, is_first_lens=1)
toolbox.register("individual", tools.initIterate, creator.Individual,
                 toolbox.attribute)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

#Operators
toolbox.register("mate", tools.cxOnePoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3) 
toolbox.register("evaluate", evaluate_system)

def main():
    CXPB, MUTPB, NGEN = 0.5, 0.2, 10
    population = toolbox.population(n=16)
    population, logbook = algorithms.eaSimple(population, toolbox,
                                        cxpb=CXPB,
                                        mutpb=MUTPB,
                                        ngen=NGEN,
                                        verbose=True)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    return population, logbook
population, logbook = main()
print(evaluate_system(population[0]))
