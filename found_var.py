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
import warnings

warnings.filterwarnings('ignore')

def generate_random_coefficients(coefficients, variation_percentage=50):
    """
    Варьирует существующие коэффициенты случайным образом в пределах заданного процента.

    Args:
        coefficients (list): Список существующих коэффициентов.
        variation_percentage (int): Процент вариации (максимальная амплитуда изменений).

    Returns:
        list: Список новых коэффициентов с учетом вариации.
    """
    new_coefficients = [
        random.uniform(coeff - (coeff * variation_percentage / 100), coeff + (coeff * variation_percentage / 100)) for coeff in coefficients
        ]




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

def generate_random_system(step=1,limit=7):
    """1) реализовать определение границ внутри этой системы 
    2) добавить аргумент,который определяет что линза последняя, для нее не будет 
    генирироваться толщина эта функция будет вызываться внутри фунции создающую  модель"""
    if step == 1:
        syst=[]
        r_1_1=2.2514274752334913 
        r_2_1=12.883118354209113
        r_3_1=-14.5102827135106
        t_1=random.uniform(0.1, 0.2*limit)
        k_1=random.uniform(0.001, 1.0)
        
        
        air_t_1=random.uniform(0.1,0.3*limit)
        
        t_2=random.uniform(0.1,0.2*limit)
        k_2=random.uniform(0.001, 1.0)
        
        air_t_2=random.uniform(0.1,0.3*limit)
    
        r_1 = random.uniform(1.001, 10.0)
        air_r_1 = random.uniform(1.001, 20.0)
        
        r_2 = random.uniform(-20.0, -1.01)
        
        sd_1=random.uniform(0.75,2)
        air_sd_1=random.uniform(0.25, 1.2)
        
        sd_2=random.uniform(0.1, 1)
    
        syst.extend([
            t_1,k_1,air_t_1,
            t_2,k_2,air_t_2,
            r_1, air_r_1, r_2,sd_1, air_sd_1, sd_2,
                    ])
    else:
        syst=[]
        syst.extend(generate_random_coefficients([0., 0., -1.895e-2, 2.426e-2, -5.123e-2, 8.371e-4, 7.850e-3, 4.091e-3, -7.732e-3, -4.265e-3]))
        syst.extend(generate_random_coefficients([0., 0., -4.966e-3, -1.434e-2, -6.139e-3, -9.284e-5, 6.438e-3, -5.72e-3, -2.385e-2, 1.108e-2]))
        syst.extend(generate_random_coefficients([0., 0., -4.388e-2, -2.555e-2, 5.16e-2, -4.307e-2, -2.831e-2, 3.162e-2, 4.630e-2, -4.877e-2]))

    return syst



def evaluate_system(system, step, paramer=[], save_system=False):
    # Здесь вызывайте вашу функцию calc_loss(), передавая параметры системы
    # и получая метрики качества изображения, например, Encircled Energy и Spot RMS.
    # Затем объедините метрики в одну целевую функцию для оптимизации.
    # Например:
    #encircled_energy = calc_encircled_energy(system)
    #spot_rms = calc_spot_rms(system)

    # Здесь можно взвешать метрики, если необходимо
    #weighted_score = 0.7 * encircled_energy + 0.3 * spot_rms
    try:
        if step==1:
            opm = OpticalModel() # create new model

            sm = opm['seq_model']
            osp = opm['optical_spec']
            pm = opm['parax_model']
            em = opm['ele_model']
            pt = opm['part_tree']
            opm.system_spec.title = 'Test Model'
            opm.system_spec.dimensions = 'mm'

            osp['pupil'] = PupilSpec(osp, value=2.5)
            osp['fov'] = FieldSpec(osp, key=['object', 'angle'], is_relative=False, flds=[0., 5., 10., 15., 20.])
            osp['wvls'] = WvlSpec([(470, 1.0), (650, 1.0)], ref_wl=1)
            sm.do_apertures = False
            sm.gaps[0].thi=1e10

            sm.add_surface([0., 0.])
            sm.set_stop()
            
            sd_1=system[9]
            r_1=system[6]
            t_1 = system[0]
            k_1 = system[1]

            medium_1_1 = 1.54 * k_1 + 1.67 * (1 - k_1)
            medium_2_1 = 75.0 * k_1 + 39.0 * (1 - k_1)
            

            sm.add_surface([r_1,t_1 , medium_1_1, medium_2_1],sd=sd_1)
            sm.ifcs[sm.cur_surface].profile = EvenPolynomial(r=r_1,)
            
            air_sd_1=system[10]
            air_r_1=system[7]
            air_t_1=system[2]

            sm.add_surface([air_r_1, air_t_1],sd=air_sd_1)
            sm.ifcs[sm.cur_surface].profile = EvenPolynomial(r=air_r_1,)
            
            sd_2=system[11]
            r_2=system[8]
            t_2 = system[3]
            k_2 = system[4]

            medium_1_2 = 1.54 * k_2 + 1.67 * (1 - k_2)
            medium_2_2 = 75.0 * k_2 + 39.0 * (1 - k_2)
            
            
            sm.add_surface([r_2, t_2, medium_1_2, medium_2_2], sd=sd_2)
            sm.ifcs[sm.cur_surface].profile = EvenPolynomial(r=r_2,)

            air_t_5=system[5]

            sm.add_surface([0.,air_t_5])
            
            opm.update_model()
            return [calc_loss_mute(opm)]
        
        else:
            
            opm = OpticalModel() # create new model

            sm = opm['seq_model']
            osp = opm['optical_spec']
            pm = opm['parax_model']
            em = opm['ele_model']
            pt = opm['part_tree']
            opm.system_spec.title = 'Test Model'
            opm.system_spec.dimensions = 'mm'

            osp['pupil'] = PupilSpec(osp, value=2.5)
            osp['fov'] = FieldSpec(osp, key=['object', 'angle'], is_relative=False, flds=[0., 5., 10., 15., 20.])
            osp['wvls'] = WvlSpec([(470, 1.0), (650, 1.0)], ref_wl=1)
            sm.do_apertures = False
            sm.gaps[0].thi=1e10

            sm.add_surface([0., 0.])
            sm.set_stop()
            
            sd_1=paramer[9]
            r_1=paramer[6]
            t_1 = paramer[0]
            k_1 = paramer[1]

            medium_1_1 = 1.54 * k_1 + 1.67 * (1 - k_1)
            medium_2_1 = 75.0 * k_1 + 39.0 * (1 - k_1)
            

            sm.add_surface([r_1,t_1 , medium_1_1, medium_2_1],sd=sd_1)
            sm.ifcs[sm.cur_surface].profile = EvenPolynomial(r=r_1, coefs=system[0:10])
            
            air_sd_1=paramer[10]
            air_r_1=paramer[7]
            air_t_1=paramer[2]

            sm.add_surface([air_r_1, air_t_1],sd=air_sd_1)
            sm.ifcs[sm.cur_surface].profile = EvenPolynomial(r=air_r_1,coefs=system[10:20])
            
            sd_2=paramer[11]
            r_2=paramer[8]
            t_2 = paramer[3]
            k_2 = paramer[4]

            medium_1_2 = 1.54 * k_2 + 1.67 * (1 - k_2)
            medium_2_2 = 75.0 * k_2 + 39.0 * (1 - k_2)
            
            
            sm.add_surface([r_2, t_2, medium_1_2, medium_2_2], sd=sd_2)
            sm.ifcs[sm.cur_surface].profile = EvenPolynomial(r=r_2,coefs=system[20:30])

            air_t_5=paramer[5]

            sm.add_surface([0.,air_t_5])
            
            opm.update_model()
            if save_system: opm.save_model(f'loss_{calc_loss_mute(opm)}.roa')
            return [calc_loss_mute(opm)]
    except: return[10000000]
    

def main(func_step=1, params=[], variation_percentage=50):    
    def custom_mutate(individual, mu, sigma, indpb, step=func_step):
        if step==1:
            for i in range(len(individual)):
                individual[i] += random.gauss(mu, sigma) if random.random() < indpb else 0.0
        else: 
            for i in range(30):
                individual[i] += random.uniform(individual[i] - (individual[i] * variation_percentage / 100), individual[i] + (individual[i] * variation_percentage / 100))
        return individual,


    def custom_initRepeat(container, func, n, step=func_step):
        if step==1: porog = 1500
        else: porog = 600
        count=0
        while True:
            ind=func()
            if evaluate_system(ind, step=func_step, paramer=params)[0]<porog: 
                container(ind)
                count+=1
                if count%15==0 and step==2: evaluate_system(ind, step=func_step, paramer=params, save_system=True)
                print(count)
            if count==n:break
        return container(func() for _ in range(n))


    #Creating obgects
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)




    toolbox = base.Toolbox()
    toolbox.register("attribute", generate_random_system, step=func_step,)
    toolbox.register("individual", tools.initIterate, creator.Individual,
                    toolbox.attribute)
    toolbox.register("population", custom_initRepeat, list, toolbox.individual, step=func_step)

    #Operators
    toolbox.register("mate", tools.cxOnePoint)
    toolbox.register("mutate", custom_mutate, mu=0, sigma=1, indpb=0.1, step = func_step)
    toolbox.register("select", tools.selTournament, tournsize=3) 
    toolbox.register("evaluate", function=evaluate_system, step=func_step, paramer=params)

    CXPB, MUTPB, NGEN = 0.5, 0.4, 60
    population = toolbox.population(n=64)
    population, logbook = algorithms.eaSimple(population, toolbox,
                                        cxpb=CXPB,
                                        mutpb=MUTPB,
                                        ngen=NGEN,
                                        verbose=True)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    return population, logbook
population, logbook = main()
from functools import partial

res = list(map(partial(evaluate_system, step=1), population))

res_1=[i[0] for i in res]
best_individ=population[res_1.index(min(res_1))]
print(evaluate_system(best_individ, step=1))

population_2, _ = main(params=best_individ, func_step=2)
res_2=list(map(partial(evaluate_system, step=2, paramer=best_individ),population_2))
res_3=[i[0] for i in res_2]
best_individ_ever=population_2[res_3.index(min(res_3))]
print(evaluate_system(best_individ_ever, step=2, paramer=best_individ, save_system=True))