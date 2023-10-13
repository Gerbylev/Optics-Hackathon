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

    
    return new_coefficients



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

def generate_random_system():
    """1) реализовать определение границ внутри этой системы 
    2) добавить аргумент,который определяет что линза последняя, для нее не будет 
    генирироваться толщина эта функция будет вызываться внутри фунции создающую  модель"""
    system = []
    t_1=random.uniform(0.6,1.8)
    k_1=random.uniform(0.001, 1.0)
    
    air_t_1=random.uniform(0.7,1.3)
    
    t_2=random.uniform(1,0.5)
    k_2=random.uniform(0.001, 1.0)
    
    air_t_3=random.uniform(0.7,1.3)
    
    t_3=random.uniform(0.35,1.2)
    k_3=random.uniform(0.001, 1.0)
    
    air_t_2=random.uniform(0.35,1)
    
    system.extend([t_1,k_1,air_t_1,t_2,k_2,air_t_2])
    system.extend(generate_random_coefficients([0., 0., -1.895e-2, 2.426e-2, -5.123e-2, 8.371e-4, 7.850e-3, 4.091e-3, -7.732e-3, -4.265e-3]))
    system.extend(generate_random_coefficients([0., 0., -4.966e-3, -1.434e-2, -6.139e-3, -9.284e-5, 6.438e-3, -5.72e-3, -2.385e-2, 1.108e-2]))
    system.extend(generate_random_coefficients([0., 0., -4.388e-2, -2.555e-2, 5.16e-2, -4.307e-2, -2.831e-2, 3.162e-2, 4.630e-2, -4.877e-2]))
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
    try:
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
        opm.radius_mode = True
        sm.do_apertures = False
        sm.gaps[0].thi=1e10

        sm.add_surface([0., 0.])
        sm.set_stop()
        
        t_1 = system[0]
        k_1 = system[1]

        medium_1_1 = 1.54 * k_1 + 1.67 * (1 - k_1)
        medium_2_1 = 75.0 * k_1 + 39.0 * (1 - k_1)
        
        coefs_1=system[6:16]

        sm.add_surface([1.962,t_1 , medium_1_1, medium_2_1])
        sm.ifcs[sm.cur_surface].profile = RadialPolynomial(r=1.962, ec=2.153,
                                coefs=coefs_1)

        air_t_1=system[2]
        
        coefs_2=system[16:26]

        sm.add_surface([33.398, air_t_1])
        sm.ifcs[sm.cur_surface].profile = RadialPolynomial(r=33.398, ec=40.18,
                                coefs=coefs_2)
        
        t_2 = system[3]
        k_2 = system[4]

        medium_1_2 = 1.54 * k_2 + 1.67 * (1 - k_2)
        medium_2_2 = 75.0 * k_2 + 39.0 * (1 - k_2)
        
        coefs_3=system[26:36]
        
        sm.add_surface([-4.182, t_2, medium_1_2, medium_2_2])
        sm.ifcs[sm.cur_surface].profile = RadialPolynomial(r=-4.182, ec=2.105,
                                coefs=coefs_3)

        air_t_2=system[5]

        sm.add_surface([0.,air_t_2])
        
        opm.update_model()
        return [calc_loss_mute(opm)]
    except: return[10000000]

test=generate_random_system()
result = evaluate_system(test)  # Здесь вызываем функцию evaluate_system
print(result)


def custom_mutate(individual, mu, sigma, indpb):
    for i in range(len(individual)):
        # Первый и третий параметры оставляем без мутации
        if i not in  [0, 1, 2, 3, 4, 5, 6, 36, 37,38]:
            individual[i] += random.gauss(mu, sigma) if random.random() < indpb else 0.0
    return individual,



#Creating obgects
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)




toolbox = base.Toolbox()
toolbox.register("attribute", generate_random_system)
toolbox.register("individual", tools.initIterate, creator.Individual,
                 toolbox.attribute)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

#Operators
toolbox.register("mate", tools.cxOnePoint)
toolbox.register("mutate", custom_mutate, mu=0, sigma=1, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3) 
toolbox.register("evaluate", evaluate_system)

def main():
    CXPB, MUTPB, NGEN = 0.5, 0.2, 40
    population = toolbox.population(n=32)
    population, logbook = algorithms.eaSimple(population, toolbox,
                                        cxpb=CXPB,
                                        mutpb=MUTPB,
                                        ngen=NGEN,
                                        verbose=True)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    return population, logbook
population, logbook = main()
res=list(map(evaluate_system,population))
res_1=[i[0] for i in res]
print(min(res_1))