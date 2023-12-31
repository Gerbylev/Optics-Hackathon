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


def generate_random_coefficients(num_coeffs=8, min_value=-0.00001, max_value=0.00001):
    """Генерирует случайные коэффициенты для RadialPolynomial.

    Args:
        num_coeffs (int): Количество коэффициентов для генерации.
        min_value (float): Минимальное значение коэффициента.
        max_value (float): Максимальное значение коэффициента.

    Returns:
        list: Список случайных коэффициентов.
    """
    coefficients_1 = [random.uniform(min_value, max_value) for _ in range(num_coeffs)]
    coefficients=[0,0]
    coefficients.extend(coefficients_1)
    return coefficients



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
    t_1=random.uniform(0.6,1.6)
    k_1=random.uniform(0.001, 1.0)
    
    air_t_1=random.uniform(0.7,1.2)
    
    t_2=random.uniform(0.5,1)
    k_2=random.uniform(0.001, 1.0)
    
    air_t_2=random.uniform(0.35,1)
    
    t_3=random.uniform(0.35,1)
    k_3=random.uniform(0.001, 1.0)
    
    air_t_3=random.uniform(0.4,0.8)
    
    t_4=random.uniform(0.4, 0.6)
    k_4=random.uniform(0.001, 1.0)
    
    air_t_4=random.uniform(0.35,0.6)
    
    t_5=random.uniform(0.35,0.6)
    k_5=random.uniform(0.001, 1.0)
    
    air_t_5=random.uniform(0.4,0.8)
    
    
    
    r_1 = random.uniform(1.001, 10.0)
    air_r_1 = random.uniform(1.001, 10.0)
    
    r_2 = random.uniform(-6.0, -1.01)
    air_r_2 = random.uniform(-6.0, -1.01)
    
    random_list = [random.choice([-1, 1]) for _ in range(5)]
    r_3 = random.uniform(1.001, 10.0)*random_list[0]
    air_r_3 = random.uniform(1.001, 10.0)*random_list[1]
    
    r_4 = random.uniform(1.001, 10.0)*random_list[2]
    air_r_4 = random.uniform(1.001, 10.0)*random_list[3]
    
    r_5 = random.uniform(1.001, 10.0)*random_list[4]
    
    sd_1=random.uniform(0.601, 1)
    air_sd_1=random.uniform(1.001, 1)
    
    sd_2=random.uniform(0.6, 1)
    air_sd_2=random.uniform(1.001, 1.0)
    
    sd_3=random.uniform(0.6001, 1.0)
    air_sd_3=random.uniform(1.001, 1.0)
    
    sd_4=random.uniform(0.6001, 1.0)
    air_sd_4=random.uniform(1.001, 1.0)
    
    sd_5=random.uniform(0.6001, 1.0)

    
    system.extend([
        t_1,k_1,air_t_1,
                   t_2,k_2,air_t_2,
                   t_3,k_3,air_t_3,
                   t_4,k_4,air_t_4,
                   t_5,k_5,air_t_5,
                   r_1, air_r_1, r_2, air_r_2 ,r_3 ,air_r_3 ,r_4 ,air_r_4 , r_5,
                   ])
    system.extend(generate_random_coefficients())
    system.extend(generate_random_coefficients())
    system.extend(generate_random_coefficients())
    system.extend(generate_random_coefficients())
    system.extend(generate_random_coefficients())
    system.extend(generate_random_coefficients())
    system.extend(generate_random_coefficients())
    system.extend(generate_random_coefficients())
    system.extend(generate_random_coefficients())
    system.extend([sd_1, air_sd_1, sd_2, air_sd_2 ,sd_3 ,air_sd_3 ,sd_4 ,air_sd_4 , sd_5])
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
        
        sd_1=system[114]
        r_1=system[15]
        t_1 = system[0]
        k_1 = system[1]

        medium_1_1 = 1.54 * k_1 + 1.67 * (1 - k_1)
        medium_2_1 = 75.0 * k_1 + 39.0 * (1 - k_1)
        
        coefs_1=system[24:34]

        sm.add_surface([r_1,t_1 , medium_1_1, medium_2_1],sd=sd_1)
        sm.ifcs[sm.cur_surface].profile = EvenPolynomial(r=r_1,
                                coefs=coefs_1)
        
        air_sd_1=system[115]
        air_r_1=system[16]
        air_t_1=system[2]
        air_coefs_1=system[34:44]

        sm.add_surface([air_r_1, air_t_1],sd=air_sd_1)
        sm.ifcs[sm.cur_surface].profile = EvenPolynomial(r=air_r_1,
                                coefs=air_coefs_1)
        
        sd_2=system[116]
        r_2=system[17]
        t_2 = system[3]
        k_2 = system[4]

        medium_1_2 = 1.54 * k_2 + 1.67 * (1 - k_2)
        medium_2_2 = 75.0 * k_2 + 39.0 * (1 - k_2)
        
        coefs_2=system[44:54]
        
        sm.add_surface([r_2, t_2, medium_1_2, medium_2_2], sd=sd_2)
        sm.ifcs[sm.cur_surface].profile = EvenPolynomial(r=r_2,
                                coefs=coefs_2)
        
        air_sd_2=system[117]
        air_r_2=system[18]
        air_t_2=system[5]
        air_coefs_2=system[54:64]

        sm.add_surface([air_r_2, air_t_2], sd=air_sd_2)
        sm.ifcs[sm.cur_surface].profile = EvenPolynomial(r=air_r_2,
                                coefs=air_coefs_2)
        
        sd_3=system[118]
        r_3=system[19]
        t_3 = system[6]
        k_3 = system[7]

        medium_1_3 = 1.54 * k_3 + 1.67 * (1 - k_3)
        medium_2_3 = 75.0 * k_3 + 39.0 * (1 - k_3)
        
        coefs_3=system[64:74]
        
        sm.add_surface([r_3, t_3, medium_1_3, medium_2_3],sd=sd_3)
        sm.ifcs[sm.cur_surface].profile = EvenPolynomial(r=r_3,
                                coefs=coefs_3)
        
        air_sd_3=system[119]
        air_r_3=system[20]
        air_t_3=system[8]
        air_coefs_3=system[74:84]

        sm.add_surface([air_r_3, air_t_3], sd=air_sd_3)
        sm.ifcs[sm.cur_surface].profile = EvenPolynomial(r=air_r_3,
                                coefs=air_coefs_3)
        
        sd_4=system[120]
        r_4=system[21]
        t_4 = system[9]
        k_4 = system[10]

        medium_1_4 = 1.54 * k_4 + 1.67 * (1 - k_4)
        medium_2_4 = 75.0 * k_4 + 39.0 * (1 - k_4)
        
        coefs_4=system[84:94]
        
        sm.add_surface([r_4, t_4, medium_1_4, medium_2_4],sd=sd_4)
        sm.ifcs[sm.cur_surface].profile = EvenPolynomial(r=r_4,
                                coefs=coefs_4)
        
        air_sd_4=system[121]
        air_r_4=system[22]
        air_t_4=system[11]
        air_coefs_4=system[94:104]

        sm.add_surface([air_r_4, air_t_4], sd=air_sd_4)
        sm.ifcs[sm.cur_surface].profile = EvenPolynomial(r=air_r_4,
                                coefs=air_coefs_4)
        
        sd_5=system[122]
        r_5=system[23]
        t_5 = system[12]
        k_5 = system[13]

        medium_1_5 = 1.54 * k_5 + 1.67 * (1 - k_5)
        medium_2_5 = 75.0 * k_5 + 39.0 * (1 - k_5)
        
        coefs_5=system[104:114]
        
        sm.add_surface([r_5, t_5, medium_1_5, medium_2_5],sd=sd_5)
        sm.ifcs[sm.cur_surface].profile = EvenPolynomial(r=r_5,
                                coefs=coefs_5)

        air_t_5=system[14]

        sm.add_surface([0.,air_t_5])
        
        opm.update_model()
        return [calc_loss_mute(opm)]
    except: return[10000000]

test=generate_random_system()
print(test)
result = evaluate_system(test)  # Здесь вызываем функцию evaluate_system
print(result)


def custom_mutate(individual, mu, sigma, indpb):
    for i in range(len(individual)):
        # Первый и третий параметры оставляем без мутации
        if i not in  [0, 1, 2, 3, 4, 5, 6,7,8,9,10,11,12,13,14]:
            individual[i] += random.gauss(mu, sigma) if random.random() < indpb else 0.0
    return individual,


def custom_initRepeat(container, func, n):
    """Call the function *func* *n* times and return the results in a
    container type `container`

    :param container: The type to put in the data from func.
    :param func: The function that will be called n times to fill the
                 container.
    :param n: The number of times to repeat func.
    :returns: An instance of the container filled with data from func.

    This helper function can be used in conjunction with a Toolbox
    to register a generator of filled containers, as individuals or
    population.

        >>> import random
        >>> random.seed(42)
        >>> initRepeat(list, random.random, 2) # doctest: +ELLIPSIS,
        ...                                    # doctest: +NORMALIZE_WHITESPACE
        [0.6394..., 0.0250...]

    See the :ref:`list-of-floats` and :ref:`population` tutorials for more examples.
    """
    count=0
    while True:
        ind=func()
        if evaluate_system(ind)[0]<2000: 
            container(ind)
            count+=1
        if count==n:break
    return container(func() for _ in range(n))


#Creating obgects
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)




toolbox = base.Toolbox()
toolbox.register("attribute", generate_random_system)
toolbox.register("individual", tools.initIterate, creator.Individual,
                 toolbox.attribute)
toolbox.register("population", custom_initRepeat, list, toolbox.individual)

#Operators
toolbox.register("mate", tools.cxOnePoint)
toolbox.register("mutate", custom_mutate, mu=0, sigma=1, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3) 
toolbox.register("evaluate", evaluate_system)

def main():
    CXPB, MUTPB, NGEN = 0.5, 0.4, 24
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
best_individ=population[res_1.index(min(res_1))]
print(evaluate_system(best_individ))

def save_system(system, do_draw=True, path='result.roa'):

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
    
    sd_1=system[114]
    r_1=system[15]
    t_1 = system[0]
    k_1 = system[1]

    medium_1_1 = 1.54 * k_1 + 1.67 * (1 - k_1)
    medium_2_1 = 75.0 * k_1 + 39.0 * (1 - k_1)
    
    coefs_1=system[24:34]

    sm.add_surface([r_1,t_1 , medium_1_1, medium_2_1],sd=sd_1)
    sm.ifcs[sm.cur_surface].profile = EvenPolynomial(r=r_1,
                            coefs=coefs_1)
    
    air_sd_1=system[115]
    air_r_1=system[16]
    air_t_1=system[2]
    air_coefs_1=system[34:44]

    sm.add_surface([air_r_1, air_t_1],sd=air_sd_1)
    sm.ifcs[sm.cur_surface].profile = EvenPolynomial(r=air_r_1,
                            coefs=air_coefs_1)
    
    sd_2=system[116]
    r_2=system[17]
    t_2 = system[3]
    k_2 = system[4]

    medium_1_2 = 1.54 * k_2 + 1.67 * (1 - k_2)
    medium_2_2 = 75.0 * k_2 + 39.0 * (1 - k_2)
    
    coefs_2=system[44:54]
    
    sm.add_surface([r_2, t_2, medium_1_2, medium_2_2], sd=sd_2)
    sm.ifcs[sm.cur_surface].profile = EvenPolynomial(r=r_2,
                            coefs=coefs_2)
    
    air_sd_2=system[117]
    air_r_2=system[18]
    air_t_2=system[5]
    air_coefs_2=system[54:64]

    sm.add_surface([air_r_2, air_t_2], sd=air_sd_2)
    sm.ifcs[sm.cur_surface].profile = EvenPolynomial(r=air_r_2,
                            coefs=air_coefs_2)
    
    sd_3=system[118]
    r_3=system[19]
    t_3 = system[6]
    k_3 = system[7]

    medium_1_3 = 1.54 * k_3 + 1.67 * (1 - k_3)
    medium_2_3 = 75.0 * k_3 + 39.0 * (1 - k_3)
    
    coefs_3=system[64:74]
    
    sm.add_surface([r_3, t_3, medium_1_3, medium_2_3],sd=sd_3)
    sm.ifcs[sm.cur_surface].profile = EvenPolynomial(r=r_3,
                            coefs=coefs_3)
    
    air_sd_3=system[119]
    air_r_3=system[20]
    air_t_3=system[8]
    air_coefs_3=system[74:84]

    sm.add_surface([air_r_3, air_t_3], sd=air_sd_3)
    sm.ifcs[sm.cur_surface].profile = EvenPolynomial(r=air_r_3,
                            coefs=air_coefs_3)
    
    sd_4=system[120]
    r_4=system[21]
    t_4 = system[9]
    k_4 = system[10]

    medium_1_4 = 1.54 * k_4 + 1.67 * (1 - k_4)
    medium_2_4 = 75.0 * k_4 + 39.0 * (1 - k_4)
    
    coefs_4=system[84:94]
    
    sm.add_surface([r_4, t_4, medium_1_4, medium_2_4],sd=sd_4)
    sm.ifcs[sm.cur_surface].profile = EvenPolynomial(r=r_4,
                            coefs=coefs_4)
    
    air_sd_4=system[121]
    air_r_4=system[22]
    air_t_4=system[11]
    air_coefs_4=system[94:104]

    sm.add_surface([air_r_4, air_t_4], sd=air_sd_4)
    sm.ifcs[sm.cur_surface].profile = EvenPolynomial(r=air_r_4,
                            coefs=air_coefs_4)
    
    sd_5=system[122]
    r_5=system[23]
    t_5 = system[12]
    k_5 = system[13]

    medium_1_5 = 1.54 * k_5 + 1.67 * (1 - k_5)
    medium_2_5 = 75.0 * k_5 + 39.0 * (1 - k_5)
    
    coefs_5=system[104:114]
    
    sm.add_surface([r_5, t_5, medium_1_5, medium_2_5],sd=sd_5)
    sm.ifcs[sm.cur_surface].profile = EvenPolynomial(r=r_5,
                            coefs=coefs_5)

    air_t_5=system[14]

    sm.add_surface([0.,air_t_5])
    
    opm.update_model()
    if do_draw:
        isdark = False
        # 1 plot
        layout_plt0 = plt.figure(FigureClass=InteractiveLayout, opt_model=opm,
                                do_draw_rays=True, do_paraxial_layout=False,
                                is_dark=isdark).plot()
        # 2 plot
        spot_plt = plt.figure(FigureClass=SpotDiagramFigure, opt_model=opm,
                      scale_type=Fit.All_Same, dpi=200, is_dark=isdark).plot()

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
save_system(best_individ)




def find_intersection(par=[], sys=[], interval=[0.2, 2.5], tolerance=0.001, ):
     try:
        def convertor(coefs, n=0):
            reversed_list = coefs[::-1]  # Реверсируем входной список
            result_list = [item for sublist in [[x, 0] for x in reversed_list] for item in sublist]
            result_list.extend([n])
            return result_list

        poly1_1 = convertor(par[0:10])
        poly2_1 = convertor(par[10:20], n=-sys[0])
        poly1_2 = convertor(par[20:30])
        poly2_2 = convertor(par[30:40], n=-sys[3])
        x_values = np.linspace(interval[0], interval[1], 50000)
        intersections = []
        intersections_1 = []

        for x in x_values:
            y1 = np.polyval(poly1_1, x)
            y2 = np.polyval(poly2_1, x)
            y3 = np.polyval(poly1_2, x)
            y4 = np.polyval(poly2_2, x)

            if abs(y1 - y2) < tolerance:
                intersections.append(x)

            if abs(y3 - y4) < tolerance:
                intersections_1.append(x)
        return [random.uniform(0.3 * max(intersections), max(intersections) * 0.8),
                random.uniform(0.3 * max(intersections_1), max(intersections_1) * 0.8)]
    except:
        return 1000


