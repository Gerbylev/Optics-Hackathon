isdark = False
from rayoptics.environment import *
from matplotlib.colors import LogNorm, PowerNorm, Normalize
from rayoptics.util.misc_math import normalize
import re
import io
from contextlib import redirect_stdout


# base_param
def calc_loss(path2model):
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

    opm = path2model

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
    plt.plot(test_psf[:,0],test_psf[:,1],'o')
    plt.rcParams['figure.figsize'] = (8, 8)
    plt.show()


    fld, wvl, foc = osp.lookup_fld_wvl_focus(0)
    sm.list_model()
    sm.list_surfaces()
    efl=pm.opt_model['analysis_results']['parax_data'].fod.efl

    pm.first_order_data()
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
    print(thickness_list)
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

            print(f'{idx_field=}, {idx_wavelength=}, {enclosed_energy=},  {loss_enclosed_energy=},  {loss_rms=}')
            temp=temp+1
    loss_enclosed_energy_all=loss_enclosed_energy_all/temp
    loss_rms_all=loss_rms_all/temp
    loss=loss_focus+loss_FD+loss_total_length+loss_min_thickness+loss_min_thickness_air+loss_enclosed_energy_all+loss_rms_all
    print(f'{loss_focus=}, {loss_FD=},  {loss_total_length=},  {loss_min_thickness=},  {loss_min_thickness_air=},  {loss_enclosed_energy_all=},  {loss_rms_all=}')
    layout_plt0 = plt.figure(FigureClass=InteractiveLayout, opt_model=opm,
                            do_draw_rays=True, do_paraxial_layout=False,
                            is_dark=isdark).plot()
    print(f'final loss:{loss}')
    return(loss)




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
        return [calc_loss(opm)]
    except: return[10000000]


result = evaluate_system([1.498187902042782, 0.15348811971351867, 1.0104282842954864, 0.6742859419362837, 0.05929292105259396, 0.3983491028584521, 0.35409304803728947, 0.9246055138758487, 0.4072960366533883, 0.5436684855035948, 0.5516336630116315, 0.5321807701654718, 0.3971638799375563, 0.8972095459912056, 0.774813602456447, 3.3406164251155688, 2.135322815187548, -4.629035986664633, -2.028775460574962, 8.963661572051038, -9.702631871437124, -3.801523412860063, -2.8976092301823666, -5.7202745733202125, 0, 0, 2.6992564632995634e-05, 3.7011019918846075e-06, 7.824698496947753e-06, 9.935177210981969e-05, -4.079040225532404e-05, -9.601296770178418e-05, 3.981645750933886e-05, 8.65183043445683e-05, 0, 0, -8.805246900943456e-05, 1.3925366530917826e-05, 4.691619995797072e-05, -4.216447735503554e-06, 4.1527018709000777e-05, 7.996465120503875e-06, -5.07014588150758e-05, 6.367837458210199e-05, 0, 0, 7.921787369747812e-05, 3.589170464242626e-05, -1.2615644226255296e-05, 1.2844274848410227e-05, -3.8447234282676734e-05, 5.538320843008323e-05, 6.678736343943978e-05, -6.474460692926953e-05, 0, 0, 1.4102438295662373e-05, -8.130038971077381e-05, 3.5704182800234187e-06, -9.719054362376174e-05, -8.406370984077131e-06, 1.339312106658113e-06, 7.193267170903793e-05, -7.995954921091372e-05, 0, 0, -1.7699192339949924e-05, 7.809022017664947e-05, -5.2548082779729956e-06, -2.384853588159377e-05, 5.683701426247667e-05, -1.689665150117779e-05, 6.077398892047034e-05, 7.122522992400228e-05, 0, 0, -3.4753870011782585e-05, -4.477487513818668e-05, -3.1551127500856806e-05, -7.146969803828862e-05, 5.5260666633393936e-05, -4.315111987233897e-05, -4.200125522042466e-05, -8.316555547496567e-05, 0, 0, -6.600561756828431e-05, 8.430585176543734e-05, -4.227677043504572e-05, 6.799100031911757e-05, 4.654833964248528e-05, 2.160758809613567e-05, -1.4718947884214821e-05, -7.337283750666476e-05, 0, 0, 2.6328239688951557e-05, -3.605292642070972e-05, 2.8633852610418826e-05, -6.035838925051427e-05, 4.6225401194922063e-07, 4.020821971009523e-05, -6.434199337672422e-05, 7.767752161406707e-05, 0, 0, -8.228539014230465e-05, 4.356420510709005e-05, 6.8203063471995e-05, -1.0907619708189654e-07, 7.835364133000783e-05, 5.9607500567005435e-05, 4.017981113712159e-05, -3.227519496895049e-05, 1.8522428923812224, 1.0887327277175107, 0.7316424002507023, 1.1812208372037696, 2.458456044379818, 2.0458382042663654, 0.6222681708515645, 2.783616380339546, 2.5235068230572058])
print(result)