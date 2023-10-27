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
    sm.set_stop()
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


result = save_system([0.7221800184530666, 0.9040627403707129, 0.8591315335530243, 0.6068763506815826, 0.5795772764687727, 0.3771414711720652, 0.5362301374384904, 0.5409249426051936, 0.451598483376773, 0.5630586944686732, 0.816099204790481, 0.5494603590015201, 0.5267508203107663, 0.47800946016465445, 0.6567353307797077, 3.6055444510670616, 5.725117085212197, -2.904937429079047, -3.1765538366235226, 1.9751301825415357, -7.870441813235621, -7.547359102931939, 3.537254457719592, -9.634209020975343, 0.0, 0.0, -4.379361430888358e-06, -4.057777849010209e-05, -8.723004974192444e-05, -4.711178314954101e-05, 2.417593644055099e-05, -4.543552521849459e-05, 1.188557625057666, 1.216013613237262, 0.0, 0.0, 3.4907624291758124e-05, -3.1733409685732105e-05, -8.50317381946011e-05, -9.887938598090453e-06, 7.058108048271521e-05, 6.808421381643516e-05, -1.1209283105584095e-05, -3.2360557818560584e-05, 0.0, 0.0, -8.894789032464681e-05, -5.893815349086709e-05, 6.602308802136643e-05, -0.44545445576231435, -9.90628201444528e-05, -8.425305595666071e-05, 4.286257919356661e-05, -7.581959568878161e-05, 0.0, 0.0, 1.3779910494395527e-05, 7.603037451956605e-05, -4.424321250101011e-05, 6.119582108499642e-05, 7.720422095961337e-05, 1.919636182471713e-06, 6.081357943010203e-05, 6.176743477385976e-05, 0, 0, 5.148241466405405e-05, 8.036424638748061e-05, -2.263505256597729e-05, -9.80913478173016e-05, -8.464499987250078e-05, -5.780843458466329e-05, -6.973602019809653e-05, 3.404488256241645e-05, 0, 0, 9.170602800090694e-05, -1.0919826503662393e-05, 5.5121644831239514e-05, -8.474596999579465e-06, 9.254848140953247e-05, 1.4234213305669717e-05, 9.556607295622885e-05, 7.746421413941437e-05, 0, 0, 5.289407392339952e-07, -5.085881567647821e-05, 4.3495780288928566e-05, 9.764235502359745e-05, -8.535696756495817e-05, 9.603532349713533e-05, 8.539200236681991e-05, 2.0807664072769707e-05, 0, 0, -4.609554329409411e-05, -5.870518977335422e-05, -4.207369862125192e-05, -4.343254002743193e-05, 5.371487531119811e-05, 8.559732660934288e-05, -1.2289594810583956e-05, 6.143735006158836e-05, 0, 0, 1.9440511817282014e-05, 5.087581985684882e-05, 6.614907328590163e-05, -5.664346935570928e-05, -5.525465796843925e-05, -6.751370513862563e-05, 6.752442107101562e-05, 9.053162133829646e-06, 0.6777283266995701, 2.4040810478388517, 0.5749639126642938, 1.5968855857326947, 2.3936120955550635, 2.8470789755986283, 1.0564874864778142, 1.7861613413665025, 1.6454547703341218])
print(result)