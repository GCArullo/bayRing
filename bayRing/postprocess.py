import corner, os, numpy as np, matplotlib.pyplot as plt, h5py, seaborn as sns

import bayRing.utils          as utils
import bayRing.waveform_utils as waveform_utils

twopi = 2.*np.pi

def read_results_object_from_previous_inference(parameters):

    if(parameters['Inference']['method'] == 'Minimization'):

        utils.minimisation_compatibility_check(parameters)

        results_object_tmp = np.genfromtxt(os.path.join( parameters['I/O']['outdir'],'Algorithm/Minimization_results.txt'), names=True, deletechars="")
        results_object = {}
        for key in results_object_tmp.dtype.names:
            print(results_object_tmp[key])
            results_object[key] = results_object_tmp[key]

    elif(parameters['Inference']['method'] == 'Nested-sampler'):

        if(parameters['Inference']['sampler'] == 'cpnest'):
            try   :   results_object = np.genfromtxt(os.path.join( parameters['I/O']['outdir'],'Algorithm/posterior.dat'), names=True, deletechars="", delimiter = ",")
            except:   results_object = np.genfromtxt(os.path.join( parameters['I/O']['outdir'],'Algorithm/posterior.dat'), names=True, deletechars="")
        elif(parameters['Inference']['sampler'] == 'raynest'):
            filename        = os.path.join( parameters['I/O']['outdir'],'Algorithm/raynest.h5')
            h5_file         = h5py.File(filename,'r')
            results_object  = h5_file['combined'].get('posterior_samples')

    else: raise ValueError('Method {} not recognised.'.format(parameters['Inference']['method']))

    return results_object

def print_point_estimate(results_object, names, method):

    """

    Print the point estimates of the results of a minimization or a nested sampling algorithm.

    Parameters
    ----------

    results_object : object
        Object containing the results of the minimization or nested sampling algorithm.

    names : list
        List of the names of the parameters.

    method : str
        Method used to obtain the results from which the point estimates will be drawn. Can be either 'Minimization' or 'Nested-sampler'.

    Returns
    -------

    Nothing, but prints the point estimates.

    """

    if(method=='Minimization'):
        longest_name_length = utils.find_longest_name_length(results_object.keys())
        for key in results_object.keys(): print('{} : {:.12f}'.format(key.ljust(longest_name_length), results_object[key]))
    else:
        longest_name_length = utils.find_longest_name_length(names)
        for key in names:
            median      = np.median(results_object[key])
            lower_bound = median-np.percentile(results_object[key], 5)
            upper_bound = np.percentile(results_object[key], 95)-median
            print('{} : {:.12f} + {:.12f} - {:.12f}'.format(key.ljust(longest_name_length), median, upper_bound, lower_bound))

    return

def save_results_minimization(results, outdir):

    """

    Save the results of a minimization algorithm.

    Parameters
    ----------

    results : dict
        Dictionary containing the results of the minimization algorithm.

    outdir : str
        Output directory.

    Returns
    -------

    Nothing, but saves the results of the minimization algorithm in a text file.

    """

    len_params    = len(results.values())
    header_string = ''
    for key in results.keys(): header_string += '{}\t'.format(key)
    
    results_output = np.array([np.array(list(results.values())[i]) for i in range(len_params)]).reshape((1,len_params))
    np.savetxt(os.path.join(outdir,'Algorithm','Minimization_Results.txt'), results_output, header = header_string)
    
    return

def store_and_print_amp_phi(amp_name, phi_name, t0, omega, tau, results_object, longest_name_length, outdir):

    """

    Store and print the amplitude and phase of the inferred mode when defined at t0.

    Parameters
    ----------

    amp_name : str
        Name of the amplitude parameter.
    
    phi_name : str
        Name of the phase parameter.

    t0 : float
        Time at which the amplitude and phase are defined.

    omega : float   
        Frequency of the mode.

    tau : float
        Damping time of the mode.

    results_object : dict
        Dictionary containing the results of the inference algorithm.

    longest_name_length : int
        Length of the longest parameter name.

    outdir : str
        Output directory.

    Returns
    -------

    Nothing, but stores and prints the amplitude and phase of the inferred mode.

    """

    exp_tau_factor   = np.exp(t0/tau)
    sum_omega_factor = t0 * omega

    amp_median =  np.exp(np.median(    results_object[amp_name]    )) *   exp_tau_factor
    amp_lower  =  np.exp(np.percentile(results_object[amp_name],  5)) *   exp_tau_factor
    amp_upper  =  np.exp(np.percentile(results_object[amp_name], 95)) *   exp_tau_factor
    phi_median =        (np.median(    results_object[phi_name]    )  - sum_omega_factor)%(2*np.pi)
    phi_lower  =        (np.percentile(results_object[phi_name],  5)  - sum_omega_factor)%(2*np.pi)
    phi_upper  =        (np.percentile(results_object[phi_name], 95)  - sum_omega_factor)%(2*np.pi)

    amp_lower_err  = amp_median - amp_lower
    amp_upper_err  = amp_upper  - amp_median
    phi_lower_err  = phi_median - phi_lower
    phi_upper_err  = phi_upper  - phi_median

    print('{} : {:.12f} + {:.12f} - {:.12f}'.format(amp_name.split('ln_')[-1].ljust(longest_name_length), amp_median, amp_upper_err, amp_lower_err))
    print('{} : {:.12f} + {:.12f} - {:.12f}'.format(phi_name.ljust(longest_name_length), phi_median, phi_upper_err, phi_lower_err))

    outFile_amp = open(os.path.join(outdir,'Peak_quantities/amps_tpeak.txt'), 'a')
    outFile_amp.write('{}\t{}\t{}\t{}\n'.format(amp_name.ljust(longest_name_length), amp_median, amp_lower, amp_upper))
    outFile_amp.close()
    outFile_phi = open(os.path.join(outdir,'Peak_quantities/phis_tpeak.txt'), 'a')
    outFile_phi.write('{}\t{}\t{}\t{}\n'.format(phi_name.ljust(longest_name_length), phi_median, phi_lower, phi_upper))
    outFile_phi.close()

    return


def post_process_amplitudes(t0, results_object, NR_metadata, qnm_cached, modes, quad_modes, outdir):

    """

    Post-process the amplitudes and phases of the inferred modes.

    Parameters
    ----------

    t0 : float
        Time at which the amplitude and phase are defined.

    results_object : dict
        Dictionary containing the results of the inference algorithm.

    NR_metadata : dict
        Dictionary containing the metadata of the NR simulation.

    qnm_interpolants : dict
        Dictionary containing the interpolants of the QNM frequencies and damping times.

    modes : list
        List of the modes to be inferred.

    quad_modes : list
        List of the quadrupole modes to be inferred.

    outdir : str
        Output directory.

    Returns
    -------

    Nothing, but stores and prints the amplitude and phase of the inferred mode.

    """

    print('\n* Amplitudes and phases at t_peak:\n')

    outFile_amp = open(os.path.join(outdir,'Peak_quantities/amps_tpeak.txt'), 'w')
    outFile_phi = open(os.path.join(outdir,'Peak_quantities/phis_tpeak.txt'), 'w')
    outFile_amp.write('#name\tmedian\tlower\tupper\n')
    outFile_phi.write('#name\tmedian\tlower\tupper\n')
    outFile_amp.close()
    outFile_phi.close()

    Mf = NR_metadata['Mf']
    af = NR_metadata['af']
    
    if 'qf' in NR_metadata.keys(): qf = NR_metadata['qf']
    else                         : qf = None

    if (quad_modes is not None): longest_name_length = len('phi_diff_x-yz_x-yz_x-yz')
    else                       : longest_name_length = len('phi_x-yz')

    for (l_x, m_x, n_x) in modes:

        amp_name = 'ln_A_{}{}{}'.format(l_x, m_x, n_x)
        phi_name =  'phi_{}{}{}'.format(l_x, m_x, n_x)

        omega, tau = qnm_cached[(2,l_x,m_x,n_x)]['f'] * twopi, qnm_cached[(2,l_x,m_x,n_x)]['tau']

        store_and_print_amp_phi(amp_name, phi_name, t0, omega, tau, results_object, longest_name_length, outdir)

    if(quad_modes is not None):
        for quad_term in quad_modes.keys():
            for ((l,m,n),(l1,m1,n1),(l2,m2,n2)) in quad_modes[quad_term]:

                quad_string = '{}_{}{}{}_{}{}{}_{}{}{}'.format(quad_term, l,m,n, l1,m1,n1, l2,m2,n2)
                amp_name = 'ln_A_{}'.format(quad_string)
                phi_name = 'phi_{}'.format(quad_string)

                omega1, tau1 = qnm_cached[(2,l1, m1, n1)]['f'] * twopi, qnm_cached[(2,l1, m1, n1)]['tau']
                omega2, tau2 = qnm_cached[(2,l2, m2, n2)]['f'] * twopi, qnm_cached[(2,l2, m2, n2)]['tau']

                tau   = (tau1 * tau2)/(tau1 + tau2)
                if  (quad_term=='sum' ): omega = omega1 + omega2
                elif(quad_term=='diff'): omega = omega1 - omega2

                store_and_print_amp_phi(amp_name, phi_name, t0, omega, tau, results_object, longest_name_length, outdir)

    return 

def l2norm_residual_vs_nr(results_object, nest_model, NR_sim, outdir):
    
    """

    Compare the residual of the fit with the NR error.

    Find the peak time of the amplitude.

    Parameters
    ----------

    results_object : dict
        Dictionary containing the results of the inference algorithm.
    
    nest_model : Nested sampler object
        Nested sampler object. 

    NR_sim : NR_sim
        NR simulation object.

    outdir : str
        output directory

    Returns
    ---------

    Nothing, but prints and stores in a file the L2 norm of residuals and NR_error.

    """

    NR_err_r, NR_err_i = np.real(NR_sim.NR_cpx_err_cut), np.imag(NR_sim.NR_cpx_err_cut)
    NR_r, NR_i         = np.real(NR_sim.NR_cpx_cut)     , np.imag(NR_sim.NR_cpx_cut)
    t_cut = NR_sim.t_NR_cut

    models_re_list = [np.real(np.array(nest_model.model(p))) for p in results_object]
    models_im_list = [np.imag(np.array(nest_model.model(p))) for p in results_object]

    wf_r = np.percentile(np.array(models_re_list),[50], axis=0)[0]
    wf_i = np.percentile(np.array(models_im_list),[50], axis=0)[0]

    l2_NR       = np.trapz(np.sqrt(      NR_err_r** 2 +       NR_err_i** 2), t_cut)
    l2_residual = np.trapz(np.sqrt((NR_r - wf_r) ** 2 + (NR_i - wf_i) ** 2), t_cut)

    print(f'\n* L2 norm of residual is {l2_residual}')
    print(f'\n* L2 norm of NR error is {l2_NR}\n')

    outFile_L2_errors = open(os.path.join(outdir,'Algorithm/L2_errors.txt'), 'w')
    outFile_L2_errors.write('# L2 norm of residual is \n')
    outFile_L2_errors.write(f'{l2_residual} \n')
    outFile_L2_errors.write('# L2 norm of NR error is \n')
    outFile_L2_errors.write(f'{l2_NR} \n')

    return 

def init_plotting():

    """
    
    Function to set the default plotting parameters.

    Parameters
    ----------
    None

    Returns
    -------
    Nothing, but sets the default plotting parameters.
    
    """
    
    plt.rcParams['figure.max_open_warning'] = 0
    
    plt.rcParams['mathtext.fontset']  = 'stix'
    plt.rcParams['font.family']       = 'STIXGeneral'

    plt.rcParams['font.size']         = 14
    plt.rcParams['axes.linewidth']    = 1
    plt.rcParams['axes.labelsize']    = plt.rcParams['font.size']
    plt.rcParams['axes.titlesize']    = 1.5*plt.rcParams['font.size']
    plt.rcParams['legend.fontsize']   = plt.rcParams['font.size']
    plt.rcParams['xtick.labelsize']   = plt.rcParams['font.size']
    plt.rcParams['ytick.labelsize']   = plt.rcParams['font.size']
    plt.rcParams['xtick.major.size']  = 3
    plt.rcParams['xtick.minor.size']  = 3
    plt.rcParams['xtick.major.width'] = 1
    plt.rcParams['xtick.minor.width'] = 1
    plt.rcParams['ytick.major.size']  = 3
    plt.rcParams['ytick.minor.size']  = 3
    plt.rcParams['ytick.major.width'] = 1
    plt.rcParams['ytick.minor.width'] = 1
    
    plt.rcParams['legend.frameon']             = False
    plt.rcParams['legend.loc']                 = 'center left'
    plt.rcParams['contour.negative_linestyle'] = 'solid'
    
    plt.gca().spines['right'].set_color('none')
    plt.gca().spines['top'].set_color('none')
    plt.gca().xaxis.set_ticks_position('bottom')
    plt.gca().yaxis.set_ticks_position('left')
    
    return

def compare_with_GR_QNMs(results_object, qnm_cached, NR_sim, outdir):

    l,m              = NR_sim.l, NR_sim.m
    f_samples        = results_object['f_0']
    f_rd_fundamental = qnm_cached[(2,l,m,0)]['f']

    plt.figure()
    sns.histplot(f_samples       , color="darkred", fill=True , alpha=0.9, label='EFT fund mode')
    plt.axvline(f_rd_fundamental, color='black', linestyle='--', lw=2.2,  label='GR fund mode')
    plt.xlabel(r'$f_{fund}$')
    plt.ylabel(r'$p(f_{fund})$')
    plt.legend(loc='best')
    plt.savefig(os.path.join(outdir,'Plots/Results/f_fundamental.pdf'), bbox_inches='tight')

    return 

def plot_NR_vs_model(NR_sim, template, metadata, results, nest_model, outdir, method, tail_flag):

    """

    Plot the NR waveform against the model waveform.

    Parameters
    ----------

    NR_sim : NR_sim
        NR simulation object.

    template : template
        Template object.

    metadata : dict
        Dictionary containing the metadata.

    results : dict
        Dictionary containing the results object.

    nest_model : nest_model
        Nested sampling model object.

    outdir : string
        Output directory.

    method : string
        Method used to fit the waveform.

    Returns
    -------

    Nothing, but plots the simulation/model comparison and saves the figure.

    """

    init_plotting()

    NR_r, NR_i, NR_r_err, NR_i_err, NR_amp, NR_f, t_NR, t_peak                                                = NR_sim.NR_r, NR_sim.NR_i, np.real(NR_sim.NR_err_cmplx), np.imag(NR_sim.NR_err_cmplx), NR_sim.NR_amp, NR_sim.NR_freq, NR_sim.t_NR, NR_sim.t_peak
    t_cut, tM_start, tM_end, NR_r_cut, NR_i_cut, NR_r_err_cut, NR_i_err_cut, NR_amp_cut, NR_phi_cut, NR_f_cut = NR_sim.t_NR_cut, NR_sim.tM_start, NR_sim.tM_end, NR_sim.NR_r_cut, NR_sim.NR_i_cut, np.real(NR_sim.NR_cpx_err_cut), np.imag(NR_sim.NR_cpx_err_cut), NR_sim.NR_amp_cut, NR_sim.NR_phi_cut, NR_sim.NR_freq_cut

    wf_data_type = NR_sim.waveform_type

    l,m = NR_sim.l, NR_sim.m

    if(NR_sim.NR_catalog=='cbhdb' or NR_sim.NR_catalog=='charged_raw'):
        f_rd_fundamental = template.qnm_cached[(2,l,m,0)]['f']
    else:
        f_rd_fundamental = template.qnm_cached[(2,l,m,0)]['f']

    try:
        m1, m2, chi1, chi2 = metadata['m1'], metadata['m2'], metadata['chi1'], metadata['chi2'],
        f_peak             = utils.F_mrg_Nagar(m1, m2, chi1, chi2, geom=1)
    except:
        f_peak             = None

    lw_small        = 0.5
    lw_medium       = 1.2
    lw_std          = 1.8
    lw_large        = 2.2

    color_NR        = 'k'
    color_model     = '#cc0033'
    color_t_start   = 'mediumseagreen' #'#990066', '#cc0033', '#ff0000'
    color_t_peak    = 'royalblue'

    alpha_std       = 1.0
    alpha_med       = 0.8

    ls_t            = '--'
    ls_f            = '--'

    if(tail_flag) :
        fontsize_legend = 20
        fontsize_labels = 25
        color_f_ring    = 'royalblue'
    else:
        fontsize_legend = 18
        fontsize_labels = 23
        color_f_ring    = 'forestgreen'

    if(not(tail_flag) and not(wf_data_type=='psi4') and (NR_sim.NR_catalog=='SXS' or NR_sim.NR_catalog=='RIT')): tM_end = 80
    if(wf_data_type=='psi4'): 
        tM_end = 120
        label_data = '\psi_{4,%s%s}'%(l,m)
    else:
        label_data = 'h_{%s%s}'%(l,m)

    ########################
    # Waveforms comparison #
    ########################

    if(tail_flag):
        f   = plt.figure(figsize=(8,12))
        ax2 = plt.subplot(2,1,1)
        ax4 = plt.subplot(2,1,2)
        
        rescale = 1.4
    else:
        f   = plt.figure(figsize=(12,8))
        ax1 = plt.subplot(2,2,1)
        ax2 = plt.subplot(2,2,2)
        ax3 = plt.subplot(2,2,3)
        ax4 = plt.subplot(2,2,4)
  
        ax1.set_xlim([-10, tM_end])
        ax3.set_xlim(ax1.get_xlim())

        rescale = 1.0

    ax2.set_xlim(-10, tM_end)
    ax4.set_xlim(ax2.get_xlim())

    # Plot NR data

    if not(tail_flag):
        ax1.plot(t_NR - t_peak, NR_r,                                                      c=color_NR,      lw=lw_std,    alpha=alpha_std, ls='-' )
        ax1.axvline(tM_start,                                                              c=color_t_start, lw=lw_std,    alpha=alpha_std, ls=ls_t)
        ax1.axvline(0.0,                          label=r'$t_{\rm peak}$',            c=color_t_peak,  lw=lw_std,    alpha=alpha_std, ls=ls_t)
        ax1.set_ylabel(r'$\mathrm{Re[%s]}$'%(label_data), fontsize=fontsize_labels)

        ax3.plot(t_NR - t_peak, NR_i,                                                      c=color_NR,      lw=lw_std,    alpha=alpha_std, ls='-' )
        ax3.axvline(tM_start, label=r'$t_{\rm start} = t_{\rm peak} \, + %d \mathrm{M}}$'%tM_start, c=color_t_start, lw=lw_std,    alpha=alpha_std, ls=ls_t)
        ax3.axvline(0.0,                                                                   c=color_t_peak,  lw=lw_std,    alpha=alpha_std, ls=ls_t)
        ax3.set_ylabel(r'$\mathrm{Im[%s]}$'%(label_data), fontsize=fontsize_labels)
        ax3.set_xlabel(r'$t - t_{peak} \, [\mathrm{M}]$', fontsize=fontsize_labels)

    ax2.semilogy(t_NR - t_peak, NR_amp, label=r'$\mathrm{NR}$',                            c=color_NR,      lw=lw_std,    alpha=alpha_std, ls='-' )
    ax2.axvline(tM_start,                                                                  c=color_t_start, lw=lw_std,    alpha=alpha_std, ls=ls_t)
    if(not(tail_flag)): ax2.axvline(0.0,                                                   c=color_t_peak,  lw=lw_std,    alpha=alpha_std, ls=ls_t)
    if(not(tail_flag) and (NR_sim.NR_catalog=='SXS' or NR_sim.NR_catalog=='RIT')): ax2.set_ylim([1e-6*np.max(NR_amp), 2*np.max(NR_amp)])
    elif(  tail_flag  and (NR_sim.NR_catalog=='SXS' or NR_sim.NR_catalog=='RIT')): ax2.set_ylim([2*1e-4, 2*np.max(NR_amp)])
    ax2.set_xlabel(r'$\mathrm{t - t_{peak} \, [M}]$', fontsize=fontsize_labels)

    ax4.plot(t_NR - t_peak, NR_f,                                                          c=color_NR,      lw=lw_std,    alpha=alpha_std, ls='-' )
    ax4.axhline(f_rd_fundamental, label=r'$\mathit{f_{%d%d0}}$'%(l,m),                     c=color_f_ring,  lw=lw_std,    alpha=alpha_std, ls=ls_f)
    if(tail_flag): 
        ax4.axhline(0.0,      label=r'$\mathit{f_{\rm tail}}$',                            c=color_model,   lw=lw_std,    alpha=alpha_std, ls=ls_t)
        ax4.axvline(tM_start, label=r'$\mathrm{t_{start} = t_{peak} \, + %d M}$'%tM_start, c=color_t_start, lw=lw_std,    alpha=alpha_std, ls=ls_t)
        ax4.axvline(0.0,                                                                   c=color_t_peak,  lw=lw_std,    alpha=alpha_std, ls=ls_t)
    else         : 
        ax4.axvline(0.0,                                                                   c=color_t_peak,  lw=lw_std,    alpha=alpha_std, ls=ls_t)
    ax4.set_xlabel(r'$t - t_{peak} \, [\mathrm{M}]$'    , fontsize=fontsize_labels)

    # Find the index of zero
    t_peak_idx = np.argmin(np.abs(t_NR - t_peak))
    
    if not(tail_flag):
        try   : ax4.set_ylim([-1.5*NR_f[t_peak_idx], 3.5*NR_f[t_peak_idx]])
        except: pass
    else:
        ax4.set_ylim([-0.08, 0.28])

    if not(nest_model==None):

        # Plot waveform reconstruction
        if(method=='Nested-sampler'):
            models_re_list = [np.real(np.array(nest_model.model(p))) for p in results]
            models_im_list = [np.imag(np.array(nest_model.model(p))) for p in results]

        for perc in [50, 5, 95]:

            if(method=='Nested-sampler'):
                wf_r = np.percentile(np.array(models_re_list),[perc], axis=0)[0]
                wf_i = np.percentile(np.array(models_im_list),[perc], axis=0)[0]
            else:
                wf_r = np.real(np.array(nest_model.model(results)))
                wf_i = np.imag(np.array(nest_model.model(results)))

            wf_amp, wf_phi = waveform_utils.amp_phase_from_re_im(wf_r, wf_i)
            wf_f           = np.gradient(wf_phi, t_cut)/(twopi)
            
            if(perc==50):
                if not(tail_flag):
                    ax1.plot(t_cut - t_peak, wf_r,                                               c=color_model, lw=lw_large*rescale, alpha=alpha_std, ls='-' )
                    ax3.plot(t_cut - t_peak, wf_i,                                               c=color_model, lw=lw_large*rescale, alpha=alpha_std, ls='-' )
                ax2.semilogy(t_cut - t_peak, wf_amp, label=r'$\mathrm{%s}$'%(template.wf_model), c=color_model, lw=lw_large*rescale, alpha=alpha_std, ls='-' )
                ax4.plot(    t_cut - t_peak, wf_f,                                               c=color_model, lw=lw_large*rescale, alpha=alpha_std, ls='-' )
            else:
                if not(tail_flag):
                    ax1.plot(t_cut - t_peak, wf_r,                                               c=color_model, lw=lw_std,           alpha=alpha_med, ls='--' )
                    ax3.plot(t_cut - t_peak, wf_i,                                               c=color_model, lw=lw_std,           alpha=alpha_med, ls='--' )
                ax2.semilogy(t_cut - t_peak, wf_amp,                                             c=color_model, lw=lw_std,           alpha=alpha_med, ls='--' )
                ax4.plot(    t_cut - t_peak, wf_f,                                               c=color_model, lw=lw_std,           alpha=alpha_med, ls='--' )


        if(tail_flag):
            # for name_x in results.names:
            #     if ('ln_A_tail' in name_x):
            #         results[name_x] = np.log(1e-32)
            try   : results['ln_A_tail_22'] = np.log(1e-32)
            except: pass
            try   : results['ln_A_tail_32'] = np.log(1e-32)
            except: pass

            # Plot QNM waveform reconstruction
            if(method=='Nested-sampler'):
                models_re_list = [np.real(np.array(nest_model.model(p))) for p in results]
                models_im_list = [np.imag(np.array(nest_model.model(p))) for p in results]
            
            for perc in [50, 5, 95]:

                if(method=='Nested-sampler'):
                    wf_r = np.percentile(np.array(models_re_list),[perc], axis=0)[0]
                    wf_i = np.percentile(np.array(models_im_list),[perc], axis=0)[0]
                else:
                    wf_r = np.real(np.array(nest_model.model(results)))
                    wf_i = np.imag(np.array(nest_model.model(results)))

                wf_amp, wf_phi = waveform_utils.amp_phase_from_re_im(wf_r, wf_i)
                wf_f           = np.gradient(wf_phi, t_cut)/(twopi)
                
                if(perc==50):
                    ax2.semilogy(t_cut - t_peak, wf_amp, label=r'$\mathrm{%s \,\, QNMs}$'%(template.wf_model), c='royalblue', lw=lw_large*1.4, alpha=alpha_std, ls='-' )
                    ax4.plot(    t_cut - t_peak, wf_f,                                                         c='royalblue', lw=lw_large*1.4, alpha=alpha_std, ls='-' )
                else:
                    ax2.semilogy(t_cut - t_peak, wf_amp,                                                       c='royalblue', lw=lw_std,       alpha=alpha_med, ls='--' )
                    ax4.plot(    t_cut - t_peak, wf_f,                                                         c='royalblue', lw=lw_std,       alpha=alpha_med, ls='--' )


                if(method=='Minimization'): break

    if not(tail_flag):
        ax1.set_ylabel(r'$\mathit{Re(%s)}$'%(label_data), fontsize=fontsize_labels*rescale)
        ax3.set_ylabel(r'$\mathit{Im(%s)}$'%(label_data), fontsize=fontsize_labels*rescale)
    ax2.set_ylabel(    r'$\mathit{A_{%d%d}(t)}$'%(l,m)  , fontsize=fontsize_labels*rescale)
    ax4.set_ylabel(    r'$\mathit{f_{%d%d}\,(t)}$'%(l,m), fontsize=fontsize_labels*rescale)

    plt.rcParams['legend.frameon'] = True

    ax2.legend(    loc='best', fontsize=fontsize_legend, shadow=True)
    ax4.legend(    loc='best', fontsize=fontsize_legend, shadow=True)

    if not(tail_flag): 
        ax1.legend(loc='best', fontsize=fontsize_legend, shadow=True)
        ax3.legend(loc='best', fontsize=fontsize_legend, shadow=True)
        ax1.get_shared_x_axes().join(ax1, ax3)
        ax1.set_xticklabels([])
        plt.suptitle('{}-{}'.format(NR_sim.NR_catalog, NR_sim.NR_ID), size=28)

    ax2.get_shared_x_axes().join(ax2, ax4)
    ax2.set_xticklabels([])
    plt.tight_layout(rect=[0,0,1,0.95])
    plt.subplots_adjust(hspace=0, wspace=0.27)
    if(tail_flag): leg_name_tail = '_tail'
    else         : leg_name_tail = ''
    plt.savefig(os.path.join(outdir, f'Plots/Comparisons/Waveform_reconstruction{leg_name_tail}.pdf'), bbox_inches='tight')

    if (tail_flag): plt.rcParams['legend.frameon'] = False

    if (nest_model==None): return

    ############################
    # Residuals reconstruction #
    ############################

    f   = plt.figure(figsize=(12,8))
    ax1 = plt.subplot(2,2,1)
    ax2 = plt.subplot(2,2,2)
    ax3 = plt.subplot(2,2,3)
    ax4 = plt.subplot(2,2,4)

    ax1.set_xlim([tM_start, tM_end])
    ax2.set_xlim(ax1.get_xlim())
    ax3.set_xlim(ax1.get_xlim())
    ax4.set_xlim(ax1.get_xlim())

    ax1.errorbar(t_cut - t_peak, np.zeros(len(NR_r_cut)), yerr=np.array(NR_r_err_cut), label=r'$\mathrm{NR error}$', c=color_NR, lw=lw_small, alpha=alpha_std, ls='-', capsize=0.15)
    ax3.errorbar(t_cut - t_peak, np.zeros(len(NR_i_cut)), yerr=np.array(NR_i_err_cut),                               c=color_NR, lw=lw_small, alpha=alpha_std, ls='-', capsize=0.15)

    for perc in [50, 5, 95]:
    
        if(method=='Nested-sampler'):
            wf_r = np.percentile(np.array(models_re_list),[perc], axis=0)[0]
            wf_i = np.percentile(np.array(models_im_list),[perc], axis=0)[0]
        else:
            wf_r = np.real(np.array(nest_model.model(results)))
            wf_i = np.imag(np.array(nest_model.model(results)))

        if(perc==50):
            ax1.plot(t_cut - t_peak, wf_r   - NR_r_cut  ,                                                  c=color_model, lw=lw_large, alpha=alpha_std, ls='-' )
            ax2.plot(t_cut - t_peak, wf_amp - NR_amp_cut,                                                  c=color_model, lw=lw_large, alpha=alpha_std, ls='-' )
            ax3.plot(t_cut - t_peak, wf_i   - NR_i_cut  , label=r'$\mathrm{%s - NR}$'%(template.wf_model), c=color_model, lw=lw_large, alpha=alpha_std, ls='-' )
            ax4.plot(t_cut - t_peak, wf_f   - NR_f_cut  ,                                                  c=color_model, lw=lw_large, alpha=alpha_std, ls='-' )
        else:
            ax1.plot(t_cut - t_peak, wf_r   - NR_r_cut  ,                                                  c=color_model, lw=lw_std, alpha=alpha_med, ls='--')
            ax2.plot(t_cut - t_peak, wf_amp - NR_amp_cut,                                                  c=color_model, lw=lw_std, alpha=alpha_med, ls='--')
            ax3.plot(t_cut - t_peak, wf_i   - NR_i_cut  ,                                                  c=color_model, lw=lw_std, alpha=alpha_med, ls='--')
            ax4.plot(t_cut - t_peak, wf_f   - NR_f_cut  ,                                                  c=color_model, lw=lw_std, alpha=alpha_med, ls='--')
            
        if(method=='Minimization'): break

    ax1.legend(loc='best', fontsize=fontsize_legend)
    ax3.legend(loc='best', fontsize=fontsize_legend)

    ax1.set_ylabel(r'$\mathit{Re(%s)}$'%(label_data), fontsize=fontsize_labels)
    ax2.set_ylabel(r'$\mathit{A(t)}$'               , fontsize=fontsize_labels)
    ax3.set_ylabel(r'$\mathit{Im(%s)}$'%(label_data), fontsize=fontsize_labels)
    ax4.set_ylabel(r'$\mathit{f\,(t)}$'             , fontsize=fontsize_labels)

    ax3.set_xlabel(r'$t - t_{peak} \, [\mathrm{M}]$', fontsize=fontsize_labels)
    ax4.set_xlabel(r'$t - t_{peak} \, [\mathrm{M}]$', fontsize=fontsize_labels)

    ax1.get_shared_x_axes().join(ax1, ax3)
    ax2.get_shared_x_axes().join(ax2, ax4)
    ax1.set_xticklabels([])
    ax2.set_xticklabels([])
    plt.suptitle('{}-{} residuals'.format(NR_sim.NR_catalog, NR_sim.NR_ID), size=28)
    plt.tight_layout(rect=[0,0,1,0.95])
    plt.subplots_adjust(hspace=0, wspace=0.3)

    plt.savefig(os.path.join(outdir, 'Plots/Comparisons/Residuals_reconstruction.pdf'), bbox_inches='tight')

    # Decay rate
    if(tail_flag):

        plt.figure(figsize=(6,6))

        log_t_NR         = np.log(t_NR  - t_peak)
        log_t_cut        = np.log(t_cut - t_peak)

        log_A_NR         = np.log(NR_amp)
        dlog_A_NR_dlog_t = utils.diff1(log_t_NR, log_A_NR)

        models_re_list = [np.real(np.array(nest_model.model(p))) for p in results]
        models_im_list = [np.imag(np.array(nest_model.model(p))) for p in results]
        
        for perc in [50, 5, 95]:
            wf_r = np.percentile(np.array(models_re_list),[perc], axis=0)[0]
            wf_i = np.percentile(np.array(models_im_list),[perc], axis=0)[0]

            wf_amp, _ = waveform_utils.amp_phase_from_re_im(wf_r, wf_i)

            log_A_wf         = np.log(wf_amp)
            dlog_A_wf_dlog_t = utils.diff1(log_t_cut, log_A_wf)

            plt.plot(t_cut - t_peak, dlog_A_wf_dlog_t, c=color_model, lw=lw_std, alpha=alpha_med, ls='-')

        plt.axhline(0.0, c='k', ls='--', lw=0.7)
        plt.axhline(-1.0, c='mediumseagreen', ls='--',  label='Okuzumi+',  lw=1.2)
        plt.axhline(-1.3, c='crimson',        ls='--', label='Albanesi+', lw=1.7)
        plt.plot(    t_NR  - t_peak, dlog_A_NR_dlog_t, c=color_NR,    lw=lw_std, alpha=alpha_med, ls='-')
        plt.xlim([75, 100])
        plt.ylim([-3.4, 1.5])
        plt.xlabel(r'$t - t_{peak} \, [\mathrm{M}]$',                   fontsize=fontsize_labels)
        plt.ylabel(r'$\mathrm{p}$', fontsize=fontsize_labels)
        plt.legend(loc='best', fontsize=fontsize_labels*0.8)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, 'Plots/Comparisons/Decay_rate.pdf'), bbox_inches='tight')

    return

def plot_fancy_residual(NR_sim, template, metadata, results, nest_model, outdir, method):

    """

    Plot the residuals vs the NR error in a single figure.

    Parameters
    ----------

    NR_sim : NR_sim
        NR simulation object.

    template : template
        Template object.

    metadata : dict
        Dictionary containing the metadata.

    results : dict
        Dictionary containing the results object.

    nest_model : nest_model
        Nested sampling model object.

    outdir : string
        Output directory.

    method : string
        Method used to fit the waveform.

    Returns
    -------

    Nothing.

    """
    plt.rcParams["mathtext.fontset"]  = "stix"
    plt.rcParams["font.family"]       = "STIXGeneral"
    plt.rcParams["font.size"]         = 14
    plt.rcParams["legend.fontsize"]   = 12
    plt.rcParams["xtick.labelsize"]   = 10
    plt.rcParams["ytick.labelsize"]   = 10
    plt.rcParams["xtick.major.size"]  = 3
    plt.rcParams["xtick.minor.size"]  = 3
    plt.rcParams["xtick.major.width"] = 1
    plt.rcParams["xtick.minor.width"] = 1
    plt.rcParams["ytick.major.size"]  = 3
    plt.rcParams["ytick.minor.size"]  = 3
    plt.rcParams["ytick.major.width"] = 1
    plt.rcParams["ytick.minor.width"] = 1

    t_peak =  NR_sim.t_peak
    t_cut, tM_start, tM_end, NR_r_cut, NR_i_cut, NR_r_err_cut, NR_i_err_cut = NR_sim.t_NR_cut, NR_sim.tM_start, NR_sim.tM_end, NR_sim.NR_r_cut, NR_sim.NR_i_cut, np.real(NR_sim.NR_cpx_err_cut), np.imag(NR_sim.NR_cpx_err_cut)

    l,m = NR_sim.l, NR_sim.m

    lw_small = 0.1
    lw_std   = 1.0
    lw_large = 1.5

    if not(nest_model==None):

        # Plot waveform reconstruction
        if(method=='Nested-sampler'):
            models_re_list = [np.real(np.array(nest_model.model(p))) for p in results]
            models_im_list = [np.imag(np.array(nest_model.model(p))) for p in results]

    ###########################
    # Waveform reconstruction #
    ###########################

    f, [ax1, ax2]   = plt.subplots(nrows = 2, ncols = 1, figsize=(4,7))

    ax1.set_xlim([tM_start, tM_end])
    ax2.set_xlim(ax1.get_xlim())

    ax1.fill_between(t_cut - t_peak, -np.array(NR_r_err_cut), np.array(NR_r_err_cut), color = 'dimgray', alpha = 0.2, label = r'$\mathrm{Numerical Error} $')
    ax2.fill_between(t_cut - t_peak, -np.array(NR_i_err_cut), np.array(NR_i_err_cut), color = 'dimgray', alpha = 0.2, label = r'$\mathrm{Numerical Error} $')

    wf_r_m = np.percentile(np.array(models_re_list), [5], axis=0)[0]
    wf_r   = np.percentile(np.array(models_re_list),[50], axis=0)[0]
    wf_r_P = np.percentile(np.array(models_re_list),[95], axis=0)[0]

    wf_i_m = np.percentile(np.array(models_im_list), [5], axis=0)[0]
    wf_i   = np.percentile(np.array(models_im_list),[50], axis=0)[0]
    wf_i_P = np.percentile(np.array(models_im_list),[95], axis=0)[0]

    ax1.plot(t_cut - t_peak, wf_r - NR_r_cut, label=r'$\mathrm{Residual}$', c='firebrick', lw=lw_std)
    ax2.plot(t_cut - t_peak, wf_i - NR_i_cut, c='firebrick', lw=lw_std)

    ax1.fill_between(t_cut - t_peak, wf_r_m - NR_r_cut, wf_r_P - NR_r_cut, color = 'firebrick', alpha = 0.2)
    ax2.fill_between(t_cut - t_peak, wf_i_m - NR_i_cut, wf_i_P - NR_i_cut, color = 'firebrick', alpha = 0.2)

    ax1.legend(loc='best')
    if(NR_sim.catalog == 'Teukolsky'):
        ax1.set_ylabel(r'$\Re(\Psi_{4,%i%i})$'%(l,m))
        ax2.set_ylabel(r'$\Im(\Psi_{4,%i%i})$'%(l,m))
    else:
        ax1.set_ylabel(r'$\Re(h_{%i%i})$'%(l,m))
        ax2.set_ylabel(r'$\Im(h_{%i%i})$'%(l,m))
    ax2.set_xlabel(r'$t/M$')
    for ax in [ax1, ax2]: ax.set_xlim([tM_start, 80])
    plt.suptitle('{}-{} residuals'.format(NR_sim.NR_catalog, NR_sim.NR_ID), size=28)
    plt.tight_layout(rect=[0,0,1,0.95])
    plt.savefig(os.path.join(outdir, 'Plots/Comparisons/Fancy_Residuals.pdf'), bbox_inches='tight')

    return

def plot_fancy_reconstruction(NR_sim, template, metadata, results, nest_model, outdir, method):

    """

    Plot the NR waveform and its reconstruction

    Parameters
    ----------

    NR_sim : NR_sim
        NR simulation object.

    template : template
        Template object.

    metadata : dict
        Dictionary containing the metadata.

    results : dict
        Dictionary containing the results object.

    nest_model : nest_model
        Nested sampling model object.

    outdir : string
        Output directory.

    method : string
        Method used to fit the waveform.

    Returns
    -------

    Nothing.

    """
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["font.family"] = "STIXGeneral"
    plt.rcParams["font.size"] = 14
    plt.rcParams["legend.fontsize"] = 12
    plt.rcParams["xtick.labelsize"] = 10
    plt.rcParams["ytick.labelsize"] = 10
    plt.rcParams["xtick.major.size"] = 3
    plt.rcParams["xtick.minor.size"] = 3
    plt.rcParams["xtick.major.width"] = 1
    plt.rcParams["xtick.minor.width"] = 1
    plt.rcParams["ytick.major.size"] = 3
    plt.rcParams["ytick.minor.size"] = 3
    plt.rcParams["ytick.major.width"] = 1
    plt.rcParams["ytick.minor.width"] = 1

    NR_r, NR_i, NR_r_err, NR_i_err, NR_amp, NR_f, t_NR, t_peak                                                = NR_sim.NR_r, NR_sim.NR_i, np.real(NR_sim.NR_err_cmplx), np.imag(NR_sim.NR_err_cmplx), NR_sim.NR_amp, NR_sim.NR_freq, NR_sim.t_NR, NR_sim.t_peak
    t_cut, tM_start, tM_end, NR_r_cut, NR_i_cut, NR_r_err_cut, NR_i_err_cut, NR_amp_cut, NR_phi_cut, NR_f_cut = NR_sim.t_NR_cut, NR_sim.tM_start, NR_sim.tM_end, NR_sim.NR_r_cut, NR_sim.NR_i_cut, np.real(NR_sim.NR_cpx_err_cut), np.imag(NR_sim.NR_cpx_err_cut), NR_sim.NR_amp_cut, NR_sim.NR_phi_cut, NR_sim.NR_freq_cut

    l,m = NR_sim.l, NR_sim.m

    f_rd_fundamental = template.qnm_cached[(2,l,m,0)]['f']

    try:
        m1, m2, chi1, chi2 = metadata['m1'], metadata['m2'], metadata['chi1'], metadata['chi2'],
        f_peak             = utils.F_mrg_Nagar(m1, m2, chi1, chi2) * (G_SI*C_SI**(-3))
    except:
        f_peak             = None

    lw_small = 0.1
    lw_std   = 1.0
    lw_large = 1.2

    if not(nest_model==None):

        # Plot waveform reconstruction
        if(method=='Nested-sampler'):
            models_re_list = [np.real(np.array(nest_model.model(p))) for p in results]
            models_im_list = [np.imag(np.array(nest_model.model(p))) for p in results]

    f, [ax1, ax2]   = plt.subplots(nrows = 2, ncols = 1, figsize=(4,7))

    ax1.set_xlim([-10, tM_end])
    ax2.set_xlim(ax1.get_xlim())

    ax1.plot(t_NR - t_peak, NR_r, color = 'black', label = r'$\mathrm{Numerical Data} $', lw = lw_large)
    ax2.plot(t_NR - t_peak, NR_i, color = 'black', lw = lw_large)
    
    wf_r =  np.percentile(np.array(models_re_list),[50], axis=0)[0]
    wf_i =  np.percentile(np.array(models_im_list),[50], axis=0)[0]

    wf_r_m = np.percentile(np.array(models_re_list),[5], axis=0)[0]
    wf_r_P = np.percentile(np.array(models_re_list),[95], axis=0)[0]

    wf_i_m = np.percentile(np.array(models_im_list),[5], axis=0)[0]
    wf_i_P = np.percentile(np.array(models_im_list),[95], axis=0)[0]

    ax1.plot(t_cut - t_peak, wf_r, label=r'$\mathrm{Reconstruction}$', c='firebrick', lw=lw_std)
    ax2.plot(t_cut - t_peak, wf_i, c='firebrick', lw=lw_std)

    ax1.fill_between(t_cut - t_peak, wf_r_m , wf_r_P, color = 'firebrick', alpha = 0.2)
    ax2.fill_between(t_cut - t_peak, wf_i_m , wf_i_P, color = 'firebrick', alpha = 0.2)

    for ax in [ax1, ax2]:
        ax.axvline(tM_start,           label=r'$t_{\rm start} = t_{\rm peak} + $' + rf'${np.round(tM_start)}M$', c='darkgreen', lw=lw_large, alpha=1.0, ls=':' )
        ax.axvline(0.0,                label=r'$t_{\rm peak}$',                                               c='k',         lw=lw_large, alpha=1.0, ls='--')

    ax1.legend(loc='best', fontsize= 11)
    ax1.set_ylabel(r'$\Re(\Psi_4)$')
    ax2.set_ylabel(r'$\Im(\Psi_4)$')
    ax2.set_xlabel(r'$t/M$')
    for ax in [ax1, ax2]:   ax.set_xlim([-10, 80])
    # plt.suptitle('SXS:BBH:{} (residuals)'.format(NR_sim.NR_ID), size=24)
    plt.tight_layout(rect=[0,0,1,0.95])
    plt.savefig(os.path.join(outdir, 'Plots/Comparisons/Fancy_Reconstruction.pdf'), bbox_inches='tight')

    return

def global_corner(x, names, output, truths=None):

    """
    
    Create a corner plot of all parameters.
    
    Parameters
    ----------

    x       : dictionary    
        Dictionary of parameters.
    names   : list
        List of parameter names.
    output  : string
        Output directory.

    Returns
    -------

    Nothing, but saves a corner plot to the output directory.

    """

    samples = []
    for xy in names: samples.append(np.array(x[xy]))
    samples = np.transpose(samples)
    mask    = [i for i in range(samples.shape[-1]) if not all(samples[:,i]==samples[0,i]) ]

    fig = plt.figure(figsize=(10,10))
    C   = corner.corner(samples[:,mask],
                        quantiles     = [0.05, 0.5, 0.95],
                        labels        = names,
                        color         = 'darkred',
                        show_titles   = True,
                        title_kwargs  = {"fontsize": 12},
                        use_math_text = True,
                        truths = truths
                        )
    plt.savefig(os.path.join(output, 'Plots', 'Results', 'corner.png'), bbox_inches='tight')

    return
