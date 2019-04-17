import numpy as np
import pylab
import nest
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_weight_matrices(E_neurons, I_neurons, enc_neurons, ntp, gamma):
    """
    ntp = number of exitatory neuron to plot
    gamma = ratio e_neurons/i_neurons, should be same as in network param to keep consistency 
    """
    import matplotlib.gridspec as gridspec
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    if len(enc_neurons) > ntp and len(enc_neurons) < len(E_neurons):
        enc_neurons = enc_neurons[0:int(ntp/2)]
    E_neurons = E_neurons[0:ntp]
    I_neurons = I_neurons[0:int(ntp*gamma)]

    
    
    W_enE = np.zeros([len(enc_neurons), len(E_neurons)])
    W_enI = np.zeros([len(enc_neurons), len(I_neurons)])
    W_EE = np.zeros([len(E_neurons), len(E_neurons)])
    W_EI = np.zeros([len(I_neurons), len(E_neurons)])
    W_IE = np.zeros([len(E_neurons), len(I_neurons)])
    W_II = np.zeros([len(I_neurons), len(I_neurons)])

    a_enE = nest.GetConnections(enc_neurons, E_neurons)
    c_enE = nest.GetStatus(a_enE, keys='weight')
    a_enI = nest.GetConnections(enc_neurons, I_neurons)
    c_enI = nest.GetStatus(a_enI, keys='weight')
    a_EE = nest.GetConnections(E_neurons, E_neurons)
    c_EE = nest.GetStatus(a_EE, keys='weight')
    a_EI = nest.GetConnections(I_neurons, E_neurons) # From In to En
    c_EI = nest.GetStatus(a_EI, keys='weight')
    a_IE = nest.GetConnections(E_neurons, I_neurons)
    c_IE = nest.GetStatus(a_IE, keys='weight')
    a_II = nest.GetConnections(I_neurons, I_neurons)
    c_II = nest.GetStatus(a_II, keys='weight')

    for idx, n in enumerate(a_enE):
        W_enE[n[0] - min(enc_neurons), n[1] - min(E_neurons)] += c_enE[idx]
    for idx, n in enumerate(a_enI):
        W_enI[n[0] - min(enc_neurons), n[1] - min(I_neurons)] += c_enI[idx]


    for idx, n in enumerate(a_EE):
        W_EE[n[0] - min(E_neurons), n[1] - min(E_neurons)] += c_EE[idx]
    for idx, n in enumerate(a_EI):
        W_EI[n[0] - min(I_neurons), n[1] - min(E_neurons)] += c_EI[idx]
    for idx, n in enumerate(a_IE):
        W_IE[n[0] - min(E_neurons), n[1] - min(I_neurons)] += c_IE[idx]
    for idx, n in enumerate(a_II):
        W_II[n[0] - min(I_neurons), n[1] - min(I_neurons)] += c_II[idx]

    #Setup the figure
    fig = pylab.figure()
    fig.suptitle('Weight matrices', fontsize=14)
    gs = gridspec.GridSpec(4, 5)
    ax0 = pylab.subplot(gs[:-1, 0:1])
    ax00 = pylab.subplot(gs[-1, 0:1])
    ax1 = pylab.subplot(gs[:-1, 1:-1])
    ax2 = pylab.subplot(gs[:-1, -1])
    ax3 = pylab.subplot(gs[-1, 1:-1])
    ax4 = pylab.subplot(gs[-1, -1])

    
    #
    plt0 = ax0.imshow(W_enE.T, cmap='jet')
    plt0.set_clim(0,1.)
    ax0.set_title('W_{enc->E}')

    #
    plt00 = ax00.imshow(W_enI.T, cmap='jet')
    plt00.set_clim(0,1.)
    ax00.set_title('W_{enc->I}')

    # 
    plt1 = ax1.imshow(W_EE, cmap='jet')
    plt1.set_clim(0,1.)
    ax1.set_title('W_{E->E}')

    plt2 = ax2.imshow(W_IE)
    plt2.set_cmap('jet')
    plt2.set_clim(0,1.)
    ax2.set_title('W_{E->I}')

    plt3 = ax3.imshow(-W_EI)
    plt3.set_cmap('jet')
    plt3.set_clim(0,1.)
    ax3.set_title('W_{I->E}')

    plt4 = ax4.imshow(-W_II)
    plt4.set_cmap('jet')
    plt4.set_clim(0,1.)
    ax4.set_title('W_{I->I}')

    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", "10%", pad="8%")
    pylab.colorbar(plt2, cax=cax)
    return fig


def plot_weight_matrices_old(E_neurons, I_neurons, enc_neurons):
    import matplotlib.gridspec as gridspec
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    W_enE = np.zeros([len(enc_neurons), len(E_neurons)])
    W_enI = np.zeros([len(enc_neurons), len(I_neurons)])
    W_EE = np.zeros([len(E_neurons), len(E_neurons)])
    W_EI = np.zeros([len(I_neurons), len(E_neurons)])
    W_IE = np.zeros([len(E_neurons), len(I_neurons)])
    W_II = np.zeros([len(I_neurons), len(I_neurons)])

    a_enE = nest.GetConnections(enc_neurons, E_neurons)
    c_enE = nest.GetStatus(a_enE, keys='weight')
    a_enI = nest.GetConnections(enc_neurons, I_neurons)
    c_enI = nest.GetStatus(a_enI, keys='weight')
    a_EE = nest.GetConnections(E_neurons, E_neurons)
    c_EE = nest.GetStatus(a_EE, keys='weight')
    a_EI = nest.GetConnections(I_neurons, E_neurons) # From In to En
    c_EI = nest.GetStatus(a_EI, keys='weight')
    a_IE = nest.GetConnections(E_neurons, I_neurons)
    c_IE = nest.GetStatus(a_IE, keys='weight')
    a_II = nest.GetConnections(I_neurons, I_neurons)
    c_II = nest.GetStatus(a_II, keys='weight')

    for idx, n in enumerate(a_enE):
        W_enE[n[0] - min(enc_neurons), n[1] - min(E_neurons)] += c_enE[idx]
    for idx, n in enumerate(a_enI):
        W_enI[n[0] - min(enc_neurons), n[1] - min(I_neurons)] += c_enI[idx]


    for idx, n in enumerate(a_EE):
        W_EE[n[0] - min(E_neurons), n[1] - min(E_neurons)] += c_EE[idx]
    for idx, n in enumerate(a_EI):
        W_EI[n[0] - min(I_neurons), n[1] - min(E_neurons)] += c_EI[idx]
    for idx, n in enumerate(a_IE):
        W_IE[n[0] - min(E_neurons), n[1] - min(I_neurons)] += c_IE[idx]
    for idx, n in enumerate(a_II):
        W_II[n[0] - min(I_neurons), n[1] - min(I_neurons)] += c_II[idx]

    #Setup the figure
    fig = pylab.figure()
    fig.suptitle('Weight matrices', fontsize=14)
    gs = gridspec.GridSpec(4, 5)
    ax0 = pylab.subplot(gs[0:3, 0:1])
    ax00 = pylab.subplot(gs[3:4, 0:1])
    ax1 = pylab.subplot(gs[:-1, 1:-1])
    ax2 = pylab.subplot(gs[:-1, -1])
    ax3 = pylab.subplot(gs[-1, 1:-1])
    ax4 = pylab.subplot(gs[-1, -1])

    #ax0 = pylab.subplot2grid((4,5),(0,0), rowspan=3)
    #ax00 = pylab.subplot2grid((4,5),(2,0))
    #ax1 = pylab.subplot2grid((4,5),(0,1), rowspan=3, colspan=2)
    #ax2 = pylab.subplot2grid((4,5),(0,3), rowspan=2)
    #ax3 = pylab.subplot2grid((4,5),(2,1), rowspan=3)
    #ax4 = pylab.subplot2grid((4,5),(3,3))
    #
    plt0 = ax0.imshow(W_enE.T, cmap='jet')
    plt0.set_clim(0,1.)
    ax0.set_title('W_{enc->E}')
    pylab.tight_layout()
    #
    plt00 = ax00.imshow(W_enI.T, cmap='jet')
    plt00.set_clim(0,1.)
    ax00.set_title('W_{enc->I}')
    pylab.tight_layout()

    # First part of the plot
    plt1 = ax1.imshow(W_EE, cmap='jet')
    plt1.set_clim(0,1.)
    ax1.set_title('W_{E->E}')
    pylab.tight_layout()


    plt2 = ax2.imshow(W_IE)
    plt2.set_cmap('jet')
    plt2.set_clim(0,1.)
    ax2.set_title('W_{E->I}')
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", "8%", pad="8%")
    pylab.colorbar(plt2, cax=cax)
    pylab.tight_layout()

    plt3 = ax3.imshow(-W_EI)
    plt3.set_cmap('jet')
    plt3.set_clim(0,1.)
    ax3.set_title('W_{I->E}')
    pylab.tight_layout()

    plt4 = ax4.imshow(-W_II)
    plt4.set_cmap('jet')
    plt4.set_clim(0,1.)
    ax4.set_title('W_{I->I}')
    pylab.tight_layout()

    # Allocate small space for the colorbar
    #divider = make_axes_locatable(ax1)
    #cax = divider.append_axes("right", "5%", pad="3%")

    #cb = pylab.colorbar( ax = [ax3,ax4], orientation ='vertical')