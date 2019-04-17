def enc_layer_description(enc_layer):
############# Describe enc_layer
    print(' Info about the encoding layer :')
    print(' # of neurons = '+ str(len(nest.GetStatus(enc_layer))))
    print(' neuronal model used: '+ str(nest.GetStatus(enc_layer)[0]['model']))
    fig = pl.figure()
    fig.suptitle('Encoding layer')
    ax11 = fig.add_subplot(121)
    ax12 = fig.add_subplot(122, sharey=ax11)
    ax11.hist(Vm0)
    ax11.set_title('Initial membrane voltage')
    ax12.hist(thresholds)
    ax12.set_title('Threshold ')
    ax11.set_xlabel('Voltage')
    ax12.set_xlabel('Voltage')
    ax11.set_ylabel('# of neuron')
    pl.show()
    
    
def plt_thresh_Vm0(thresholds, Vm0):
############# Describe enc_layer
    import pylab as pl
    fig = pl.figure()
    fig.suptitle('Encoding layer')
    ax11 = fig.add_subplot(121)
    ax12 = fig.add_subplot(122, sharey=ax11)
    ax11.hist(Vm0)
    ax11.set_title('Initial membrane voltage')
    ax12.hist(thresholds)
    ax12.set_title('Threshold ')
    ax11.set_xlabel('Voltage')
    ax12.set_xlabel('Voltage')
    ax11.set_ylabel('# of neuron')
    pl.show()
    return fig