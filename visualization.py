
def plot_efficiency(plants):
    raise NotImplementedError
    eff_bin_width = 0.25
    mags = np.arange(20, np.max(plants[:, 4])+eff_bin_width, eff_bin_width)
    n = mags*0.0
    f = [mags*0.0, mags*0.0, mags*0.0, mags*0.0]
    k = ((plants[:, 4]-mags[0])/(mags[1]-mags[0])).astype('int')
    for i in range(len(plants)):
        n[k[i]] += 1.
        for j in [5, 6, 7, 8]:
            if plants[i, j]:
                f[j-5][k[i]] += 1.
    labels = ['det', 'filt', 'clust', 'final']
    for j in range(len(labels)):
        pyl.scatter(mags+(mags[1]-mags[0])/2.+j*0.02,
                    f[j]/n, label=labels[j])
    pyl.legend()
    pyl.show()
