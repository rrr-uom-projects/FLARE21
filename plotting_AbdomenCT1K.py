import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
matplotlib.rcParams.update({'font.size': 14})

def pos_multi(observers, observer_num, num_organs=4):
    pos_list = []
    for ordx in range(num_organs):
        pos_list.append(((observer_num) / (observers + 1)) + ordx)
    return np.array(pos_list)

res_orig = np.load("C:/PhD/FLARE21/full_res_results_grid.npy").reshape((5*73, 4, 2))[:-4]
res_orig = np.delete(res_orig, 79, 0) * 100

res_abdo = np.load("C:/PhD/FLARE21/abdo_full_res_results_grid.npy")[0] * 100

fig, (ax0, ax1) = plt.subplots(ncols=2)

organs = ("Liver", "Kidneys", "Spleen", "Pancreas")
cmap = matplotlib.colors.ListedColormap(["xkcd:azure", "xkcd:neon green", "xkcd:goldenrod", "xkcd:neon pink"])
colors = ["xkcd:azure", "xkcd:neon green", "xkcd:goldenrod", "xkcd:neon pink"]
# plot data
bps_orig = []
bps_abdo = []
g_s = np.array([[[95.5, 84.9, 92.6, 80], [98.2,96,98.5,90.1]],[[77.4,78.9,86,60.7],[92.1,92.4,97,82.3]]])
pos = np.array([0.9,1.9,2.9,3.9])
for ax, mdx in zip([ax0,ax1], range(2)):
    bps_orig.append(ax.boxplot([np.delete(res_orig[:,organ_idx,mdx], np.argwhere(~(res_orig[:,organ_idx,mdx]>0))) for organ_idx in range(4)], positions=pos_multi(2,1,4), widths=1/4, patch_artist=True, zorder=0, flierprops={"ms":3}))
    ax.plot(pos_multi(2,1,4), [np.nanmean(res_orig[ :, organ_idx, mdx]) for organ_idx in range(4)], "wD", mew=1, mec='k', ms=4)
    ax.plot(pos, [g_s[mdx, 0, organ_idx] for organ_idx in range(4)], "yv", ms=6)
    ax.plot(pos, [g_s[mdx, 1, organ_idx] for organ_idx in range(4)], "y^", ms=6)
    bps_abdo.append(ax.boxplot([np.delete(res_abdo[:,organ_idx,mdx], np.argwhere(~(res_abdo[:,organ_idx,mdx]>0))) for organ_idx in range(4)], positions=pos_multi(2,2,4), widths=1/4, patch_artist=True, zorder=0, flierprops={"ms":3}))
    ax.plot(pos_multi(2,2,4), [np.nanmean(res_abdo[ :, organ_idx, mdx]) for organ_idx in range(4)], "kD", mew=1, mec='k', ms=4)

pos = np.array([0.1,1.1,2.1,3.1])
def add_val(ax, mean_list, std_list):
    for mean, std, x in zip(mean_list, std_list, pos):
        ax.plot([x,x], [mean+std, mean-std], 'k', linewidth=1, markersize=0)
        ax.plot([x], [mean], 'gD', linewidth=0, markersize=6)
        cap_width = 0.075
        ax.plot([x-cap_width,x+cap_width], [mean+std, mean+std], 'k', linewidth=1, markersize=0)
        ax.plot([x-cap_width,x+cap_width], [mean-std, mean-std], 'k', linewidth=1, markersize=0)

add_val(ax0, [90.6,68.1,83.8,53.9], [13.5,29.5,21.6,26.6])
add_val(ax1, [62.7,54.8,66.1,38.2], [18.2,25.9,21.3,21.7])

# fill colors & median colors
for bp in bps_orig + bps_abdo:
    for color_idx, (patch, line) in enumerate(zip(bp['boxes'], bp['medians'])):
        patch.set_facecolor(colors[color_idx]) 
        line.set_color('xkcd:white')
        line.set_linewidth(1.5)

# ticks and labels
for ax in [ax0,ax1]:
    ax.set_xticks([0.5,1.5,2.5,3.5])
    ax.set_xticklabels(organs)
    ax.set_xticks(np.linspace(0,4,5), minor=True)
    ax.tick_params(which='minor', direction='inout', length=16, axis='x')
    ax.tick_params(which='both', direction='inout', length=0, axis='y')
    # make some of the ticks invisible
    for t in ax.xaxis.get_ticklines():
        t.set_color((0,0,0,0))
    ax.set_xlim(0,4)
    ax.set_ylim(0,100)
    ax.grid(which='both', axis='y')

ax0.set_ylabel("DSC (%)")
ax1.set_ylabel("NSD (%)")

m_s = []
m_s.append(mlines.Line2D([],[], mfc='w', mec='k', marker='D', linestyle='None', mew=1, markersize=10, label="Orig Training"))
m_s.append(mlines.Line2D([],[], mfc='y', mec='k', marker='^', linestyle='None', mew=0, markersize=10, label="SOTA range"))
m_s.append(mlines.Line2D([],[], mfc='g', mec='k', marker='D', linestyle='None', mew=0, markersize=10, label="Orig Validation"))
m_s.append(mlines.Line2D([],[], mfc='k', mec='k', marker='D', linestyle='None', mew=1, markersize=10, label="AbdomenCT-1K"))
ax0.legend(handles=m_s, loc='lower left', ncol=1)
plt.subplots_adjust(top=0.949,bottom=0.088,left=0.085,right=0.98,hspace=0.122,wspace=0.209)
plt.show()