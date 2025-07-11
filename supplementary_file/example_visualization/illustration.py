# Re-import necessary libraries since execution state was reset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import os


save_dir = "illustration_landscape_distribution"
os.makedirs(save_dir, exist_ok=True)

x = np.linspace(-2, 3, 30)
y = np.linspace(-1, 3, 30)
X, Y = np.meshgrid(x, y)

Z = (np.sin(1.5 * X) * np.cos(1.5 * Y) +
     np.exp(-X**2 - Y**2) * 2 +
     1.5 * np.exp(-((X - 1.5)**2 + (Y - 0.5)**2) * 3) -
     1.2 * np.exp(-((X + 1.2)**2 + (Y + 1.2)**2) * 3))

Z = np.round(Z, 1)

colormap = sns.color_palette("magma", as_cmap=True)

# ----------- Configuration Landscape -----------
fig1 = plt.figure(figsize=(6, 6))
ax1 = fig1.add_subplot(111, projection='3d')
surf = ax1.plot_surface(X, Y, Z, cmap='Spectral', edgecolor='black', linewidth=0.5)

ax1.set_xlabel("option 1", fontsize=18, labelpad=3)
ax1.set_ylabel("option 2", fontsize=18, labelpad=3)

ax1.view_init(elev=30, azim=-45)


ax1.set_xticklabels([])
ax1.set_yticklabels([])
ax1.set_zticklabels([])


landscape_path = os.path.join(save_dir, "configuration_landscape.pdf")
plt.savefig(landscape_path, bbox_inches='tight', format='pdf')


plt.close(fig1)


# ----------- Performance Distribution -----------
fig2, ax2 = plt.subplots(figsize=(6, 6))


performance_values = Z.ravel()


sns.kdeplot(performance_values, fill=True, color='grey', ax=ax2, linewidth=1.5, bw_adjust=0.5)

ax2.set_xticks([])
ax2.set_yticks([])
ax2.set_ylabel("")


for spine in ax2.spines.values():
    spine.set_visible(False)

performance_path = os.path.join(save_dir, "performance_distribution.pdf")
plt.savefig(performance_path, bbox_inches='tight', format='pdf')

plt.close(fig2)

