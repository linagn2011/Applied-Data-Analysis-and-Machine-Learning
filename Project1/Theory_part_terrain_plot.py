import matplotlib.pyplot as plt

fuji_tif_file=imread("n35_e138_1arc_v3.tif")
fuji=fuji_tif_file[1500:3000:,1700:3200] #zooming in on the mountain, while still preserving quadratic shape

x = np.linspace(0, 1, np.shape(fuji)[0])
y = np.linspace(0, 1, np.shape(fuji)[1])
Y,X = np.meshgrid(y,x)
Z = np.array(fuji)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
terrain=ax.plot_surface(X,Y,Z, cmap="terrain", linewidth=0, antialiased=False)
ax.invert_xaxis()
fig.colorbar(terrain, pad=0.1)
ax.dist = 10
ax.set_xlabel("North-South")
ax.set_ylabel('East-West')
ax.set_zlabel('Altitude [m]')
ax.set_title("Mount Fuji DTM")
plt.show()
plt.savefig("Fuji_DTM_plot")
