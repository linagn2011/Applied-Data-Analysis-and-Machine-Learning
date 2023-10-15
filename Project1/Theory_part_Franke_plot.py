n=50 #???
sigma = 0.1
def FrankeFunction(x,y):
	term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
	term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
	term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
	term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
	return term1 + term2 + term3 + term4

x = np.sort(np.random.uniform(0, 1, n))
y = np.sort(np.random.uniform(0, 1, n))
x, y = np.meshgrid(x,y)
z = FrankeFunction(x,y)+np.random.randn(n,n)*sigma

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.invert_yaxis()
FRANKE_plot = ax.plot_surface(x, y, z, cmap='terrain', antialiased=True)
ax.dist = 12 #so all labels are visible
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_title("Franke's function")
fig.savefig('Franke Function')
fig.colorbar(FRANKE_plot)
