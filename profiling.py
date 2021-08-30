import numpy as np
import matplotlib.pyplot as plt


##DEFINING CONSTANTS
H = 584-48 #height of ice sheet, minus for dead ice
h = 217-48 #step height, minus for dead ice
fb = 0.29 #bottom sliding coefficient
He = H - h*(1-fb) #effective height
lam = 0.33/(3600*24*365) #accumulation rate
Q = 48*10**-3 #geothermanl heat  flux
K = 2.43 #thermal conductivity
Ts = 253.95 #surface temp
t = 3600*24*30  #time step, one month
kappa = 44.5/(3600*24*365) #thermal diffusivity
delz = 1 #spatial step, 1 metre



##STEADYSTATE INITIALISATION
n_steady = int((H + 48/delz)) #number of model runs
z_steady = np.linspace(0, n_steady-1, n_steady) #height array
w_steady = np.zeros([n_steady, 1]) #vertical velocity array
coeffs_steady = np.zeros([n_steady, n_steady]) #coefficient matrix M
funcs_steady = np.zeros([n_steady, 1]) #f vector, where zeroth element is bedrock and final element is surface



##STEADYSTATE MODEL
#define vertical velocity with the step-model
for i in range(0, n_steady-1):
    if z_steady[i] > h + 48:
        w_steady[i] = -(lam/He) * (z_steady[i] - 48 - (h*(1 - fb)))
    if 48 < z_steady[i] < h + 48:
        w_steady[i] = -fb*lam*(z_steady[i] - 48)/He
    if 0 < z_steady[i] < 48:
        w_steady[i] = 0

#generate the coefficient matrix
for i in range(0, n_steady):

    if i == 0:
        coeffs_steady[i, i] = -1 #b1
        coeffs_steady[i, i+1] = 1 #c1
        funcs_steady[i] = -Q*delz/K #f1

    if i == n_steady-1:
        coeffs_steady[i, i] = 1 #bn
        coeffs_steady[i, i-1] = 0 #an
        funcs_steady[i] = Ts #fn

    elif i != 0 and i != n_steady-1:
        coeffs_steady[i, i] = -2
        coeffs_steady[i, i-1] = 1 + ((w_steady[i]*delz)/(2*kappa))
        coeffs_steady[i, i+1] = 1 - ((w_steady[i]*delz)/(2*kappa))



temps_steady = np.dot(np.linalg.inv(coeffs_steady),funcs_steady) #steady state temperature array

plt.figure()
plt.plot(temps_steady-273.15, z_steady) #minus for conversion to Celsius
plt.xlabel('Temperature (C)')
plt.ylabel('Height above bedrock (m)')
plt.title('Steady-state stepmodel')


satellite = np.loadtxt(r'C:\Users\elloi\Documents\Uni\Bachelor\Projekt\corrected.txt', delimiter=',') #importing satellite data


##NON STEADY STATE INITIALISATION
Lam = lam*917 #for use in density calculation
n = int(584/delz) #number of model runs
num_points = 403 #number of time steps in satellite data
g  = 9.81 #gravity

z = np.linspace(0, n-1, n) #as before
w = np.zeros([n])
coeffs = np.zeros([n,n])
funcs = np.zeros([n])

##VERTICAL VELOCITY

for i in range(0,n-1):
    if z[i] > h + 48:
        w[i] = -(lam/He) * ((z[i] - 48) - (h*(1 - fb)))
    if 48 < z[i] < h + 48:
        w[i] = -fb*lam*(z[i] - 48)/He
    if 0 < z[i] < 48: #if dead ice
        w[i] = 0
w[-1] = -lam


##DENSITY FUNCTION

d = z[::-1] #finds z reversed, for depths rather than heights
d_crit = 7.97 #critical depth
d_co = 57.02 #closeoff depth
rho_0 = 0.391084 #initial density
rho_crit = 0.497782 #critical density
rho_co = 0.820 #close-off density
lh = 0.42 #accumulation rate
R = 8.314 #gas constant
K0 = 11 * np.exp(-10160/(R*Ts)) #Arrhenius type numbers
K1 = 575 * np.exp(-21400/(R*Ts))
rho_ice = 0.917 #density of ice


rho = np.zeros([n]) #density array, as a function of depth

Z1 = np.zeros([n])#empty array for Z1 func
Z0 = np.zeros([n]) #empty array for Z0 func

for i in range(0,n):

    if d[i] == 0:
        rho[i] = (rho_0)

    if 0 < d[i] < d_crit: #critical density and above
        Z0[i] = np.exp((rho_ice*K0*d[i]) + np.log(rho_0/(rho_ice - rho_0)))
        rho[i] = ((rho_ice * Z0[i])/(1 + Z0[i]))

    if d_crit < d[i] < d_co: #between critical and closeoff density
        Z1[i] = np.exp( (rho_ice*K1*(d[i]-d_crit)*(lh**-0.5)) + np.log(0.55/(rho_ice - 0.55)) )
        rho[i] =((rho_ice * Z1[i])/(1 + Z1[i]))

    if d_co < d[i]:  #below close off depth, constant density
        rho[i] = rho_ice

##DENSITY DEPENDENT PARAMS

rho_z = rho[::-1] #finding rho as function of z

L = [] #overlying load
K_ice = [] #thermal conductivity of ice
c = [] #specific heat capacity using steady state guess
K_firn = [] #thermal  conductivity of firn
kappa_firn = [] #thermal diffusivity of firn
kappa_func = [] #d kappa / d rho

for i in range(0,n):
    L.append(z[i] * rho_z[i])

    K_ice.append(9.828 * np.exp(-0.0057 * temps_steady[i]))

    c.append(152.5 + (7.122 * temps_steady[i]))

    K_firn.append(K_ice[i] * ((rho[i] / rho_ice)**(1 - (rho[i] / (2 * rho_ice)))))

    kappa_firn.append(K_ice[i] / (c[i] * rho_ice * 1000) * ((rho[i] / rho_ice)**(1 - (rho[i] / (2 * rho_ice)))))

    kappa_func.append(kappa_firn[i] * ((1 / (rho_z[i] * 1000)) - \
        (1 / (2 * rho_ice * 1000))) * (1 + np.log((rho_z[i]) / (rho_ice))))


rho_func = [] #d rho / dz
for i in range(0,n-1):
    rho_func.append((rho_z[i+1]-rho_z[i])/delz)



##GENERATING PROFILE

T_list = [ [] for i in range(num_points+1) ]
T_list[0] = temps_steady #first element of T = steadystate guess

temps = [ [] for i in range(num_points+1) ] #list of profiles - each profile from a new model run
temps[0] = temps_steady #steady state guess as the first profile

for j in range(1,num_points): #looping over sat points

    for i in range(0,n): #looping over height steps

        if i == 0: #first row

            coeffs[i,i] = -1 #b1
            coeffs[i,i+1] = 1 #c1
            funcs[i] = -Q * delz/ K_ice[-1] #f1

        if i == n-1: #last row

            coeffs[i,i] = 1 #bn
            coeffs[i, i-1] = 0 #an
            funcs[i] = satellite[j] #fn

        elif i != 0 and i != n-1:

            D = [(T_list[j-1][i+1] - T_list[j-1][i-1])/2*delz \
                 for i in range(n-1)]

            rj = t/(4*delz) * (w[i] + 0.0057 * D[i] * kappa_firn[i] \
            - (rho_func[i]*((kappa_firn[i]/rho[i]) + kappa_func[i])))

            sj = (kappa_firn[i] * t)/(2*(delz**2))

            coeffs[i,i] = 1+(2*sj) #bj
            coeffs[i,i-1] = -rj - sj #aj
            coeffs[i,i+1] = rj - sj #cj

            funcs[i] = (sj - rj)*T_list[j-1][i+1] + \
            (1-(2*sj))*T_list[j-1][i] + (sj + \
            rj)*T_list[j-1][i-1] + \
            (L[i] * (Lam*g/(c[i]*(rho[i]*1000)**3)) * (rho_func[i]) )#fj


    T_list[j] = np.dot(np.linalg.inv(coeffs), funcs) #inverts the matrix
    temps[j] = T_list[j][:] - 273.15 #converts all to Celsius, for plotting


profile = temps[-4] #use profile corresponding to the month in which the ice core was drilled - May in this case
#as satellite data extends a few months after core is drilled


##PLOTTING

recap = np.loadtxt(r'C:\Users\elloi\Documents\Uni\Bachelor\Projekt\recap.txt', delimiter=',') #loading recap data

plt.figure()
plt.plot(profile, z, label='Satellite data model')
plt.plot(recap[:,1], H-recap[:,0]+48, label='RECAP data')
#add 48m to recap data, as the H defined above subtracts 48m for dead ice
plt.xlabel('Temperature (C)')
plt.ylabel('Height above bedrock (m)')
plt.legend()
plt.show()
