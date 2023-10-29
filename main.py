import numpy as np
from skyfield.api import load, EarthSatellite
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

TLE = """ISS (ZARYA)             
1 25544U 98067A   23160.39424544  .00014660  00000+0  25863-3 0  9990
2 25544  51.6428   7.1636 0005453  65.5181 349.5815 15.50617471400583"""

todeg = 180 / np.pi
torad = np.pi / 180

phi = 1.347580 * torad
f = 0.003353
h = 30 / 1000
re = 6378
mu = 389600

theta1 = (12 + 3/60 + 3.2/3600) * 15 * torad
theta2 = (12 + 3/60 + 56.7/3600) * 15 * torad
theta3 = (12 + 4/60 + 42.6/3600) * 15 * torad

ra1 = (7 + 16/60 + 55.04/3600) * 15 * torad
dec1 = (-1 - 30/60 - 51.2/3600) * torad
ra2 = (7 + 45/60 + 46.35/3600) * 15 * torad
dec2 = (-23 - 11/60 - 35.5/3600) * torad
ra3 = (8 + 17/60 + 38.14/3600) * 15 * torad
dec3 = (-43 - 45/60 - 43.4/3600) * torad

t1 = 0
t2 = 53.35
t3 = 99.4

def posroot(roots):
    # Extract positive real roots
    posroots = roots[np.logical_and(roots > 0, np.isreal(roots))].real
    
    # Check if any positive real roots exist
    npositive = len(posroots)
    if npositive == 0:
        raise ValueError("There are no positive real roots.")
    
    # If there is only one positive real root, return it
    if npositive == 1:
        return posroots[0]
    
    # If there are multiple positive real roots, prompt the user to select one
    print("There are multiple positive real roots. Please select one:")
    for i, root in enumerate(posroots):
        print(f"Root #{i+1}: {root}")
    
    while True:
        choice = input("Enter the root number to use: ")
        try:
            choice = int(choice)
            if 1 <= choice <= npositive:
                return posroots[choice-1]
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a valid root number.")

# calculating topocentric position vectors
def position_vector(theta):
    a = 1 - (2 * f - f**2) * (np.sin(phi))**2
    b = np.sqrt(a)
    c = re / b
    D = c + h
    e = c * (1 - f)**2
    F = e + h

    Rx = D * np.cos(phi) * np.cos(theta)
    Ry = D * np.cos(phi) * np.sin(theta)
    Rz = F * np.sin(phi)
    R = np.array([Rx, Ry, Rz])

    return R

R1 = position_vector(theta1)
R2 = position_vector(theta2)
R3 = position_vector(theta3)

#ISS direction unit vector
def rhohat(ra, dec):
    rx = np.cos(dec) * np.cos(ra)
    ry = np.cos(dec) * np.sin(ra)
    rz = np.sin(dec)
    rhohat = np.array([rx, ry, rz])
    return rhohat

rhohat1 = rhohat(ra1, dec1)
rhohat2 = rhohat(ra2, dec2)
rhohat3 = rhohat(ra3, dec3)

#position vectors of ISS
tau1 = t1 - t2
tau3 = t3 - t2
tau = tau3 - tau1

p1 = np.cross(rhohat2, rhohat3)
p2 = np.cross(rhohat1, rhohat3)
p3 = np.cross(rhohat1, rhohat2)

D0 = np.dot(rhohat1, p1)
D11 = np.dot(R1, p1)
D21 = np.dot(R2, p1)
D31 = np.dot(R3, p1)
D12 = np.dot(R1, p2)
D22 = np.dot(R2, p2)
D32 = np.dot(R3, p2)
D13 = np.dot(R1, p3)
D23 = np.dot(R2, p3)
D33 = np.dot(R3, p3)

A = 1 / D0 * (-1 * D12 * tau3/tau + D22 + D32 * tau1/tau)
B = 1 / 6 / D0 * (D12 * (tau3**2-tau*2) * tau3/tau + D32 * (tau**2 - tau1**2) * tau1 / tau)

E = np.dot(R2, rhohat2)

a = -1 * (A**2 + 2 * A * E + np.dot(R2, R2))
b = -2 * mu * B * (A + E)
c = -1 * mu**2 * B**2

r2 = posroot(np.roots([1, 0, a, 0, 0, b, 0, 0, c]))

rho1 = 1 / D0 * ((6 * (D31 * tau1 / tau3 + D21 * tau / tau3) * r2**3 + mu * D31 * (tau**2 - tau1**2) * tau1 / tau3)/(6 * r2**3 + mu * (tau**2 - tau3**2))- D11)
rho2 = A + mu * B / r2**3
rho3 = 1 / D0 * ((6 * (D13 * tau3 / tau1 - D23 * tau / tau1) * r2**3 + mu * D13 * (tau**2 - tau3**2) * tau3 / tau1)/(6 * r2**3 + mu * (tau**2 - tau1**1))- D33)

r1vec = R1 + rho1 * rhohat1
r2vec = R2 + rho2 * rhohat2
r3vec = R3 + rho3 * rhohat3
print(r1vec)
print(r2vec)
print(r3vec)

#calculating lagrange coefficients
f1 = 1 - 0.5 * mu / r2**3 * tau1**2
f3 = 1 - 0.5 * mu / r2**3 * tau3**2
g1 = tau1 - 1 / 6 * mu / r2**3 * tau1**3
g3 = tau3 - 1 / 6 * mu / r2**3 * tau3**3

v2vec = (-f3*r1vec + f1*r3vec) / (f1*g3 - f3*g1)
print(v2vec)
print(np.sqrt(np.dot(v2vec, v2vec)))

#plot the ground track
ts = load.timescale(builtin=True)
name, L1, L2 = TLE.splitlines()

sat = EarthSatellite(L1, L2)

minutes = np.arange(0, 200, 0.1) # about two orbits
times   = ts.utc(2019, 7, 23, 0, minutes)

geocentric = sat.at(times)
subsat = geocentric.subpoint()

fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

ax.stock_img()

plt.scatter(subsat.longitude.degrees, subsat.latitude.degrees, transform=ccrs.PlateCarree(),
            color='red')
plt.show()
