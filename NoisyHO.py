import numpy as np
#
# Stationary solution of the harmonic oscillator, with damping and noise
# Parameters
dt=1.0e-3
l=2.5
sqrdt=np.sqrt(dt)
amp=1.
niter=int(1.0e10)
gamma=0.1
mu, sigma = 0, 2.0*gamma
#
# Initial condition
xiter=0.0
viter=1.0
xvmax=0    # this is do identify the phase space window
#
# Let's do a first run
for i in np.arange(1000000):
    xsave=xiter
    xiter=xiter+viter*dt
    viter=viter-(gamma*viter+xsave)*dt+np.random.normal(mu, sigma)*sqrdt
    if np.max([xiter,viter]) > xvmax: xvmax=np.max([xiter,viter])
# first identify the size of the phase space window we want to scan
l=xvmax*1.1
# 
dx=l/100.
nx=2*int(l/dx)
nv=2*int(l/dx)
Dx=np.zeros((nx,nv))
Dv=np.zeros((nx,nv))
Dxx=np.zeros((nx,nv))
Dxv=np.zeros((nx,nv))
Dvv=np.zeros((nx,nv))
Count=np.ones((nx,nv)).astype(int)
#
xold=xiter
vold=viter
#
print('Start iteration',flush=True)
# Now we start to iterate and fill the drift and Diffusion coefficient matrices
i=1
while i < niter:
    i+=1
# Those are the "derivatives"
    xnew=xold+vold*dt
    vnew=vold-(gamma*vold+xold)*dt+amp*np.random.normal(mu, sigma)*sqrdt
    xd=(xnew-xold)/dt
    vd=(vnew-vold)/dt
    xold=xnew
    vold=vnew
#
    xindex=(np.floor((xnew+l)/dx)).astype(int)
    vindex=(np.floor((vnew+l)/dx)).astype(int)
    #print(xindex,vindex,end='\r')
    if xindex<nx and vindex<nv:
        Count[xindex,vindex]+=1
        Dx[xindex,vindex]+=xd
        Dv[xindex,vindex]+=vd
        Dxx[xindex,vindex]+=xd*xd
        Dxv[xindex,vindex]+=xd*vd
        Dvv[xindex,vindex]+=vd*vd
    if i%(niter//10000)==0:
        print(i,flush=True)
        Dx=Dx/Count
        Dv=Dv/Count
        Dxx=Dxx/Count
        Dxv=Dxv/Count
        Dvv=Dvv/Count
        np.savez('D.npz', 'wb', Dx=Dx,Dv=Dv,Dxx=Dxx,Dxv=Dxv,Dvv=Dvv,xnew=xnew,vnew=vnew,step=i)
#
# Average of the D's by dividing with number of events for every phase space cell
Dx=Dx/Count
Dv=Dv/Count
Dxx=Dxx/Count
Dxv=Dxv/Count
Dvv=Dvv/Count
# Save the D's
np.savez('D.npz', 'wb', Dx=Dx,Dv=Dv,Dxx=Dxx,Dxv=Dxv,Dvv=Dvv,xnew=xnew,vnew=vnew)
