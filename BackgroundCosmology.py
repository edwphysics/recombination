import numpy as np
from   matplotlib import pyplot as plt
from   scipy.interpolate import CubicSpline
import scipy.integrate as integrate

from   Global import const

class BackgroundCosmology:
  """
  This is a class for the cosmology at the background level.
  It holds cosmological parameters and functions relevant for the background.
  
  Input Parameters: 
    h           (float): The little Hubble parameter h in H0 = 100h km/s/Mpc
    OmegaB      (float): Baryonic matter density parameter at z = 0
    OmegaCDM    (float): Cold dark matter density parameter at z = 0
    OmegaK      (float,optional): Curvative density parameter at z = 0
    name        (float,optional): A name for describing the cosmology
    TCMB        (float,optional): The temperature of the CMB today in Kelvin. Fiducial value is 2.725K
    Neff        (float,optional): The effective number of relativistic neutrinos

  Attributes:    
    OmegaR      (float): Radiation matter density parameter at z = 0
    OmegaNu     (float): Massless neutrino density parameter at z = 0
    OmegaM      (float): Total matter (CDM+b+mnu) density parameter at z = 0
    OmegaK      (float): Curvature density parameter at z = 0
  
  Functions:
    H_of_x               (float->float) : Hubble parameter as function of x=log(a)     
  """
  
  # Settings for integration and splines of eta
  x_start = np.log(1e-8)
  x_end = np.log(1.0)
  n_pts_splines = 1000

  def __init__(self, h = 0.7, OmegaB = 0.046, OmegaCDM = 0.224, OmegaK = 0.0, 
      name = "FiducialCosmology", TCMB_in_K = 2.725, Neff = 3.046):
    self.OmegaB      = OmegaB
    self.OmegaCDM    = OmegaCDM
    self.OmegaK      = OmegaK
    self.h           = h
    self.H0          = const.H0_over_h * h
    self.name        = name
    self.TCMB        = TCMB_in_K * const.K
    self.Neff        = Neff
 
    # Set the constants
    self.OmegaR      = 2*(np.pi**2/30)*(const.k_b**4*self.TCMB**4/(const.hbar**3*const.c**5))*(8*np.pi*const.G/(3*self.H0**2)) # Radiation
    self.OmegaNu     = self.Neff*(7/8)*(4/11)**(4/3)*self.OmegaR # Neutrino radiation
    self.OmegaLambda = 1.0 - (self.OmegaK + self.OmegaB + self.OmegaCDM + self.OmegaR + self.OmegaNu) # Dark energy (from Sum Omega_i = 1)
  
  #=========================================================================
  # Methods availiable after solving
  #=========================================================================
  

  #Hubble Parameter
  def H_of_x(self,x):
    return self.H0*np.sqrt((self.OmegaB + self.OmegaCDM)*np.exp(-3*x) + (self.OmegaR + self.OmegaNu)*np.exp(-4*x) + self.OmegaK*np.exp(-2*x) + self.OmegaLambda)

  #=========================================================================
  #=========================================================================
  #=========================================================================
  
  def info(self):
    """
    Print some useful info about the class
    """
    print("")
    print("Background Cosmology:")
    print("OmegaB:        %8.7f" % self.OmegaB)
    print("OmegaCDM:      %8.7f" % self.OmegaCDM)
    print("OmegaLambda:   %8.7f" % self.OmegaLambda)
    print("OmegaR:        %8.7e" % self.OmegaR)
    print("OmegaNu:       %8.7e" % self.OmegaNu)
    print("OmegaK:        %8.7f" % self.OmegaK)
    print("TCMB (K):      %8.7f" % (self.TCMB / const.K))
    print("h:             %8.7f" % self.h)
    print("H0:            %8.7e" % self.H0)
    print("H0 (km/s/Mpc): %8.7f" % (self.H0 / (const.km / const.s / const.Mpc)))
    print("Neff:          %8.7f" % self.Neff)
  
  def plot(self):
    """
    Plot some useful quantities
    """

    npts = 2000
    xarr = np.linspace(self.x_start, self.x_end, num = npts)
    fac  = [self.H_of_x(xarr[i]) for i in range(npts)]
    plt.yscale('log')
    plt.title('Hubble Parameter')
    plt.plot(xarr, fac, label = 'H(x)')
    plt.legend()
    plt.show()
  
  #=========================================================================
  #=========================================================================
  #=========================================================================