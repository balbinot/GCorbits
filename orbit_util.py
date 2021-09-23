import vaex
from astropy import coordinates as coord
from astropy import units as u
import numpy as np

# Gala
import gala.dynamics as gd
import gala.integrate as gi
import gala.potential as gp
from gala.units import galactic
from tqdm import tqdm, trange
from joblib import Parallel, delayed
import multiprocessing

k = 4.7404705
potential = gp.MilkyWayPotential()

def _integrate(w, ids=None, dt=1, n_steps=500):
    orbit = potential.integrate_orbit(w, dt=-dt*u.Myr, n_steps=n_steps, Integrator=gi.DOPRI853Integrator)
    forbit = potential.integrate_orbit(w, dt=dt*u.Myr, n_steps=n_steps, Integrator=gi.DOPRI853Integrator)
    for n, orb in enumerate([orbit, forbit]):
        time = orb.t.value
        x = orb.x.value
        y = orb.y.value
        z = orb.z.value
        vx = orb.v_x.to(u.km/u.s).value
        vy = orb.v_y.to(u.km/u.s).value
        vz = orb.v_z.to(u.km/u.s).value
        E = orb.energy().to(u.km**2/u.s**2).value
        L = orb.angular_momentum()
        Lx, Ly, Lz = L[0], L[1], L[2]
        Lx = Lx.to(u.kpc*u.km/u.s).value
        Ly = Ly.to(u.kpc*u.km/u.s).value
        Lz = Lz.to(u.kpc*u.km/u.s).value
        ecc = np.repeat(orb.eccentricity(), len(x))
        ID = np.repeat(ids, len(x))
        _orb = orb.to_coord_frame(coord.ICRS)
        _ra = _orb.ra.value
        _dec = _orb.dec.value
        _pmra = _orb.pm_ra_cosdec.value
        _pmdec = _orb.pm_dec.value
        _Vlos = _orb.radial_velocity.value
        _hdist = _orb.distance.value
        out = np.array([time, x, y, z, vx, vy, vz, _ra, _dec, _pmra, _pmdec, _Vlos, _hdist, ecc, E, Lz, ID]).T
        try:
            OUT = np.r_[out[::-1], OUT]
        except:
            OUT = out
    return OUT

def sample_orbits(df, id_column='Cluster', cols=['RA', 'DEC', 'Rsun', 'ERsun',
                                                 'pmra', 'e_pmra', 'pmdec',
                                                 'e_pmdec', 'RV', 'ERV'], dt=1,
                  n_steps=500, num_cores=32, nsamp=100):
    """
    dt:        time step for output in Myr
    n_steps:   number of steps
    num_cores: number of cores to use in parallel

    Returns: vaex DF

    """

    cname =  np.repeat(df[id_column].values, nsamp)
    ra =    np.repeat(df[cols[0]].values, nsamp)
    dec =   np.repeat(df[cols[1]].values, nsamp)
    pmra =  np.random.normal(df[cols[4]].values, df[cols[5]].values, (nsamp, len(df))).T.flatten()
    pmdec =  np.random.normal(df[cols[6]].values, df[cols[7]].values, (nsamp, len(df))).T.flatten()
    vlos =  np.random.normal(df[cols[8]].values, df[cols[9]].values, (nsamp, len(df))).T.flatten()
    dist =  np.random.normal(df[cols[2]].values, df[cols[3]].values, (nsamp, len(df))).T.flatten()

    cooE = coord.SkyCoord(ra=ra*u.deg,
                          dec=dec*u.deg,
                          distance=dist*u.kpc,
                          radial_velocity=vlos*u.km/u.s,
                          pm_ra_cosdec=pmra*u.mas/u.yr,
                          pm_dec=pmdec*u.mas/u.yr)

    samples = vaex.from_arrays(ra=ra, dec=dec, pmra=pmra, pmdec=pmdec, vlos=vlos, distance=dist)

    ## If running a lot (>1e6) orbits, running the velocities inside vaex may be better.
    #df.add_variable('vlsr',232.8);  vlsr = 232.8
    #df.add_variable('R0',8.20)   ;  R0 = 8.2
    #df.add_variable('_U',11.1)   ;  _U = 11.1
    #df.add_variable('_V',12.24)  ;  _V = 12.24
    #df.add_variable('_W',7.25)   ;  _W = 7.25

    cooG = cooE.transform_to(coord.Galactic)
    cooGc = cooE.transform_to(coord.Galactocentric(galcen_distance=8.2*u.kpc,
                                                   galcen_v_sun=coord.CartesianDifferential((11.1,
                                                                                             232.8+12.24,
                                                                                             7.25)*u.km/u.s)))
    w0 = gd.PhaseSpacePosition(cooGc.data)
    
    ## This uses a LOT of memory
    results = Parallel(n_jobs=num_cores)(delayed(_integrate)(i, ids=cname[n], dt=dt, n_steps=n_steps) for n,i in tqdm(enumerate(w0)))
    all = np.vstack(results)
    odf = vaex.from_arrays(ID = all[:,16].astype(np.str),
                           time=all[:,0].astype(np.float32),
                           x=all[:,1].astype(np.float32),
                           y=all[:,2].astype(np.float32),
                           z=all[:,3].astype(np.float32),
                           vx=all[:,4].astype(np.float32),
                           vy=all[:,5].astype(np.float32),
                           vz=all[:,6].astype(np.float32),
                           ra=all[:,7].astype(np.float32),
                           dec=all[:,8].astype(np.float32),
                           pmra=all[:,9].astype(np.float32),
                           pmdec=all[:,10].astype(np.float32),
                           Vlos=all[:,11].astype(np.float32),
                           dist = all[:,12].astype(np.float32),
                           ecc = all[:,13].astype(np.float32),
                           E = all[:,14].astype(np.float32),
                           Lz = all[:,15].astype(np.float32),
                           )
    return (odf, samples)
