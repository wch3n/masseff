import numpy as np
from pymatgen.io.vasp.outputs import Vasprun
from pymatgen.io.vasp.inputs import Kpoints
from pymatgen.io.vasp.sets import MPNonSCFSet
from pymatgen.electronic_structure.bandstructure import Spin
from os.path import join

class Mass_eff():
    '''Calculating effective mass via finite difference method.
    '''
    def __init__(self, h):
        ''' h: step size
        '''
        self.h = h

    def stencil(self):
        '''hardcoded 5-point stencil in cartesian coordinates
        [x, y, z];
        h: step size in 1/Angstrom
        '''
        stcl = np.array([[0.0, 0.0, 0.0], [-2.0, 0.0, 0.0], [-1.0, 0.0, 0.0], 
                         [1.0, 0.0, 0.0], [ 2.0, 0.0, 0.0], [ 0.0,-2.0, 0.0], [ 0.0,-1.0, 0.0], 
                         [0.0, 1.0, 0.0], [ 0.0, 2.0, 0.0], [ 0.0, 0.0,-2.0], [ 0.0, 0.0,-1.0], 
                         [0.0, 0.0, 1.0], [ 0.0, 0.0, 2.0], [-2.0,-2.0, 0.0], [-1.0,-2.0, 0.0],
                         [1.0,-2.0, 0.0], [ 2.0,-2.0, 0.0], [-2.0,-1.0, 0.0], [-1.0,-1.0, 0.0], 
                         [1.0,-1.0, 0.0], [ 2.0,-1.0, 0.0], [-2.0, 1.0, 0.0], [-1.0, 1.0, 0.0], 
                         [1.0, 1.0, 0.0], [ 2.0, 1.0, 0.0], [-2.0, 2.0, 0.0], [-1.0, 2.0, 0.0], 
                         [1.0, 2.0, 0.0], [ 2.0, 2.0, 0.0], [-2.0, 0.0,-2.0], [-1.0, 0.0,-2.0], 
                         [1.0, 0.0,-2.0], [ 2.0, 0.0,-2.0], [-2.0, 0.0,-1.0], [-1.0, 0.0,-1.0],
                         [1.0, 0.0,-1.0], [ 2.0, 0.0,-1.0], [-2.0, 0.0, 1.0], [-1.0, 0.0, 1.0], 
                         [1.0, 0.0, 1.0], [ 2.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 2.0], 
                         [1.0, 0.0, 2.0], [ 2.0, 0.0, 2.0], [ 0.0,-2.0,-2.0], [ 0.0,-1.0,-2.0], 
                         [0.0, 1.0,-2.0], [ 0.0, 2.0,-2.0], [ 0.0,-2.0,-1.0], [ 0.0,-1.0,-1.0], 
                         [0.0, 1.0,-1.0], [ 0.0, 2.0,-1.0], [ 0.0,-2.0, 1.0], [ 0.0,-1.0, 1.0],
                         [0.0, 1.0, 1.0], [ 0.0, 2.0, 1.0], [ 0.0,-2.0, 2.0], [ 0.0,-1.0, 2.0], 
                         [0.0, 1.0, 2.0], [ 0.0, 2.0, 2.0]])

        return self.h*stcl

    def kpoints(self, recp_vec, k0_recp):
        '''List of kpoints for the nscf calculation
        in reciprocal lattice vectors; 
        ''' 
        recp_vec_inv = np.linalg.inv(recp_vec)
        k0_cart = np.dot(k0_recp, recp_vec)
        kl_cart = k0_cart + self.stencil()
        kl_recp = np.dot(kl_cart, recp_vec_inv)

        return kl_recp

    def write_vasp(self, k0_recp, scfdir, destdir):
        '''Generate VASP nscf input files
        '''
        r = Vasprun(join(scfdir, 'vasprun.xml'))
        recp_vec = r.lattice_rec.matrix
        kl_recp = self.kpoints(recp_vec, k0_recp)
        nscf = MPNonSCFSet.from_prev_calc(scfdir)
        nscf.write_input(destdir)
        kpt = Kpoints(style=Kpoints.supported_modes.Reciprocal, kpts=kl_recp,
              num_kpts=len(kl_recp), kpts_weights=[1.0]*len(kl_recp),
              comment="5-point stencil h:{}".format(self.h))
        kpt.write_file(join(destdir, 'KPOINTS'))

    def calc_mass(self, type='hole', destdir='./'):
        '''Calculate the effective mass (in m_0);
        e is a length-61 array (in eV).
        '''
        h = self.h
        e = self.read_eigenvalues(type, destdir)

        m_t = np.zeros((3,3)) # 3x3 tensor
        m_t[0,0] = (-(e[1]+e[4])  + 16.0*(e[2]+e[3])   - 30.0*e[0])/(12.0*h**2)
        m_t[1,1] = (-(e[5]+e[8])  + 16.0*(e[6]+e[7])   - 30.0*e[0])/(12.0*h**2)
        m_t[2,2] = (-(e[9]+e[12]) + 16.0*(e[10]+e[11]) - 30.0*e[0])/(12.0*h**2)
        m_t[0,1] = (-63.0*(e[15]+e[20]+e[21]+e[26]) + 63.0*(e[14]+e[17]+e[27]+e[24]) + 
                     44.0*(e[16]+e[25]-e[13]-e[28]) + 74.0*(e[18]+e[23]-e[19]-e[22]))/(600.0*h**2)
        m_t[0,2] = (-63.0*(e[31]+e[36]+e[37]+e[42]) + 63.0*(e[30]+e[33]+e[43]+e[40]) +
                     44.0*(e[32]+e[41]-e[29]-e[44]) + 74.0*(e[34]+e[39]-e[35]-e[38]))/(600.0*h**2)
        m_t[1,2] = (-63.0*(e[47]+e[52]+e[53]+e[58]) + 63.0*(e[46]+e[49]+e[59]+e[56]) +
                     44.0*(e[48]+e[57]-e[45]-e[60]) + 74.0*(e[50]+e[55]-e[51]-e[54]))/(600.0*h**2)

        m_t[1,0] = m_t[0,1]
        m_t[2,0] = m_t[0,2]
        m_t[2,1] = m_t[1,2] 

        w, v = np.linalg.eig(m_t)
        w /= 27.2116/1.8897**2 # -> atomic units
        m = 1.0/w
        m_cond = 3.0/np.sum(w)

        return m, m_cond

    @staticmethod
    def read_eigenvalues(type, destdir):
        r = Vasprun(join(destdir, 'vasprun.xml'))
        bs = r.get_band_structure()
        if type == 'hole':
            ind = max(bs.get_vbm()['band_index'][Spin.up])
            eig = r.eigenvalues[Spin.up][:,ind][:,0]
        else:
            ind = min(bs.get_cbm()['band_index'][Spin.up]) 
            eig = r.eigenvalues[Spin.up][:,ind][:,0]
        
        return eig

if __name__ == '__main__':
    # example
    scfdir = "../"
    destdir = "./"
    h = 0.01            # stepsize in 1/angstrom
    k0_recp = [0, 0, 0] # band extremum
    em = Mass_eff(h)    # instantiate 
    em.write_vasp(k0_recp, scfdir, destdir)  # generate the k-point list for nscf calculation
    #
    # do a vasp nscf calculation
    #
    m, m_cond = em.calc_mass('hole')  # determine the hole effective mass
    print(h, m, m_cond)
