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

    def kpoints(self, rlv, k0_rlv):
        '''List of kpoints for the nscf calculation
        in reciprocal lattice vectors; 
        a: scaling factor found in POSCAR
        ''' 
        rlv_inv = np.linalg.inv(rlv)
        k0_cart = np.dot(k0_rlv, rlv)
        kl_cart = k0_cart + self.stencil()
        kl_rlv = np.dot(kl_cart, rlv_inv)
        return kl_rlv

    def write_vasp(self, k0_rlv, scfdir, destdir):
        '''Generate VASP nscf input files
        '''
        r = Vasprun(join(scfdir, 'vasprun.xml'))
        rlv = r.get_band_structure().lattice_rec.matrix

        is_k_spin = True
        try:
            len(k0_rlv[0])
        except:
            is_k_spin = False
            k0_up = k0_rlv
            k0_dn = None
    
        if is_k_spin:
            k0_up, k0_dn = k0_rlv[0], k0_rlv[1]
         
        list_k = self.kpoints(rlv, k0_up)
        if r.is_spin and is_k_spin:
           list_k = np.concatenate((list_k, self.kpoints(rlv, k0_dn))) 
        nscf = MPNonSCFSet.from_prev_calc(scfdir, user_incar_settings=
                                          {"NCORE": 4, "EDIFF":"1E-7"})
        nscf.write_input(destdir)
        mode = Kpoints.supported_modes.Reciprocal
        kpt = Kpoints(style=mode, kpts=list_k, num_kpts=len(list_k), 
                      kpts_weights=[1.0]*len(list_k),
                      comment="5-point stencil h:{}".format(self.h))
        kpt.write_file(join(destdir, 'KPOINTS'))

    def calc_mass(self, typ='hole', band_ind=-1, destdir='./'):
        '''Calculate the effective mass (in m_0);
        e is a length-61 array (in eV).
        output: eigenvalues and eigenvectors (in cartesian and rlv).
        '''
        h = self.h
        self.band_ind = band_ind
        r = Vasprun(join(destdir, 'vasprun.xml'))
        eig = self.read_eigenvalues(r, typ, destdir)

        is_k_spin = eig[Spin.up.name].size == 122
        
        mass = {}; v={}
        if not is_k_spin:
            for spin in eig.keys():
                mass[spin], v[spin] = self.mass_tensor(eig[spin], self.h)
        else:
            mass[Spin.up.name], v[Spin.up.name] = self.mass_tensor(eig[Spin.up.name][:61], self.h)
            try:
                mass[Spin.down.name], v[Spin.down.name] = self.mass_tensor(eig[Spin.down.name][61:], self.h)
            except:
                pass
        
        rlv_inv = np.linalg.inv(r.get_band_structure().lattice_rec.matrix)
        v_rlv = {}
        for i in v.keys():
            v_rlv[i] = np.dot(rlv_inv.T, v[i])
        
        return mass, v, v_rlv
    
    @staticmethod
    def mass_tensor(e,h):
        m_t = np.zeros((3,3)) # 3x3 tensor
        m_t[0,0] = (-(e[1]+e[4])  + 16*(e[2]+e[3])   - 30*e[0])/(12*h**2)
        m_t[1,1] = (-(e[5]+e[8])  + 16*(e[6]+e[7])   - 30*e[0])/(12*h**2)
        m_t[2,2] = (-(e[9]+e[12]) + 16*(e[10]+e[11]) - 30*e[0])/(12*h**2)
        m_t[0,1] = (-63*(e[15]+e[20]+e[21]+e[26]) + 63*(e[14]+e[17]+e[27]+e[24]) + 
                     44*(e[16]+e[25]-e[13]-e[28]) + 74*(e[18]+e[23]-e[19]-e[22]))/(600*h**2)
        m_t[0,2] = (-63*(e[31]+e[36]+e[37]+e[42]) + 63*(e[30]+e[33]+e[43]+e[40]) +
                     44*(e[32]+e[41]-e[29]-e[44]) + 74*(e[34]+e[39]-e[35]-e[38]))/(600*h**2)
        m_t[1,2] = (-63*(e[47]+e[52]+e[53]+e[58]) + 63*(e[46]+e[49]+e[59]+e[56]) +
                     44*(e[48]+e[57]-e[45]-e[60]) + 74*(e[50]+e[55]-e[51]-e[54]))/(600*h**2)
        m_t[1,0] = m_t[0,1]
        m_t[2,0] = m_t[0,2]
        m_t[2,1] = m_t[1,2] 

        w, v = np.linalg.eig(m_t)
        w /= 27.2114*0.52917721067**2 # -> atomic units
        m = 1.0/w

        return m, v

    def read_eigenvalues(self, r, typ, destdir):
        eig = {}
        if r.is_spin:
            spins = [Spin.up, Spin.down]
        else:
            spins = [Spin.up]

        for spin in spins:
            if self.band_ind == -1:
                vbi, cbi = self.extremum_band_index(r, spin)
                if typ == 'hole':
                    eig[spin.name] = r.eigenvalues[spin][:,vbi,0]
                else:
                    eig[spin.name] = r.eigenvalues[spin][:,cbi,0]
            else: #band index specified manually (starts from 1)
                eig[spin.name] = r.eigenvalues[spin][:, self.band_ind-1,0]
        
        return eig

    @staticmethod
    def extremum_band_index(r, spin):
        bs = r.eigenvalues[spin]
        ind = max(np.argwhere(bs[0][:,1] > 0))        
        return ind, ind+1
