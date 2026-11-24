import json
from pathlib import Path
import numpy as np
from scipy.io import loadmat, savemat

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'benchmark_audit'
OUT.mkdir(exist_ok=True)


def solve(n, dt, method='etd'):
    x = np.linspace(-1, 1, n, endpoint=False)
    k = np.fft.fftfreq(n, 2/n)*2*np.pi
    k[n//2] = 0
    linear = 1j*.0025*k**3
    pad = 3*n//2
    def nonlinear(v):
        vp = np.zeros(pad, complex)
        vp[:n//2] = v[:n//2]
        vp[-n//2+1:] = v[-n//2+1:]
        up = np.fft.ifft(vp*pad/n).real
        fp = np.fft.fft(up**2)
        f = np.zeros(n, complex)
        f[:n//2] = fp[:n//2]
        f[-n//2+1:] = fp[-n//2+1:]
        return -.5j*k*f*n/pad
    if method == 'etd':
        r = np.exp(2j*np.pi*(np.arange(64)+.5)/64)
        z = dt*linear[:, None] + r
        q = dt*np.mean(np.expm1(z/2)/z, axis=1)
        f1 = dt*np.mean((-4-z+np.exp(z)*(4-3*z+z*z))/z**3, axis=1)
        f2 = dt*np.mean((2+z+np.exp(z)*(-2+z))/z**3, axis=1)
        f3 = dt*np.mean((-4-3*z-z*z+np.exp(z)*(4-z))/z**3, axis=1)
        e, e2 = np.exp(dt*linear), np.exp(dt*linear/2)
        def step(v):
            nv = nonlinear(v)
            a = e2*v+q*nv
            na = nonlinear(a)
            b = e2*v+q*na
            nb = nonlinear(b)
            c = e2*a+q*(2*nb-nv)
            return e*v+f1*nv+2*f2*(na+nb)+f3*nonlinear(c)
    else:
        def rhs(v):
            return linear*v+nonlinear(v)
        def step(v):
            a = rhs(v)
            b = rhs(v+dt*a/2)
            c = rhs(v+dt*b/2)
            d = rhs(v+dt*c)
            return v+dt*(a+2*b+2*c+d)/6
    v = np.fft.fft(np.cos(np.pi*x))
    nt = round(1/dt)
    stride = round(.01/dt)
    saved = []
    for i in range(nt+1):
        if i % stride == 0:
            saved.append(np.fft.ifft(v).real)
        if i < nt:
            v = step(v)
    u = np.array(saved).T
    assert np.isfinite(u).all()
    print(n, dt, method, 'complete', flush=True)
    return x, np.linspace(0, 1, 101), u


if __name__ == '__main__':
    old = loadmat(ROOT/'data/KdV_reference_data_lambda2_0.0025.mat')
    x, t, a = solve(256, 5e-5)
    _, _, b = solve(256, 2.5e-5)
    _, _, c = solve(512, 2.5e-5)
    _, _, d = solve(256, 1e-5, 'rk4')
    rel = lambda v, w: float(np.linalg.norm(v-w)/np.linalg.norm(w)*100)
    report = {
        'old_reference_vs_corrected_percent': rel(old['usol'], b),
        'old_reference_max_difference': float(np.max(np.abs(old['usol']-b))),
        'dt_halving_difference_percent': rel(a, b),
        'space_doubling_difference_percent': rel(b, c[::2]),
        'independent_RK4_difference_percent': rel(d, b),
        'mass_drift': float(np.max(np.abs(b.mean(axis=0)-b[:,0].mean()))),
        'quadratic_invariant_relative_drift': float(np.max(np.abs(np.mean(b*b,axis=0)/np.mean(b[:,0]**2)-1))),
        'coefficient_rule': 'Full complex contour; no real-part projection of imaginary-linear-operator coefficients',
        'source': 'https://people.maths.ox.ac.uk/trefethen/publication/PDF/2005_111.pdf'
    }
    savemat(OUT/'KdV_crosschecked_reference.mat', {'x':x[:,None], 't':t[:,None], 'usol':b})
    (OUT/'KdVreference_crosscheck.json').write_text(json.dumps(report,indent=2), encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)
