"""Finite architecture controls. Existing studies and budgets stay frozen."""
import math
from pathlib import Path

SEEDS = (42, 43, 44)
TASKS = (('forward', 'Burgers'), ('forward', 'KdV'), ('forward', 'NS_periodic'),
         ('inverse', 'KdV'), ('inverse', 'NS_periodic'))
HIDDEN = ('Fourier', 'RBF', 'Wavelet', 'MixedFixed', 'MixedAdaptive')


def jobs():
    result = []
    for mode, case in TASKS:
        designs = [('legacy_bridge', 'LegacyMLP', 'original'),
                   ('legacy_bridge', 'LegacyHybrid', 'original'),
                   ('fourier_hidden', 'Fourier', 'Fourier')]
        designs += [('hidden_basis', source, hidden) for source in ('Adaptive', 'Uniform') for hidden in HIDDEN]
        for condition, source, hidden in designs:
            for seed in SEEDS:
                variant = source if condition == 'legacy_bridge' else source + '_H' + hidden
                key = '%s_%s_%s_seed%s' % (mode, case, variant, seed)
                result.append(dict(id=key, condition=condition, mode=mode, input_variant=source,
                                   hidden_basis=hidden, job=dict(case=case, variant=variant, seed=seed)))
    assert len(result) == len({s['id'] for s in result}) == 195
    return result


def hidden_values(x, family, terms):
    import torch
    if family == 'Chebyshev':
        values = [torch.ones_like(x), x]
        for _ in range(2, terms):
            values.append(2*x*values[-1]-values[-2])
        return torch.stack(values[:terms], -1)
    if family == 'Fourier':
        freq = torch.arange(1, terms//2+1, device=x.device, dtype=x.dtype)
        phase = math.pi*x.unsqueeze(-1)*freq
        return torch.cat([phase.cos(), phase.sin()], -1)
    centers = torch.linspace(-1, 1, terms, device=x.device, dtype=x.dtype)
    r = (x.unsqueeze(-1)-centers)/(2/math.sqrt(terms))
    gaussian = torch.exp(-.5*r.square())
    if family == 'RBF':
        return gaussian
    if family == 'Wavelet':
        return (1-r.square())*gaussian
    raise ValueError(family)


def hidden_layer(old, family):
    import torch
    from torch import nn

    class HiddenBasis(nn.Module):
        def __init__(self):
            super().__init__()
            # Preserve every coefficient and all other tensors from the paired model.
            self.coeff = nn.Parameter(old.coeff.detach().clone())
            self.families = ('Chebyshev', 'Fourier', 'RBF', 'Wavelet') if family.startswith('Mixed') else (family,)
            self.terms = self.coeff.shape[-1] // len(self.families)
            assert self.terms*len(self.families) == self.coeff.shape[-1] and self.terms % 2 == 0
            gates = torch.zeros(self.coeff.shape[0], len(self.families), device=self.coeff.device, dtype=self.coeff.dtype)
            if family == 'MixedAdaptive':
                self.logits = nn.Parameter(gates)
            else:
                self.register_buffer('logits', gates)

        def forward(self, x):
            # Hidden channels are bounded by the preceding tanh, not physical coordinates.
            values = torch.cat([hidden_values(x, f, self.terms) for f in self.families], -1)
            gates = (len(self.families)*self.logits.softmax(-1)).repeat_interleave(self.terms, -1)
            return torch.einsum('nik,iok->no', values*gates, self.coeff)

    return HiddenBasis()


def install(spec):
    import torch
    from torch import nn
    import adaptive_basis as study
    import forward_training as full
    import inverse_training as inverse
    import burgers_training as burgers
    assert spec in jobs()
    case = spec['job']['case']
    if case == 'Burgers':
        burgers.install()
    elif spec['mode'] == 'forward':
        full.protocol_test()
        study.config, study.digest = full.config, full.digest
    module = inverse if spec['mode'] == 'inverse' else study
    original_model = study.AdaptiveModel
    old_config, old_digest = module.config, module.digest
    hashes = dict(old_digest())
    extension_hash = study.base.sha(Path(__file__))

    class ControlModel(original_model):
        def __init__(self, name, variant):
            assert name == case and variant == spec['job']['variant']
            if spec['condition'] == 'legacy_bridge':
                nn.Module.__init__(self)
                self.case, self.variant = name, variant
                self.has_legacy_kan = spec['input_variant'] == 'LegacyHybrid'
                original_kind = 'Hybrid' if self.has_legacy_kan else 'MLP'
                existing = study.base.existing
                if name in ('Burgers', 'KdV'):
                    old = existing.ScalarModel(name, original_kind)
                else:
                    old = existing.NSModel(([0., 0., 0.], [2*math.pi, 2*math.pi, 1.]), original_kind, [1.5, .02])
                self.mlp = old.mlp
                if self.has_legacy_kan:
                    self.kan = old.kan
                self.register_buffer('coefficients', torch.tensor([1., .01/math.pi if name == 'Burgers' else .0025 if name == 'KdV' else .01]))
            else:
                super().__init__(name, spec['input_variant'])
                self.kan[2] = hidden_layer(self.kan[2], spec['hidden_basis'])
                self.variant = variant

        def correction(self, z):
            if spec['condition'] != 'legacy_bridge':
                return super().correction(z)
            if self.case == 'KdV':
                q = z
            elif self.case == 'Burgers':
                q = torch.stack([z[:, 0], 2*z[:, 1]-1], 1)
            else:
                q = torch.stack([z[:, 0]/math.pi-1, z[:, 1]/math.pi-1, 2*z[:, 2]-1], 1)
            return self.mlp(q) + (self.kan(q) if self.has_legacy_kan else 0.)

    def config(name, smoke=False):
        cfg = dict(old_config(name, smoke))
        cfg.update(architecture_control=spec, comparison='Frozen current protocol; only declared network components change',
                   historical_protocol=False, initial_boundary_enforcement='Current hard-IC and BC retained; legacy bridge is NOT an exact old-protocol reproduction',
                   hidden_domain='tanh-bounded latent channels; not physical coordinates',
                   hidden_normalization='No added feature normalization; unchanged coefficient tensors',
                   parameter_match='Hidden replacements retain total coefficient count; mixed gates add parameters; legacy branches have original sizes')
        return cfg

    def digest():
        assert old_digest() == hashes
        assert study.base.sha(Path(__file__)) == extension_hash
        return dict(hashes, **{Path(__file__).name: extension_hash})

    study.AdaptiveModel = ControlModel
    study.gate_diagnostic = lambda model: None
    module.config, module.digest = config, digest
    return module


def self_test(spec, module, device='cpu'):
    import torch
    import adaptive_basis as study
    import inverse_training as inverse
    base = study.base
    base.seed(spec['job']['seed'])
    case, variant = spec['job']['case'], spec['job']['variant']
    factory = inverse.InverseModel if spec['mode'] == 'inverse' else study.AdaptiveModel
    model = factory(case, variant).to(device).double()
    # Nonzero readout exercises input/hidden coefficients and gates through third order.
    with torch.no_grad():
        for p in model.parameters():
            p.add_(.002*torch.randn_like(p))
    coordinates = torch.rand(5, 2 if case != 'NS_periodic' else 3, device=device, dtype=torch.float64)
    coordinates[:, 0] = 2*coordinates[:, 0]-1 if case != 'NS_periodic' else 2*math.pi*coordinates[:, 0]
    if case == 'NS_periodic':
        coordinates[:, 1] *= 2*math.pi
    coordinates.requires_grad_()
    direct = model(coordinates)
    scales = torch.tensor([.7, .3] if case != 'NS_periodic' else [.7, 1.3, .3], device=device, dtype=torch.float64)
    offsets = torch.full_like(scales, .17)
    q = ((coordinates.detach()-offsets)/scales).requires_grad_()
    transformed = model(q*scales+offsets)
    errors = []
    for channel in range(direct.shape[1]):
        for axis in range(coordinates.shape[1]):
            left, right = direct[:, channel:channel+1], transformed[:, channel:channel+1]
            for order in range(1, 4):
                left = base.derivative(left, coordinates)[:, axis:axis+1]
                right = base.derivative(right, q)[:, axis:axis+1]/scales[axis]
                assert torch.isfinite(left).all() and torch.isfinite(right).all()
                assert torch.allclose(left, right, atol=1e-8, rtol=1e-7)
                errors.append(float((left-right).abs().max().detach()))
    loss = direct.square().mean()
    for _ in range(3):
        direct = base.derivative(direct, coordinates)
        loss = loss + direct.square().mean()
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    initial = coordinates.detach().clone(); initial[:, -1] = 0.
    if case != 'NS_periodic':
        target = -torch.sin(math.pi*initial[:, :1]) if case == 'Burgers' else torch.cos(math.pi*initial[:, :1])
        assert torch.allclose(model(initial), target, atol=1e-12, rtol=0)
    return dict(passed=True, optimizer_steps=0, chain_max_absolute_difference=max(errors),
                finite_third_order_backward=True, parameters=sum(p.numel() for p in model.parameters()), spec=spec)
