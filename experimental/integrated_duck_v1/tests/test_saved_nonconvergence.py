"""Frozen real problems: unchanged physical certificates, repeat and rollback.

Fixtures are N20/R51/C3 inputs assembled before two rejected robot steps, not
policy rollouts. The inherited 14 generic algebra tests run unchanged too.
"""
import hashlib
from pathlib import Path
import unittest
import numpy as np
import test_coupled_impulse as algebra


class SavedNonconvergenceTests(algebra.CoupledImpulseTests):
    def saved(self, name, digest):
        path = Path(__file__).parent / 'fixtures' / ('saved_' + name + '_nonconvergence.npz')
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
        with np.load(path, allow_pickle=False) as data:
            a = dict(data)
        p = self.problem(a['mass'], a['smooth_velocity'], a['jacobian'],
                         target=a['target'], regularizer=a['regularizer'],
                         lower=a['lower'], upper=a['upper'], warm=a['warm'],
                         contacts=list(zip(a['contact_rows'], a['friction'])),
                         iterations=4096, tolerance=1e-8)
        velocity, impulse, result = self.solve(p)
        again_v, again_i, again_r = self.solve(p)
        np.testing.assert_array_equal(velocity, again_v)
        np.testing.assert_array_equal(impulse, again_i)
        self.assertEqual(result.iterations, again_r.iterations)
        self.assertTrue(np.isfinite(velocity).all() and np.isfinite(impulse).all())
        self.assertLessEqual(result.iterations, 4096)
        # Independently reconstruct physical residuals from returned velocity,
        # not native diagnostic fields or the native Cholesky response.
        G, M = a['jacobian'], a['mass']
        response = np.linalg.solve(M, G.T)
        K = np.einsum('ik,kj->ij', G, response) + np.diag(a['regularizer'])
        residual = np.einsum('ij,j->i', G, velocity) - a['target'] + a['regularizer']*impulse
        kind = np.zeros(len(impulse), dtype=int)
        for r, mu in zip(a['contact_rows'], a['friction']):
            r = int(r); kind[r:r+3] = [1, 2, 3]
            normal, tangent = impulse[r], impulse[r+1:r+3]
            cap, norm = mu*normal, np.linalg.norm(tangent)
            self.assertGreaterEqual(normal, 0)
            # Disk feasibility is part of the published impulse certificate,
            # with the same 1e-8 tolerance as normal/tangent stationarity.
            # Do not impose a separate exact-projection law on a block solver.
            self.assertLessEqual(max(0., norm-cap), p.impulse_tolerance)
            ne = min(abs(residual[r]/K[r,r]), normal if residual[r] >= 0 else np.inf)
            small = np.linalg.eigvalsh(K[r+1:r+3,r+1:r+3])[0]
            if cap == 0:
                te = norm
            elif norm == 0:
                te = np.linalg.norm(residual[r+1:r+3])/small
            else:
                unit = tangent/norm
                m = max(0., -float(np.sum(residual[r+1:r+3]*unit)))
                te = max(np.linalg.norm(residual[r+1:r+3]+m*unit)/small,
                         m/small*max(0., cap-norm)/norm, max(0., norm-cap))
            self.assertLessEqual(max(ne, te), 1e-8)
        for r in np.flatnonzero(kind == 0):
            lo, hi = a['lower'][r], a['upper'][r]
            if lo == hi:
                error = abs(impulse[r]-lo)
            else:
                correction = residual[r]/K[r,r]
                distance = impulse[r]-lo if correction > 0 else hi-impulse[r]
                error = min(abs(correction), distance)
            self.assertLessEqual(error, 1e-8)
        momentum = np.einsum('ij,j->i', M, velocity-a['smooth_velocity']) - np.einsum('ij,i->j', G, impulse)
        self.assertLessEqual(np.max(abs(momentum)), 1e-8)
        p.max_iterations = 1
        self.solve(p, expected=3)  # complete result struct and buffers unchanged

    def test_saved_training_normal_conditioning(self):
        self.saved('training', '261f95a1129c410919e57e6171d3b623f16d17e7008cd0de1d927fed4994a7bc')

    def test_saved_evaluation_redundant_contact_self_stress(self):
        self.saved('evaluation', 'ad87bd67975f2dbc142facdb7c513b3c2e7a83ca9d704eef52c214a7a8507908')


if __name__ == '__main__':
    unittest.main(verbosity=2)
