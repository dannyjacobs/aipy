import unittest
import aipy.amp as amp, numpy as n

class TestRadioBody(unittest.TestCase):
    def setUp(self):
        self.fqs = n.arange(.1,.2,.01)
        bm = amp.Beam(self.fqs)
        ant0 = amp.Antenna(0,0,0,bm)
        self.aa = amp.AntennaArray(('0','0'), [ant0])
    def test_attributes(self):
        s = amp.RadioFixedBody('0:00', '0:00',
            jys=100, index=-2, mfreq=.1, name='src1')
        self.assertEqual(s._jys, 100)
        self.assertEqual(s.index, -2)
        s.compute(self.aa)
        self.assertTrue(n.all(s.get_jys() == 100 * (self.fqs / .1)**-2))

class TestBeam(unittest.TestCase):
    def setUp(self):
        self.fqs = n.arange(.1,.2,.01)
        self.bm = amp.Beam(self.fqs)
    def test_response(self):
        xyz = (0,0,1)
        self.assertTrue(n.all(self.bm.response(xyz) == n.ones_like(self.fqs)))
        x = n.array([0, .1, .2])
        y = n.array([.1, .2, 0])
        z = n.array([.2, 0, .1])
        xyz = (x,y,z)
        self.assertTrue(n.all(self.bm.response(xyz) == \
            n.ones((self.fqs.size,3))))
        self.bm.select_chans([0,1,2])
        self.assertTrue(n.all(self.bm.response(xyz) == n.ones((3,3))))

class TestBeam2DGaussian(unittest.TestCase):
    def setUp(self):
        self.fqs = n.arange(.1,.2,.01)
        self.bm = amp.Beam2DGaussian(self.fqs, .05, .025)
    def test_response(self):
        xyz = (0,0,1)
        self.assertTrue(n.all(self.bm.response(xyz) == n.ones_like(self.fqs)))
        x = n.array([0, .05,   0])
        y = n.array([0,   0, .05])
        z = n.array([1,   1,   1])
        xyz = (x,y,z)
        resp = self.bm.response(xyz)
        self.assertEqual(resp.shape, (self.fqs.size,3))
        ans = n.sqrt(n.array([1., n.exp(-1), n.exp(-4)]))
        ans.shape = (1,3)
        self.bm.select_chans([0])
        resp = self.bm.response(xyz)
        self.assertTrue(n.all(n.round(resp - ans, 3) == 0))

# TODO: other beam types

class TestAntenna(unittest.TestCase):
    def setUp(self):
        self.fqs = n.arange(.1,.2,.01)
        bm = amp.Beam2DGaussian(self.fqs, .05, .025)
        self.ant = amp.Antenna(0,0,0, beam=bm)
    def test_passband(self):
        pb = self.ant.passband()
        self.assertTrue(n.all(pb == n.ones_like(self.fqs)))
        self.ant.select_chans([0,1,2])
        pb = self.ant.passband()
        self.assertEqual(pb.shape, (3,))
    def test_bm_response(self):
        xyz = (.05,0,1)
        self.ant.select_chans([0])
        resp = self.ant.bm_response(xyz, pol='x')
        self.assertAlmostEqual(resp, n.sqrt(n.exp(-1)), 3)
        resp = self.ant.bm_response(xyz, pol='y')
        self.assertAlmostEqual(resp, n.sqrt(n.exp(-4)), 3)
        
if __name__ == '__main__':
    unittest.main()
