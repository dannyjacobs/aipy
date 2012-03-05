#! /usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import aipy as a, numpy as n
import pylab as p

class TestDeconv(unittest.TestCase):
    def setUp(self):
        SIZE = 50
        NOISE = .001
        i = n.zeros((SIZE,SIZE), n.float)
        i[10,10] = 10.
        i[20:25,20:25] = 1.
        i[30:40,30:40] = .1
        self.b = a.img.gaussian_beam(2, shape=i.shape)
        self.b[0,0] = .05
        self.i = i
        self.d = n.abs(n.fft.ifft2(n.fft.fft2(i) * n.fft.fft2(self.b)))
        ns = n.random.normal(scale=NOISE, size=i.shape)
        #self.d = n.abs(self.d + ns)
        
    def test_clean(self):
        """Test that the standard clean deconvolution runs"""
        #print 'Clean'
        p.subplot(131)
        c,info = a.deconv.clean(self.d, self.b, verbose=True,tol=1.e-4,stop_if_div=False,pos_def=True)
        print "max i,c,i-c"
        print self.i.max(),c.max(),(self.i-c).max()
        p.imshow(n.log10(self.i),vmin=-5, vmax=0)
#        p.imshow(self.i,vmin=0.1,vmax=5)
        p.title('model')

        p.subplot(132)
#        p.imshow(c,vmin=0.1,vmax=5)
        p.imshow(n.log10(c),vmin=-5, vmax=0)
        p.title('cc')

        p.subplot(133)
        p.imshow(n.log10(n.abs(c - self.i)),vmin=-5, vmax=0)
#        p.imshow(c-self.i,vmin=0.1,vmax=5)
        p.title('cc - model')
        #print n.sum(c - self.i)
        #p.title('CLEAN')
        #p.imshow(n.log10(c), vmin=-5, vmax=1)
        p.show()
        self.assertAlmostEqual(n.sum(self.i - c),0)

    def test_lsq(self):
        """Test that least squared deconvolution runs"""
        #print 'LSQ'
        #p.subplot(222)
        c,info = a.deconv.lsq(self.d, self.b, verbose=False)
        #p.title('LSQ')
        #p.imshow(n.log10(c), vmin=-5, vmax=1)
        
    def test_mem(self):
        """Test the maximum entropy deconvolution runs"""
        #print 'MEM'
        #p.subplot(223)
        c,info = a.deconv.maxent(self.d, self.b, n.var(self.d**2)*.5, verbose=False)
        #p.title('MEM')
        #p.imshow(n.log10(c), vmin=-5, vmax=1)

    def test_anneal(self):
        """Test that simulated annealing deconvolution runs"""
        #print 'Anneal'
        #p.subplot(224)
        c,info = a.deconv.anneal(self.d, self.b, verbose=False)
        #p.title('Anneal')
        #p.imshow(n.log10(c), vmin=-5, vmax=1)
        #p.colorbar()
        #p.show()

class TestSuite(unittest.TestSuite):
    """A unittest.TestSuite class which contains all of the aipy.deconv unit tests."""

    def __init__(self):
        unittest.TestSuite.__init__(self)

        loader = unittest.TestLoader()
        self.addTests(loader.loadTestsFromTestCase(TestDeconv))

if __name__ == '__main__':
    unittest.main()
