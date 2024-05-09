"""
This test checks the dice score.
"""

# Imports
import numpy as np

from ..image_utils import dice

def same_image_test():
    m1 = np.random.randn(200,200)
    m2 = m1
    computed_dice = dice(m1,m2)
    assert computed_dice == 1.0
    computed_dice2 = dice(m2, m1)
    assert computed_dice2 == 1.0

def symmetry_test():
    m1 = np.random.randn(50,50)
    m2 = np.random.randn(50,50)
    assert dice(m1,m2) == dice(m2,m1)

def bounds_test():
    m1 = np.random.randn(500,500)
    m2 = np.random.randn(500,500)
    computed_dice = dice(m1,m2)
    assert computed_dice < 1.0
    assert computed_dice > 0.0

def small_shift_test():
    # See https://www.kaggle.com/code/yerramvarun/understanding-dice-coefficient
    m1 = np.random.randn(500,500)
    pxl_shift1 = 5
    m2 = np.zeros((500+pxl_shift1, 500))
    m2[pxl_shift1:,:] = m1
    m2 = m2[:-pxl_shift1,:]
    computed_dice1 = dice(m1,m2)
    pxl_shift2 = 10
    m3 = np.zeros((500+pxl_shift2, 500))
    m3[pxl_shift2:,:] = m1
    m3 = m3[:-pxl_shift2,:]
    computed_dice2 = dice(m1,m3)
    assert computed_dice1 > computed_dice2

def manual_compute_test():
    m1 = np.random.randn(5,5)
    m2 = np.random.randn(5,5)
    m1 = m1 > 0.5
    m2 = m2 > 0.5
    overlap = 0
    m1pos = 0
    m2pos = 0
    for i in range(5):
        for j in range(5):
            overlap += (m1[i,j] * m2[i,j] == 1)
            m1pos += (m1[i,j] == 1)
            m2pos += (m2[i,j] == 1)
    target = 2 * overlap / (m1pos + m2pos)
    computed = dice(m1,m2)
    assert target == computed

same_image_test()
symmetry_test()
bounds_test()
small_shift_test()
manual_compute_test()
