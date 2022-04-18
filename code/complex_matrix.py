# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 06:47:15 2021

@author: ssh
"""
import tensorflow as tf
import numpy as np

def cdot(A,B):
    Are = tf.real(A)
    Aim = tf.imag(A)
    Bre = tf.real(B)
    Bim = tf.imag(B)
    Cre = tf.matmul(Are,Bre) - tf.matmul(Aim,Bim)
    Cim = tf.matmul(Are,Bim) + tf.matmul(Aim,Bre)
    return tf.complex(Cre,Cim)

def cmul(A,B):
    Are = tf.real(A)
    Aim = tf.imag(A)
    Bre = tf.real(B)
    Bim = tf.imag(B)
    Cre = tf.multiply(Are,Bre) - tf.multiply(Aim,Bim)
    Cim = tf.multiply(Are,Bim) + tf.multiply(Aim,Bre)
    return tf.complex(Cre,Cim)

def cinv(A):
    Are = tf.real(A)
    Aim = tf.imag(A)
    Cre = tf.matrix_inverse(A + tf.matmul(tf.matmul(Aim, tf.matrix_inverse(Are)),Aim))
    Cim = -tf.matmul(tf.matmul(tf.matrix_inverse(Are),Aim),Cre)
    return tf.complex(Cre,Cim)
