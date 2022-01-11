X = [0.5,2.5]
Y = [0.2,0.9]

def f(w,b,x):
  return 1.0 / (1.0+np.exp(-(w*x+b)))
def error(w,b):
  err = 0.0
  for x,y in zip(X,Y):
    fx = f(w,b,x)
    err = err + (0.5*(fx-y)**2)   
  return err
def grad_b(w,b,x,y):
  fx = f(w,b,x)
  return (fx-y)*fx*(1-fx)
def grad_w(w,b,x,y):
  fx = f(w,b,x)
  return (fx-y)*fx*(1-fx)*x

""" 
## **# Gradient_Descent**

"""

def do_gradient_descent(w,b,eta,max_epochs,error_eps):
  max_epochs = max_epochs
  for i in range(max_epochs):
    dw,db = 0,0
    for x,y in zip(X,Y):
      dw = dw + grad_w(w,b,x,y)
      db = db + grad_b(w,b,x,y)
    w = w - eta * dw
    b = b - eta * db
    err = error(w,b)
    if (err < error_eps):
      print("error %f"%err)
      print("no .of iteration for gd is %d"%i)
      break
  return w,b



"""### **#Gradient_Descent with moment**

"""

def do_momentum_gradient_descent(w,b,eta,max_epochs,error_eps):
  prev_vw = 0  
  prev_vb = 0
  gamma = 0.9
  max_epochs = max_epochs
  for i in range(max_epochs):
    dw,db = 0,0
    for x,y in zip(X,Y):
      dw = dw + grad_w(w,b,x,y)
      db = db + grad_b(w,b,x,y)
    vw = gamma*prev_vw + eta*dw
    vb = gamma*prev_vb + eta*db
    w = w - eta*vw
    b = b - eta*vb
    prev_vw = vw
    prev_vb = vb
    err = error(w,b)
    if (err < error_eps):
      print("error %f"%err) 
      print("no .of iteration for gdm is %d"%i)
      break
  return w,b

""" **#Nesterov_Gradient Descent**"""

def do_nesterov_gradient_descent(w,b,eta,max_epochs,error_eps):
  prev_vw = 0
  prev_vb = 0
  gamma = 0.9
  max_epochs = max_epochs
  for i in range(max_epochs):
    vw = gamma*prev_vw
    vb = gamma*prev_vb
    dw,db = 0,0
    for x,y in zip(X,Y):
      dw = dw + grad_w(w-vw,b-vb,x,y)
      db = db + grad_b(w-vw,b-vb,x,y) 
    vw = gamma*prev_vw + eta*dw
    vb = gamma*prev_vb + eta*db
    w = w - eta*vw
    b = b - eta*vb
    prev_vw = vw
    prev_vb = vb
    err = error(w,b)
    if (err < error_eps):
      print("error %f"%err)
      print("no .of iteration for ngd is %d"%i)
      break
  return w,b

""" **#Adaptive_Gradient Descent**"""

def do_adagrad(w,b,eta,max_epochs,error_eps):
  max_epochs = max_epochs
  vw, vb, eps = 0,0,1e-8
  for i in range(max_epochs):
    dw,db = 0,0
    for x,y in zip(X,Y):
      dw = dw + grad_w(w,b,x,y)
      db = db + grad_b(w,b,x,y)
    vw = vw + dw**2
    vb = vb + db**2
    w = w - (eta/np.sqrt(vw+eps)) * dw
    b = b - (eta/np.sqrt(vb+eps)) * db
    err = error(w,b)
    if (err < error_eps):
      print("error %f"%err)
      print("no .of iteration for agd is %d"%i)
      break
  return w,b

""" **# RMS_Prop**"""

def do_rmsprop(w,b,eta,max_epochs,error_eps):
  max_epochs = max_epochs
  vw,vb,eps,beta = 0,0,1e-8,0.9
  for i in range(max_epochs):
    dw,db = 0,0
    for x,y in zip(X,Y):
      dw = dw + grad_w(w,b,x,y)
      db = db + grad_b(w,b,x,y)
    vw = beta * vw + (1-beta)+(dw**2)
    vb = beta * vb + (1-beta)+(db**2)
    w = w - (eta/np.sqrt(vw+eps)) * dw
    b = b - (eta/np.sqrt(vb+eps)) * db
    err = error(w,b)
    if (err<error_eps):
      print("enter %f"%err)
      print("no .of iteration for rms-prop is %d"%i)
      break
  return w,b

"""## ***Adam***"""

def do_adam(w,b,eta,max_epochs,error_eps):
  max_epochs = max_epochs
  eps = 1e-8
  vw,vb,vw_hat,vb_hat = 0,0,0,0
  mw,mb,mw_hat,mb_hat = 0,0,0,0
  beta1,beta2 = 0.9,0.999
  for i in range(max_epochs):
    dw,db = 0, 0
    for x,y in zip(X,Y):
      dw = dw + grad_w(w,b,x,y)
      db = db + grad_b(w,b,x,y)
    mw = beta1*mw + (1-beta1)*dw
    mb = beta1*mb + (1-beta1)*db
    vw = beta2*vw + (1-beta2)*(dw**2)
    vb = beta2*vb + (1-beta2)*(db**2)
    
    mw_hat = mw / (1-math.pow(beta1,i+1)) 
    mb_hat = mb / (1-math.pow(beta1,i+1))
    vw_hat = vw / (1-math.pow(beta2,i+1))
    vb_hat = vb / (1-math.pow(beta2,i+1)) 
    
    w = w - (eta/np.sqrt(vw_hat+eps)) * mw_hat
    b = b - (eta/np.sqrt(vb_hat+eps)) * mb_hat
    err = error(w,b)
    if (err < error_eps):
      print("error %f"%err)
      print("no .of iteration for adam is %d"%i)
      break
  return w,b

import numpy as np
import math
init_w = -2
init_b = -2
init_eta = 1.0
max_epochs = 100000
error_eps = 0.00000001
w,b,eta = init_w,init_b,init_eta

#Calling Gradient Descent 
w_gd,b_gd = do_gradient_descent(w,b,eta,max_epochs,error_eps)
err_gd = error(w_gd,b_gd)
print("w_gd is %f"%w_gd)
print("b_gd is %f"%b_gd)

#Calling Gradient Descent with momentum
w_gdm,b_gdm = do_momentum_gradient_descent(w,b,eta,max_epochs,error_eps)
err_gdm = error(w_gdm,b_gdm)
print("w_gdm is %f"%w_gdm)
print("b_gdm is %f"%b_gdm)

#Calling Nesrerov Gradient Descent
w_ngd,b_ngd = do_nesterov_gradient_descent(w,b,eta,max_epochs,error_eps)
err_ngd = error(w_ngd,b_ngd)
print("w_ngd is %f"%w_ngd)
print("b_ngd is %f"%b_ngd)

#Calling Adagrad
w_agd,b_agd = do_adagrad(w,b,eta,max_epochs,error_eps)
err_agd = error(w_agd,b_agd)
print("w_agd is %f"%w_agd)
print("b_agd is %f"%b_agd)

#Calling rms-prop
w_rmsprop,b_rmsprop = do_rmsprop(w,b,eta,max_epochs,error_eps)
err_rmsprop = error(w_rmsprop,b_rmsprop)
print("w_rmsprop is %f"%w_rmsprop)
print("b_rmsprop is %f"%b_rmsprop)

#Calling Adam
w_adm,b_adm = do_adam(w,b,eta,max_epochs,error_eps)
err_adm = error(w_adm,b_adm)
print("w_adm is %f"%w_adm)
print("b_adm is %f"%b_adm)

