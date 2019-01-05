#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 15:18:10 2019

@author: Nazmul
"""
"""1. Autograd"""
# In[ ]:
import torch
# In[ ]:
x = torch.ones(2,2,requires_grad = True)
x
# In[ ]:
y = x+2
y
# In[ ]:
y.grad_fn
# In[ ]:
z = y*y*3
out = z.mean()
out
# In[ ]:3
a = torch.randn(2,2)
a = ((a*3)/(a-1))
a.requires_grad_(True)
# In[ ]:
out.backward()
# In[ ]:
x.grad
# In[ ]: