import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import gc

class EarlyStopper():
    def __init__(self, patience=5, min_delta=0, filename='optimal_weight.pth'):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_val_loss = np.inf
        self.filename = filename

    def early_stop(self, model, val_loss):
            
        if val_loss < self.min_val_loss:
            self.min_val_loss = val_loss
            torch.save(model.state_dict(), self.filename)
            self.counter = 0
        elif val_loss > (self.min_val_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                model.load_state_dict(torch.load(self.filename))
                return True
        return False
    
class SignSTE(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return torch.where(input >= 0,
                           torch.ones_like(input),
                           -torch.ones_like(input))

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors

        # Clipped STE: derivative = 1 if |x| <= 1
        grad_input = grad_output.clone()
        grad_input[input.abs() > 1] = 0

        return grad_input

class Sat(nn.Module):
    def __init__(self, a=1.0):
        super().__init__()
        self.a = float(a)

    def forward(self, x):
        return torch.clamp(x, -self.a, self.a)
    
# RHN class definition
class Nonlinear_RHN(nn.Module):

    def __init__(self, arch = np.array([784, 500, 300, 200]), act_type='tanh', boost=True, device='cuda'):
        super().__init__()    
        self.title = 'rhn'             
        
        self.arch = arch
        self.weights = torch.nn.ParameterList()
        
        for ix in range(len(arch) - 1):
            w = torch.rand(arch[ix], arch[ix + 1], dtype=torch.float32)
            w = torch.nn.init.orthogonal_(w)
            
            self.weights.append(
                torch.nn.Parameter(w, requires_grad=False)
            )
        
                
        self.n_layer = len(arch) - 1
        self.boost = boost
        
        self.act_type = act_type
        
        self.stop = False

        
        self.loss_lst = list()
        
        if self.act_type == 'tanh':
            self.act_fun = torch.tanh
        else:
            self.act_fun = torch.sign
            
        self.device = device
        self.dtype = torch.float32
            
        self.to(self.device, dtype=torch.float32)
            
        

    def left_train_layer(self, x):

        if self.stop == True:
            return 0
        
        x = x.to(self.device, dtype=self.dtype)   # ⭐ IMPORTANT
        output_x = x.clone()
        input_x = x.clone()
        
        for ix in range(self.n_layer):
            #print("Inside Layer : ", ix)
            output_x = self.act_fun(output_x @ self.weights[ix])
                 
        # train the model from inside to the outside
        for ix in range(self.n_layer - 1, -1, -1):
            #print("Layer IX : ", ix)
            forward_x = input_x.clone()
            forward_x_b = input_x.clone()
            
            for iy in range(ix):
                #print("IY : ", iy)
                forward_x_b = forward_x @ self.weights[iy]
                forward_x = self.act_fun(forward_x_b)
                
            
            backward_x = output_x.clone()
            backward_x_b = output_x.clone()
            for iz in range(self.n_layer - 1, ix-1, -1):
                #print("Iz : ", iz)
                backward_x_b = backward_x @ self.weights[iz].T
                backward_x = self.act_fun(backward_x_b)
                
                
            if ix == 0:
                trans_mat = forward_x_b.T @ backward_x_b
            else:
                trans_mat = forward_x_b.T @ backward_x
            #trans_mat = forward_x_b.T @ backward_x_b
            u, d, v = torch.linalg.svd(trans_mat)
            self.weights[ix] = u @ v @ self.weights[ix]
            
            del trans_mat, u, d, v, forward_x, backward_x, backward_x_b, forward_x_b
            gc.collect()
        
        y_pred = self.forward(input_x)
        loss_val_orig = F.mse_loss(y_pred, x)
        loss_val = F.mse_loss(torch.sign(y_pred), x)
        print("Left Loss Value : ", loss_val.item(), " Original Loss Value : ", loss_val_orig.item())
        
        self.loss_lst.append(loss_val.item())
        
        if loss_val == 0:
            print("Object Reach!")
            self.stop = True
        
        del y_pred, output_x, input_x
        gc.collect()
            
                 
        return 0

      
    def right_train_layer(self, x):
        
        if self.stop == True:
            return 0
        
        x = x.to(self.device, dtype=self.dtype)   # ⭐ IMPORTANT

        output_x = x.clone()
        input_x = x.clone()
        for ix in range(self.n_layer):
            output_x = torch.tanh(output_x @ self.weights[ix])

        # train the model from inside to the outside
        # since the last layer is not meaning to update
        # therefore the layer is updated from the second last layer
        for ix in range(self.n_layer-2, -1, -1):
            #print("IX : ", ix)
            forward_x = input_x.clone()
            forward_x_b = input_x.clone()
            for iy in range(ix+1):
                forward_x_b = forward_x @ self.weights[iy]
                forward_x = torch.tanh(forward_x_b)
                
            #print("Forward X Shape :", forward_x.shape)
            backward_x = output_x.clone()
            backward_x_b = output_x.clone()
            for iz in range(self.n_layer-1, ix, -1):
                backward_x_b = backward_x @ self.weights[iz].T
                backward_x = torch.tanh(backward_x_b)
                
            #print("Backward X Shape :", backward_x.shape)
            trans_mat = forward_x_b.T @ backward_x
            u, d, v = torch.linalg.svd(trans_mat)
            
            self.weights[ix] = self.weights[ix] @ u @ v

            del trans_mat, u, d, v, forward_x, backward_x
            gc.collect()
                

        y_pred = self.forward(input_x)
        loss_val_orig = F.mse_loss(y_pred, input_x)
        loss_val = F.mse_loss(torch.sign(y_pred), input_x)
        print("Right Loss Value : ", loss_val.item(), " Original Loss Value :", loss_val_orig.item())
        
        
        self.loss_lst.append(loss_val.item())
        
        if loss_val == 0:
            print("Object Reach!")
            self.stop = True
        
        del y_pred, output_x, input_x
        gc.collect()
                 
        return 0
    
            

    def loss(self, x, y):
        return F.mse_loss(x, y)

    # query the neural network forward and backward once
    def bp_train(self, patterns, lr=1e-4, epoches=1000):
        
        for ix in range(self.n_layer):
            self.weights[ix] = Variable(self.weights[ix], requires_grad=True)

        mse_loss = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.weights, lr=lr)

        for epoch in range(epoches):
            sum_loss = []

            optimizer.zero_grad()
            f_output = self.forward(patterns, logit=True)

            loss_val = mse_loss(f_output, patterns)
            sum_loss.append(loss_val.item())

            loss_val.backward(retain_graph=True)
            optimizer.step()
            
            act_loss = mse_loss(torch.sign(f_output), patterns)

            print("Epoch : ", epoch+1,  "MSE Loss : ", loss_val.item(), \
                  "Actual Loss value : ", act_loss.item())
            
            if act_loss.item() == 0 and loss_val.item() < 5e-3:
                break
            
        return 0
    
    def forward(self, x, logit=True):
        forward_x = x.clone()
        
        for ix in range(self.n_layer):
            forward_x = self.act_fun(forward_x @ self.weights[ix])
            
            backward_x = forward_x.clone()
            
        for iy in range(self.n_layer-1, -1, -1):
            backward_x_b = backward_x @ self.weights[iy].T
            backward_x = self.act_fun(backward_x_b)
            
        if logit == True:
            return backward_x_b
        else:
            return backward_x
        
                 
    def query(self, x, num=100) :
        stable = False
        inputs = x
        counter = 0
        record_change = list()
        while(stable == False):
            output = self.forward(inputs, logit=False)
            final_y = torch.sign(output)

            err = self.loss(inputs, final_y)
            if err < 1e-5 or counter==num:
                stable = True
            else:
                inputs = final_y
                counter += 1
                record_change.append(final_y)

        return final_y, record_change
    
# RHN class definition
class Linear_RHN(nn.Module):

    def __init__(self, arch = np.array([784, 500, 300, 200]), boost=True):
        super(Linear_RHN, self).__init__()                 

        self.title = 'linear_rhn'
        self.arch = arch
        self.weights = torch.nn.ParameterList()
        
        for ix in range(len(arch) - 1):
            w = torch.rand(arch[ix], arch[ix + 1], dtype=torch.float32)
            w = torch.nn.init.orthogonal_(w)
            self.weights.append(torch.nn.Parameter(w, requires_grad=False))
        self.n_layer = len(arch) - 1

        self.stop = False
        

    def left_train_layer(self, x, epoches):
        
        
        if self.stop == True:
            return 0
            
        for index_epoch in range(epoches):

            output_x = x.clone()
            input_x = x.clone()
            
            for ix in range(self.n_layer):
                output_x = output_x @ self.weights[ix]
                
                     
            # train the model from inside to the outside
            for ix in range(self.n_layer - 1, -1, -1):
                #print("Layer IX : ", ix)
                forward_x = input_x.clone()
                for iy in range(ix):
                    forward_x = forward_x @ self.weights[iy]
                    
                
                backward_x = output_x.clone()
                for iz in range(self.n_layer - 1, ix-1, -1):
                    backward_x = backward_x @ self.weights[iz].T
                
                
                trans_mat = forward_x.T @ backward_x
                u, d, v = torch.linalg.svd(trans_mat)
                self.weights[ix] = u @ v @ self.weights[ix]
                
                del trans_mat, u, d, v, forward_x, backward_x
                gc.collect()
            
            y_pred = self.forward(input_x)
            loss_val_orig = F.mse_loss(y_pred, x)
            loss_val = F.mse_loss(torch.sign(y_pred), x)
            print("Left Loss Value : ", loss_val.item(), " Original Loss Value :", loss_val_orig.item())
            
            if loss_val == 0:
                print("Object Reach!")
                self.stop = True
                break
            
            del y_pred, output_x, input_x
            gc.collect()
                 
        return 0
    
    def right_train_layer(self, x, epoches):
        
        
        if self.stop == True:
            return 0

        
        for index_epoch in range(epoches):
            output_x = x.clone()
            input_x = x.clone()
            
            for ix in range(self.n_layer):
                output_x = output_x @ self.weights[ix]

            # train the model from inside to the outside
            # since the last layer is not meaning to update
            # therefore the layer is updated from the second last layer
            for ix in range(self.n_layer-2, -1, -1):
                #print("IX : ", ix)
                forward_x = input_x.clone()
                for iy in range(ix+1):
                    forward_x = forward_x @ self.weights[iy]
                    
                #print("Forward X Shape :", forward_x.shape)
                backward_x = output_x.clone()
                for iz in range(self.n_layer-1, ix, -1):
                    backward_x = backward_x @ self.weights[iz].T
                    
                #print("Backward X Shape :", backward_x.shape)
                trans_mat = forward_x.T @ backward_x
                u, d, v = torch.linalg.svd(trans_mat)
                self.weights[ix] = self.weights[ix] @ u @ v

                del trans_mat, u, d, v, forward_x, backward_x
                gc.collect()
                    

            y_pred = self.forward(input_x)
            loss_val = F.mse_loss(torch.sign(y_pred), x)
            print("Right Loss Value : ", loss_val.item())
            
            
            if loss_val == 0:
                print("Object Reach!")
                self.stop = True
                break
            
            del y_pred, output_x, input_x
            gc.collect()
                 
        return 0

    def loss(self, x, y):
        return F.mse_loss(x, y)

    # query the neural network forward and backward once
    def bp_train(self, patterns, lr=1e-4, epoches=1000):
        
        for ix in range(self.n_layer):
            self.weights[ix] = Variable(self.weights[ix], requires_grad=True)

        optimizer = torch.optim.Adam(self.weights, lr=lr)

        for epoch in range(epoches):
            sum_loss = []

            optimizer.zero_grad()
            f_output = self.forward(patterns)

            loss_val = F.mse_loss(f_output, patterns)
            sum_loss.append(loss_val.item())

            loss_val.backward(retain_graph=True)
            optimizer.step()
            
            act_loss = F.mse_loss(torch.sign(f_output), patterns)

            print("Epoch : ", epoch+1,  "MSE Loss : ", loss_val.item(), \
                  "Actual Loss value : ", act_loss.item())
            
            if act_loss.item() == 0 and loss_val.item() < 5e-3:
                break
            
        return 0
    
    def forward(self, x, logit=None):
        forward_x = x.clone()
        
        for ix in range(self.n_layer):
            forward_x = forward_x @ self.weights[ix]
            
            backward_x = forward_x.clone()
        
        for iy in range(self.n_layer-1, -1, -1):
            backward_x = backward_x @ self.weights[iy].T
            
        return backward_x
        
                 
    def query(self, x, num=100):
        stable = False
        inputs = x
        counter = 0
        record_change = list()
        while(stable == False):
            output = self.forward(inputs)
            final_y = torch.sign(output)

            err = self.loss(inputs, final_y)
            if err < 1e-5 or counter==num:
                stable = True
            else:
                inputs = final_y
                counter += 1
                record_change.append(final_y)

        return final_y, record_change
