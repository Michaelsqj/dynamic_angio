import torch
import torch.nn as nn
import torch.nn.functional as F
    
class autoencoder(nn.Module):
    def __init__(self,input_dimension,hidden_dimension,model_layers = 2, nonlinearity = 'leakyrelu'):
        super().__init__()
        
        if(nonlinearity == 'leakyrelu'):
            self.activation = F.leaky_relu
        elif(nonlinearity == 'relu'):
            self.activation = F.relu
        elif(nonlinearity == 'tanh'):
            self.activation = torch.tanh
        elif(nonlinearity == 'nalini'):
            self.activation = self.nalini
        elif(nonlinearity == 'htanh'):
            self.activation = F.hardtanh
        elif(nonlinearity == 'selu'):
            self.activation = torch.nn.SELU()
            
        self.encoding_layers = nn.ModuleList([])
        for ii in range(model_layers):
            if(ii < (model_layers - 1)):#last layer goes from current dimension directly to latent dimension
                self.encoding_layers.append(nn.Linear(input_dimension // 2**ii, input_dimension // 2**(ii+1)))              
            else:
                self.encoding_layers.append(nn.Linear(input_dimension // 2**ii, hidden_dimension))
                
        
        self.decoding_layers = nn.ModuleList([])
        self.decoding_layers.append(nn.Linear(hidden_dimension,input_dimension//2**(model_layers-1)))
        for ii in range(model_layers-1):
            self.decoding_layers.append(nn.Linear(input_dimension//2**(model_layers-(ii+1)),\
                                       input_dimension//2**(model_layers-(ii+1)-1)))

    def nalini(self,x):
        return x + F.relu((x-1)/2) + F.relu((-x-1)/2)
        
    def decode(self,x):
        out = x
        out = self.decoding_layers[0](out)
        for ii in range(len(self.decoding_layers)-1):
            out = self.activation(out)
            out = self.decoding_layers[ii+1](out)
            
        return out
    
    def encode(self,x):
        out = x
        for layer in self.encoding_layers:
            out = layer(out)
            out = self.activation(out)
        return out

    def forward(self,x):
        return self.decode(self.encode(x))