import torch 
import os
from PIL import Image
from re import split
import pickle

from masking.hooked_model import HookedModel


class ActivationModel(HookedModel):
    def __init__(self,model,layers):
        super(ActivationModel,self).__init__(model)

        self.layers = (layers if isinstance(layers,list) else [layers])
        print (self.layers)
        for layer in self.layers:
            self.apply_hook(layer,GetActivationsHook(layer))

    def forward(self,*args,**kwargs):
        self.model.eval()
        self.model(*args,**kwargs)
        if len(self.layers) == 1:
            return self.hooks[self.layers[0]].get_activations()
        else:
            return self.get_activations()

    def get_activations(self):
        return dict(zip(self.hooks.keys(),
                        [self.hooks[k].get_activations() for k in self.hooks.keys()]))

    def save_activations(self, image_list, transforms, output_location, device='cuda'):
        """
        Mask all layers
        For every image
            Evaluate model
            Save activations
        """
        self.model.eval()

        for file in image_list:

            filename = split('\.|\/|\_',file)[-2]+'.pkl'

            # if os.path.isfile(os.path.join(output_location,filename)):
            #     continue

            print('saving ',filename)

            im = Image.open(file).convert('RGB')
            im = transforms(im).unsqueeze(0).to(device)

            with torch.no_grad():
                self(im)
            
            activations = self.get_activations()

            pickle.dump(activations,open(os.path.join(output_location,filename),'wb'))


class GetActivationsHook:
    """
    Hook for retrieving activations from output of layer.
    """
    def __init__(self,name):
        self.name = name
        self.activations = []

    def __call__(self,model, input, output):
        self.activations = output.detach().clone()
    
    def get_activations(self):
        return self.activations