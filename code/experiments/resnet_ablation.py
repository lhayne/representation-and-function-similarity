from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import numpy as np
from scipy.io import loadmat
from scipy import stats
from scipy.spatial import distance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import pickle
import math
import glob
from re import split
import torch
from PIL import Image
import torchvision
import gc
import sys
import time


def activation_to_magnitude(coordinates):
    """
    Magnitude is proportional to the sum of animate and inanimate activations
    
    dot([x,y],[1,1]) / norm([1,1])
    """
    magnitude = np.sum(coordinates,axis=1)/np.sqrt(2)
    return magnitude 


def activation_to_selectivity(coordinates):
    """
    Selectivity is proportional to Animate - Inanimate activations
    
    dot([x,y],[-1,1]) / norm([-1,1])
    """
    selectivity = (coordinates[:,0] - coordinates[:,1])/np.sqrt(2)
    return selectivity


def grid_space(x,y,y_partitions=28,x_partitions=28,symmetric=False):
    """
    Takes in set of coordinates in 2D space and returns geopandas.GeoDataFrame
    where each entry represents a single cell in the grid space with an equal number
    of units. Symmetric grids don't necessarily have the same number of units per cell.
    
    Parameters
    ----------
        x (1D list) : List of x coordinates
        y (1D list) : List of y coordinates
        y_partitions: Number of partitions in Y direction, should be even number so that
                      cells can be symmetrical around zero
        x_partitions: Number of partitions in X direction, should be even number so that
                      cells can be symmetrical around zero
        symmetric (bool): Whether or not to make it symmetric around zero
                      
    Returns
    -------
        geopandas.GeoDataFrame : one entry per cell in grid starting at bottom left and going right
        
    """
    if symmetric:
        y_neg_sorted = np.sort(y[y<0])
        y_pos_sorted = np.sort(y[y>0])

        # First half of bounds come from negative region, second from positive
        y_bounds = ([y_neg_sorted[int(((2*i)/y_partitions)*len(y_neg_sorted))] 
                         for i in range(int(y_partitions/2))] + [0] +
                    [y_pos_sorted[int(((2*i)/y_partitions)*len(y_pos_sorted))] 
                         for i in range(1,int(y_partitions/2))] + [y_pos_sorted[-1]])
    else:
        y_sorted = np.sort(y)
        y_bounds = ([y_sorted[math.floor((i/y_partitions)*len(y_sorted))] 
                         for i in range(y_partitions)] + [y_sorted[-1]])

    grid_cells = []
    
    for i,y_lower_bound in enumerate(y_bounds[:-1]):
        y_upper_bound = y_bounds[i+1]
        
        if symmetric:
            # Only look at x coordinates which fall within vertical (y direction) strip of interest
            x_neg_sorted = np.sort(x[(y > y_lower_bound) & (y < y_upper_bound) & (x < 0)])
            x_pos_sorted = np.sort(x[(y > y_lower_bound) & (y < y_upper_bound) & (x > 0)])

            # First half of bounds come from negative region, second from positive
            x_bounds = ([x_neg_sorted[int(((2*k)/x_partitions)*len(x_neg_sorted))] 
                             for k in range(int(x_partitions/2))] + [0] +
                        [x_pos_sorted[int(((2*k)/x_partitions)*len(x_pos_sorted))] 
                             for k in range(1,int(x_partitions/2))] + [x_pos_sorted[-1]])
        else:
            x_sorted = np.sort(x[(y > y_lower_bound) & (y < y_upper_bound)])
            x_bounds = ([x_sorted[int((k/x_partitions)*len(x_sorted))] 
                             for k in range(x_partitions)] + [x_sorted[-1]])
        
        # Add bounds to list
        for j,x_lower_bound in enumerate(x_bounds[:-1]):
            x_upper_bound = x_bounds[j+1]
            # grid_cells.append(shapely.geometry.box(x_lower_bound, y_lower_bound, 
            #                                        x_upper_bound, y_upper_bound))
            grid_cells.append([x_lower_bound, y_lower_bound, 
                               x_upper_bound, y_upper_bound])
    
    # I don't know what this CRS projection is...
    # crs = "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +a=6371007.181 +b=6371007.181 +units=m +no_defs"
    # return geopandas.GeoDataFrame(grid_cells, columns=['geometry'], crs=crs)

    return grid_cells


class HookedModel(torch.nn.Module):
    """
    Constructs a model for applying forward hooks.
    Interface:
        1. Choose model
        2. Choose layer
        3. Choose mask
        4. Apply mask
        5. Evaluate model
        6. Remove mask
    """
    def __init__(self,model):
        super(HookedModel,self).__init__()
        self.model = model
        self.hooks = []
    
    def forward(self,*args,**kwargs):
        return self.model(*args,**kwargs)

    def apply_hook(self,layer_name,hook):
        self.hooks.append(
            get_module(self.model,layer_name).register_forward_hook(hook)
        )

    def remove_hooks(self):
        for _ in range(len(self.hooks)):
            hook = self.hooks.pop()
            hook.remove()


class OutputMaskHook:
    """
    Hook for applying elementwise mask to output of layer.
    """
    def __init__(self,mask):
        self.mask = mask

    def __call__(self, model, input, output):
        output = torch.mul(output, self.mask) # Elementwise multiplication
        return output


class GetActivationsHook:
    """
    Hook for retrieving activations from output of layer.
    """
    def __init__(self,name):
        self.name = name
        self.activations = []

    def __call__(self, model, input, output):
        # self.activations[threading.get_native_id()] = output.clone().cpu().detach().numpy()
        self.activations = output.clone().cpu().detach().numpy()
    
    def get_activations(self):
        return self.activations

    def clear(self):
        self.activations = []

def get_module(model, name):
    """
    Finds the named module within the given model.
    Courtesy of https://github.com/kmeng01/rome/blob/main/util/nethook.py#L355
    """
    for n, m in model.named_modules():
        if n == name:
            return m
    raise LookupError(name)
    
    
def top5accuracy(true, predicted):
    """
    Function to predict the top 5 accuracy
    """
    assert len(true) == len(predicted)
    result = []
    flag  = 0
    for i in range(len(true)):
        flag  = 0
        temp = true[i]
        for j in predicted[i][0:5]:
            if j == temp:
                flag = 1
                break
        if flag == 1:
            result.append(1)
        else:
            result.append(0)
    counter = 0.
    for i in result:
        if i == 1:
            counter += 1.
    error = 1.0 - counter/float(len(result))
    #print len(np.where(np.asarray(result) == 1)[0])
    return len(np.where(np.asarray(result) == 1)[0]), error


def get_ranks(true, predicted):
    assert len(true) == len(predicted)
    ranks = []
    for i,row in enumerate(predicted):
        ranks.append((np.asarray(row)==true[i]).nonzero()[0].item())
    return ranks


def mean_rank_deficit(original_ranks, predicted_ranks):
    """
    Average number of ranks the correct class dropped in predicted ranks compared with
    the true or original ranks. If ranks improved, we can't say anything about the lesion
    so we just return zero which means that no deficit occured.
    """
    assert len(original_ranks) == len(predicted_ranks)
    diff = np.asarray(predicted_ranks,dtype=float)-np.asarray(original_ranks,dtype=float)
    diff[diff < 0] = 0 # If the model improves we count that as no deficit
    return np.mean(diff)


def collect_image_activations(model,image_path_list,existing_activation_ids,save_path=None,layers=[]):
    """
    Construct a M x N matrix, where M is the number of images and N the number of
    neurons by collecting activations from all the neurons in the network.
    """

    for j in range(len(image_path_list)):
        wid = image_path_list[j].split('/')[-1].split('.')[0].split('_')[-1]
        if wid in existing_activation_ids:
            continue
        
        print(j,image_path_list[j])
        im_temp = Image.open(image_path_list[j]).convert('RGB')
        preprocess = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1.transforms(),
        ])
        im_temp = preprocess(im_temp).unsqueeze(0).to('cuda:0')
        data = dict()

        hooked_model = HookedModel(model)

        for layer in layers:
            mask_hook = GetActivationsHook(layer)
            hooked_model.apply_hook(layer,mask_hook)
            with torch.no_grad():
                hooked_model(im_temp)
            hooked_model.remove_hooks()
            
            data[layer] = mask_hook.get_activations()
            print(layer,data[layer].shape)
    
        # Save as we go
        if save_path != None:
            if not os.path.isdir(os.path.join(save_path)):
                os.makedirs(os.path.join(save_path))
            with open(os.path.join(save_path,wid+'.pkl'), 'wb') as handle:
                pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def euclidean_rdm(activations):
    """
    Calculates a euclidean distance RDM between two M x N activation matrices,
    where M is the number of inputs/examples and N is the number of neuron activations.
    """
    rdm = np.zeros((len(activations),len(activations)))
    for i in range(len(activations)):
        for j in range(len(activations)):
            rdm[i][j] = distance.euclidean(activations[i],activations[j])

    return rdm


def main():

    def id_to_words(id_):
        return synsets[corr_inv[id_] - 1][2][0]


    def pprint_output(out, n_max_synsets=10):
        wids = []
        best_ids = out.argsort()[::-1][:n_max_synsets]
        for u in best_ids:
            wids.append(str(synsets[corr_inv[u] - 1][1][0]))
        #print('%.2f' % round(100 * out[u], 2) + ' : ' + id_to_words(u)+' '+ str(synsets[corr_inv[u] - 1][1][0]))
        return wids

    model_name = 'resnet50'
    data_path = '../data/'
    experiment_path = '../data/experiments/'
    
    classes = ['schooner','brain_coral','junco_bird','snail','grey_whale','siberian_husky','electric_fan','bookcase','fountain_pen','toaster']
    class_wids = ['n04147183','n01917289','n01534433','n01944390','n02066245','n02110185','n03271574','n02870880','n03388183','n04442312']

    # Load model
    model = torchvision.models.resnet50(weights='DEFAULT').to('cuda:0')
    layers = [n for n,m in model.named_modules() if isinstance(m,torchvision.models.resnet.Bottleneck)]
    model.eval()

    # Using command line to process classes selectively
    args = sys.argv[1:]
    if len(args)!=0 and args[0] == '--classes':
        classes_to_process = args[1:]
    else:
        classes_to_process = classes

    if len(args)!=0 and '--layer_start' in args:
        layer_start = args[args.index('--layer_start')+1]
    else:
        layer_start = layers[0]

    #Load the details of all the 1000 classes and the function to convert the synset id to words
    meta_clsloc_file = data_path+'meta_clsloc.mat'
    synsets = loadmat(meta_clsloc_file)['synsets'][0]
    synsets_imagenet_sorted = sorted([(int(s[0]), str(s[1][0])) for s in synsets[:1000]],key=lambda v: v[1])
    corr = {}
    for j in range(1000):
        corr[synsets_imagenet_sorted[j][0]] = j

    corr_inv = {}
    for j in range(1, 1001):
        corr_inv[corr[j]] = j

    #Code snippet to load the ground truth labels to measure the performance
    truth = {}
    with open(data_path+'ILSVRC2014_clsloc_validation_ground_truth.txt') as f:
        line_num = 1
        for line in f.readlines():
            ind_ = int(line)
            temp  = None
            for i in synsets_imagenet_sorted:
                if i[0] == ind_:
                    temp = i
            #print ind_,temp
            if temp != None:
                truth[line_num] = temp
            else:
                print('##########', ind_)
                pass
            line_num += 1
    
    # Get list of all images
    im_valid_test = glob.glob(data_path+'images/*')
    im_valid_test = np.asarray(im_valid_test)

    # Make list of wids
    true_valid_wids = []
    for i in im_valid_test:
        temp1 = i.split('/')[-1]
        temp = temp1.split('.')[0].split('_')[-1]
        true_valid_wids.append(truth[int(temp)][1])
    true_valid_wids = np.asarray(true_valid_wids)

    # load images
    images = []
    for im in im_valid_test:
        im_temp = Image.open(im).convert('RGB')
        preprocess = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.models.ResNet50_Weights.IMAGENET1K_V2.transforms(),
        ])
        im_temp = preprocess(im_temp)
        images.append(im_temp)

    images = torch.stack(images)

    print ('loading activations')

    # If activations for all neurons for all images exist, grab those
    # Average them across images in each category and save them
    # If activations for all neurons for all images exist, grab those
    # Average them across images in each category and save them
    if os.path.isdir(os.path.join(data_path,'activations')):
        existing_activations = os.listdir(os.path.join(data_path,'activations'))
        existing_activation_ids = [split('\.',di)[0] for di in existing_activations]
    
    if len(existing_activations)<len(im_valid_test):
        collect_image_activations(model,im_valid_test,existing_activation_ids,
                                    os.path.join(data_path,'activations'),layers)

    unit_activations = {l:[] for l in layers}
    for im in im_valid_test:
        wid = im.split('/')[-1].split('.')[0].split('_')[-1]
        with open(os.path.join(data_path,'activations',wid+'.pkl'), 'rb') as f:
            activations = pickle.load(f)
            for l in layers:
                unit_activations[l].append(activations[l].flatten())

    unit_activations = {l:np.row_stack(unit_activations[l]) for l in layers}
    print(unit_activations[layers[0]].shape)

    print ('calculating baseline ranks')
    if os.path.isfile(os.path.join(experiment_path,'baseline_ranks.pkl')):
        baseline_ranks = pickle.load(open(os.path.join(experiment_path,'baseline_ranks.pkl'),'rb'))
    else:
        outs = []
        for im in images:
            with torch.no_grad():
                out = model(im.unsqueeze(0).to('cuda:0'))
                outs.append(out)
        out = torch.row_stack(outs).cpu().numpy()

        print(out)

        predicted_valid_wids = []
        for i in range(len(im_valid_test)):
            predicted_valid_wids.append(pprint_output(out[i],1000))
        predicted_valid_wids = np.asarray(predicted_valid_wids)

        # Count errors and save baseline ranks
        count, error  = top5accuracy(true_valid_wids, predicted_valid_wids)
        baseline_ranks  = np.asarray(get_ranks(true_valid_wids,predicted_valid_wids))

        print (baseline_ranks.shape)
        print('baseline '+str(count)+' '+str(len(true_valid_wids))+' '+str(error)+' '+str(1-error))

        with open(os.path.join(experiment_path,'baseline_ranks.pkl'), 'wb') as handle:
            pickle.dump(baseline_ranks, handle, protocol=pickle.HIGHEST_PROTOCOL)

    del model
    gc.collect()

    # For each class
        # Calculate Selectivity and Magnitude
        # Grid the space
        # For each cell
            # Calculate SRD
            # Calculate SRS
    # Estimate Distribution of 
        # SRD
        # SRS

    for class_idx,c in enumerate(classes):
        layer_start_seen = False

        if c not in classes_to_process:
            continue
        
        print ('calculating activations')
        class_indexes = [idx for idx in range(len(true_valid_wids)) if true_valid_wids[idx]==class_wids[class_idx]]
        other_indexes = [idx for idx in range(len(true_valid_wids)) if true_valid_wids[idx]!=class_wids[class_idx]]
        class_activations = {l:np.mean(unit_activations[l][class_indexes],axis=0) for l in layers}
        other_activations = {l:np.mean(unit_activations[l][other_indexes],axis=0) for l in layers}

        X = {l:np.column_stack((class_activations[l],other_activations[l])) for l in layers}

        # Labels correspond to class indexes
        y = np.asarray([1 if i in class_indexes else 0 for i in range(len(unit_activations[layers[0]]))])

        for layer_idx,layer in enumerate(layers):
            if not layer_start_seen:
                if layer == layer_start:
                    layer_start_seen = True
                else:
                    continue
            # Magnitude and selectivity on a per layer basis
            magnitude = activation_to_magnitude(X[layer])
            selectivity = activation_to_selectivity(X[layer])

            print ('generating grid')
            x_partitions,y_partitions = 4,4
            cell = grid_space(magnitude,selectivity,x_partitions=x_partitions,y_partitions=y_partitions)

            with open(os.path.join(experiment_path,'grid_specifications_'+c+'_'+layer+'.pkl'), 'wb') as handle:
                pickle.dump([cell], handle, protocol=pickle.HIGHEST_PROTOCOL)

            # Don't have access to geopandas in python 2.7, so we're brute forcing unit assignments to cells.
            units_in_cells = dict()
            for cell_index in range(len(cell)):
                units_in_cells[cell_index] = []
                for unit_index in range(len(magnitude)):
                    if (magnitude[unit_index] >= cell[cell_index][0] and
                        selectivity[unit_index] >= cell[cell_index][1] and
                        magnitude[unit_index] <= cell[cell_index][2] and
                        selectivity[unit_index] <= cell[cell_index][3]):
                        units_in_cells[cell_index].append(unit_index)
                print (len(units_in_cells[cell_index]),
                        min(units_in_cells[cell_index]),
                        max(units_in_cells[cell_index]))

            with open(os.path.join(experiment_path,'units_in_cells_'+c+'_'+layer+'.pkl'), 'wb') as handle:
                pickle.dump([units_in_cells], handle, protocol=pickle.HIGHEST_PROTOCOL)

            # for each cell in grid
            print ('calculating rank deficits for each cell')
            cell_srd = {}
            for bbx in range(len(cell)):
                start = time.time()
                
                # Query indices of units in that cell, create mask and set activations to zero
                loc_new = units_in_cells[bbx]
                lambda_mask = np.ones(shape=((len(magnitude),)),dtype=np.float32)

                lambda_mask[loc_new] = 0.
                lambda_mask = lambda_mask.reshape(activations[layer].shape[1:])
                print('Cell: ', bbx, ' Units: ', len(loc_new))
                
                # Skip this cell if no units lie within it
                if len(loc_new) == 0.:
                    cell_srd[bbx] = [0,0,0]
                    continue

                model = torchvision.models.resnet50(weights='DEFAULT')
                hooked_model = HookedModel(model).to('cuda:0')
                mask_hook = OutputMaskHook(torch.from_numpy(lambda_mask).to('cuda:0'))
                hooked_model.apply_hook(layer,mask_hook)
                hooked_model.eval()
                
                outs = []
                for im in images:
                    with torch.no_grad():
                        out = hooked_model(im.unsqueeze(0).to('cuda:0'))
                        outs.append(out)
                out = torch.row_stack(outs).cpu().numpy()

                predicted_valid_wids = []
                for i in range(len(im_valid_test)):
                    predicted_valid_wids.append(pprint_output(out[i],1000))
                predicted_valid_wids = np.asarray(predicted_valid_wids)

                # calculate ranks
                count, error  = top5accuracy(true_valid_wids[class_indexes], predicted_valid_wids[class_indexes])
                class_ranks  = get_ranks(true_valid_wids[class_indexes],
                                        predicted_valid_wids[class_indexes])
                other_ranks  = get_ranks(true_valid_wids[other_indexes],
                                        predicted_valid_wids[other_indexes])
                class_mrd = mean_rank_deficit(baseline_ranks[class_indexes],class_ranks)
                other_mrd = mean_rank_deficit(baseline_ranks[other_indexes],other_ranks)

                print(class_mrd,other_mrd)
                print(c+' '+str(count)+' '+str(len(class_indexes))+' '+str(error)+' '+str(1-error))
            
                srd_score = class_mrd - other_mrd
                cell_srd[bbx] = [srd_score, class_mrd, other_mrd]

                del model
                gc.collect()
                print("time : ", time.time()-start)

            # Dump SRD
            with open(os.path.join(experiment_path,'srd_grid_4x4_'+c+'_'+layer+'.pkl'), 'wb') as handle:
                pickle.dump([cell_srd], handle, protocol=pickle.HIGHEST_PROTOCOL)

            # # Create class template RDM
            # class_template_RDM = np.ones((len(im_valid_test),len(im_valid_test)))
            # for row in range(len(im_valid_test)):
            #     for column in range(len(im_valid_test)):
            #         if ((row in class_indexes) and (column in class_indexes)) or (row == column):
            #             class_template_RDM[row][column] = 0

            # # Semantic scores for each cell calculation
            # print ('calculating semantic score for each cell')
            # srs_result = {}
            # # cell_probe = {}
            # for bbx in range(len(cell)):
            #     start = time.time() 
            #     loc_new = units_in_cells[bbx]
            #     print('Cell: ', bbx,'Units: ',len(loc_new))

            #     if len(loc_new) == 0.:
            #         srs_result[bbx] = [0,0,0,0]
            #         continue

            #     # All images, only units in cell
            #     act = unit_activations[layer][:,loc_new]
            #     cell_RDM_pearson = 1 - np.corrcoef(act)
            #     cell_RDM_euclidean = euclidean_rdm(act)    
            #     srs_result[bbx]    = [stats.kendalltau(cell_RDM_pearson,class_template_RDM)[0],
            #                         stats.kendalltau(cell_RDM_euclidean,class_template_RDM)[0],
            #                         stats.spearmanr(cell_RDM_pearson,class_template_RDM,axis=None)[0],
            #                         stats.spearmanr(cell_RDM_euclidean,class_template_RDM,axis=None)[0]]
            #     print (srs_result[bbx])
            #     print("time : ", time.time()-start)

                # # Fit classifier to 200 groups of 100 random neurons and record losses
                # losses = []
                # start = time.time()
                # for _ in range(200):
                #     clf = LogisticRegression(penalty=None,random_state=0)
                #     random_indexes = np.random.choice(np.arange(act.shape[1]),
                #                                         size=min(act.shape[1],100),
                #                                         replace=False)
                #     clf.fit(act[:,random_indexes],y)
                #     losses.append(log_loss(y,clf.predict(act[:,random_indexes])))
                
                # cell_probe[bbx] = [np.mean(losses)]
                # print(np.mean(losses),time.time()-start)

            # with open(os.path.join(experiment_path,'rsa_grid_4x4_'+c+'_'+layer+'.pkl'), 'wb') as handle:
            #     pickle.dump([srs_result], handle, protocol=pickle.HIGHEST_PROTOCOL)
            # with open(os.path.join(experiment_path,'linear_probe_grid_4x4_'+c+'_'+layer+'.pkl'), 'wb') as handle:
            #         pickle.dump([cell_probe], handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()