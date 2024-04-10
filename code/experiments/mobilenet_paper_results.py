import numpy as np
from scoring import *
from dist_metrics import *
import pickle
import os
import glob
import pandas as pd
from scipy.io import loadmat
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import sys

import warnings
warnings.filterwarnings("ignore")

def get_activation_matrix(activation_path,start_index=0,end_index=-0):
    activation_files = glob.glob(os.path.join(activation_path,'*'))
    activation_files = np.sort(activation_files)
    activations = []
    for f in activation_files:
        with open(f,'rb') as f:
            activations.append(pickle.load(f)[0][start_index:end_index])
    return np.row_stack(activations)

def decoding_accuracy(activations,y,iterations=200,neurons=100):
    losses = []
    for _ in range(iterations):
        clf = LogisticRegression(random_state=0)
        random_indexes = np.random.choice(np.arange(activations.shape[1]),
                                            size=min(activations.shape[1],neurons),
                                            replace=False)
        clf.fit(activations[:,random_indexes],y)
        losses.append(log_loss(y,clf.predict(activations[:,random_indexes])))

    mean_loss = np.mean(losses)
    return np.exp(-mean_loss)

def main():
    classes = ['schooner','brain_coral','junco_bird','snail','grey_whale','siberian_husky','electric_fan','bookcase','fountain_pen','toaster']
    class_wids = ['n04147183','n01917289','n01534433','n01944390','n02066245','n02110185','n03271574','n02870880','n03388183','n04442312']

    data_path = '/projects/luha5813/data/representations/'
    model_name = 'mobilenetv2'
    experiment_path = '/projects/luha5813/10_classes/experiments/'
    layers = ['Conv1', 'bn_Conv1', 'expanded_conv_depthwise', 'expanded_conv_depthwise_BN', 
                    'expanded_conv_project', 'expanded_conv_project_BN', 'block_1_expand', 'block_1_expand_BN', 
                    'block_1_depthwise', 'block_1_depthwise_BN', 'block_1_project', 'block_1_project_BN', 
                    'block_2_expand', 'block_2_expand_BN', 'block_2_depthwise', 'block_2_depthwise_BN', 
                    'block_2_project', 'block_2_project_BN', 'block_3_expand', 'block_3_expand_BN', 
                    'block_3_depthwise', 'block_3_depthwise_BN', 'block_3_project', 'block_3_project_BN', 
                    'block_4_expand', 'block_4_expand_BN', 'block_4_depthwise', 'block_4_depthwise_BN', 
                    'block_4_project', 'block_4_project_BN', 'block_5_expand', 'block_5_expand_BN', 
                    'block_5_depthwise', 'block_5_depthwise_BN', 'block_5_project', 'block_5_project_BN', 
                    'block_6_expand', 'block_6_expand_BN', 'block_6_depthwise', 'block_6_depthwise_BN', 
                    'block_6_project', 'block_6_project_BN', 'block_7_expand', 'block_7_expand_BN', 
                    'block_7_depthwise', 'block_7_depthwise_BN', 'block_7_project', 'block_7_project_BN', 
                    'block_8_expand', 'block_8_expand_BN', 'block_8_depthwise', 'block_8_depthwise_BN', 
                    'block_8_project', 'block_8_project_BN', 'block_9_expand', 'block_9_expand_BN', 
                    'block_9_depthwise', 'block_9_depthwise_BN', 'block_9_project', 'block_9_project_BN', 
                    'block_10_expand', 'block_10_expand_BN', 'block_10_depthwise', 'block_10_depthwise_BN', 
                    'block_10_project', 'block_10_project_BN', 'block_11_expand', 'block_11_expand_BN', 
                    'block_11_depthwise', 'block_11_depthwise_BN', 'block_11_project', 'block_11_project_BN', 
                    'block_12_expand', 'block_12_expand_BN', 'block_12_depthwise', 'block_12_depthwise_BN', 
                    'block_12_project', 'block_12_project_BN', 'block_13_expand', 'block_13_expand_BN', 
                    'block_13_depthwise', 'block_13_depthwise_BN', 'block_13_project', 'block_13_project_BN', 
                    'block_14_expand', 'block_14_expand_BN', 'block_14_depthwise', 'block_14_depthwise_BN', 
                    'block_14_project', 'block_14_project_BN', 'block_15_expand', 'block_15_expand_BN', 
                    'block_15_depthwise', 'block_15_depthwise_BN', 'block_15_project', 'block_15_project_BN', 
                    'block_16_expand', 'block_16_expand_BN', 'block_16_depthwise', 'block_16_depthwise_BN', 
                    'block_16_project', 'block_16_project_BN', 'Conv_1', 'Conv_1_bn']

    layer_indexes = [0, 200704, 401408, 602112, 802816, 903168, 1003520, 1605632, 2207744, 2358272, 2508800, 
    2533888, 2558976, 2709504, 2860032, 3010560, 3161088, 3186176, 3211264, 3361792, 3512320, 3549952, 3587584, 
    3600128, 3612672, 3687936, 3763200, 3838464, 3913728, 3926272, 3938816, 4014080, 4089344, 4164608, 4239872, 
    4252416, 4264960, 4340224, 4415488, 4434304, 4453120, 4457824, 4462528, 4490752, 4518976, 4547200, 4575424, 
    4580128, 4584832, 4613056, 4641280, 4669504, 4697728, 4702432, 4707136, 4735360, 4763584, 4791808, 4820032, 
    4824736, 4829440, 4857664, 4885888, 4914112, 4942336, 4948608, 4954880, 4992512, 5030144, 5067776, 5105408, 
    5111680, 5117952, 5155584, 5193216, 5230848, 5268480, 5274752, 5281024, 5318656, 5356288, 5365696, 5375104,
    5377848, 5380592, 5397056, 5413520, 5429984, 5446448, 5449192, 5451936, 5468400, 5484864, 5501328, 5517792, 
    5520536, 5523280, 5539744, 5556208, 5572672, 5589136, 5594624, 5600112, 5662832, 5725552]

    #Load the details of all the 1000 classes and the function to conver the synset id to words
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

    activation_files = glob.glob(os.path.join(data_path,model_name,'activations','*'))
    activation_files = np.sort(activation_files)

    # Make list of wids
    true_valid_wids = []
    for i in activation_files:
        temp1 = i.split('/')[-1]
        temp = temp1.split('.')[0]
        true_valid_wids.append(truth[int(temp)][1])
    true_valid_wids = np.asarray(true_valid_wids)

    # Using command line to process classes selectively
    args = sys.argv[1:]
    if len(args)!=0 and args[0] == '-class':
        classes_to_process = [args[1]]

    if len(args)!=0 and args[2] == '-layer_start':
        layer_start = [args[3]]

    
    for class_index,c in enumerate(classes):
        if c not in classes_to_process:
            continue

        results = pd.DataFrame([],columns=['class','layer','tile',
                                            'decoding_accuracy_delta',
                                            'ablation_impact',
                                            'cka','procrustes','pwcca','mean_sq_cca_corr','mean_cca_corr'])

        class_indexes = [idx for idx in range(len(true_valid_wids)) if true_valid_wids[idx]==class_wids[class_index]]
        # Labels correspond to class indexes
        y = np.asarray([1 if i in class_indexes else 0 for i in range(500)])

        layer_start_seen = False        

        for layer_index,layer in enumerate(layers):
            if not layer_start_seen:
                if layer in layer_start:
                    layer_start_seen = True
                else:
                    continue
            
            print(c,layer)
            # Load entire layer's activations as layer_activations
            layer_activations = get_activation_matrix(os.path.join(data_path,model_name,'activations'),
                                                    start_index=layer_indexes[layer_index],
                                                    end_index=layer_indexes[layer_index+1])
            print('layer_activations shape ',layer_activations.shape)
            # Center and normalize each activation matrix as in Ding et al. (2021)
            # center each column, so that each neuron representation has mean 0
            layer_activations = layer_activations - layer_activations.mean(axis=0, keepdims=True)
            # normalize each representation (Messes up the linear decoding)
            # layer_activations = layer_activations / np.linalg.norm(layer_activations)
        
            # Compute decoding accuracy for class across entire layer
            layer_decoding_accuracy = decoding_accuracy(layer_activations,y,
                                                        iterations=max(int(layer_activations.shape[1]/100),200),
                                                        neurons=100)
            print('decoding accuracy: ',layer_decoding_accuracy)
            
            # Load units_in_cells dictionary
            with open(os.path.join(experiment_path,model_name,'units_in_cells_'+c+'_'+layer+'.pkl'),'rb') as f:
                units_in_cells = pickle.load(f)[0]
                
            # Load ablation impacts
            with open(os.path.join(experiment_path,model_name,'srd_grid_4x4_'+c+'_'+layer+'.pkl'),'rb') as f:
                ablation_impacts = pickle.load(f)[0]
            
            for tile in units_in_cells.keys():
                # Select all the neurons that weren't ablated in this tile
                ablated_indexes = np.asarray(units_in_cells[tile]) - layer_indexes[layer_index]
                intact_indexes = list(set([i for i in range(layer_activations.shape[1])]) - set(ablated_indexes))
                intact_activations = layer_activations[:,intact_indexes]
                print(layer_activations.shape[1],intact_activations.shape[1])
                
                # Compute decoding accuracy change
                ablated_decoding_accuracy = decoding_accuracy(intact_activations,y,
                                                              iterations=max(int(layer_activations.shape[1]/100),200),
                                                              neurons=100)
                print('ablated accuracy: ',ablated_decoding_accuracy)
                
                # Ablation impact
                ablation_impact = ablation_impacts[tile][1]
                print('ablation impact: ',ablation_impact)
                
                # In Ding et al. (2021) the matrices are neurons x examples, so we need to transpose
                # Compute CKA
                cka_sim = lin_cka_dist_2(layer_activations, 
                                        intact_activations)
                print('cka: ',cka_sim)
                
                # Compute Procrustes
                procrustes_sim = procrustes_2(layer_activations/np.linalg.norm(layer_activations),
                                            intact_activations/np.linalg.norm(intact_activations))
                print('procrustes: ',procrustes_sim)
                
                # Compute PWCCA
                _, cca_rho, _, transformed_rep1, _ = cca_decomp_kernel_trick(
                                                                            layer_activations, 
                                                                            intact_activations,
                                                                            pen_a=1e3,pen_b=1e3,
                                                                            )
                pwcca_sim = pwcca_dist(layer_activations.T, cca_rho, transformed_rep1)
                print('pwcca: ',pwcca_sim)
                
                mean_sq_cca_sim = mean_sq_cca_corr(cca_rho)
                
                mean_cca_sim = mean_cca_corr(cca_rho)
                print('mean_cca_sim',mean_cca_sim)
                
                # Add to dataframe
                results.loc[len(results)] = [c,layer,tile,
                                            layer_decoding_accuracy-ablated_decoding_accuracy,
                                            ablation_impact,
                                            cka_sim,procrustes_sim,pwcca_sim,mean_sq_cca_sim,mean_cca_sim]

                results.to_csv(os.path.join(experiment_path,model_name,'iclr_results_'+c+'_3.csv'))

if __name__ == "__main__":
    main()
