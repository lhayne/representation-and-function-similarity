from __future__ import division

import os
from os.path import dirname
from os.path import join
os.environ['KERAS_BACKEND'] = 'theano'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    
import pickle
import numpy as np

from scipy.io import loadmat
from scipy import stats
from scipy.spatial import distance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, MaxPooling2D, Input, ZeroPadding2D, merge, Lambda
from keras.layers.convolutional import Convolution2D
from keras.models import Model
from keras.optimizers import SGD
from keras import backend as K
from keras.engine import Layer
from keras.layers.core import Lambda
from keras.utils.vis_utils import plot_model
from keras.layers.core import  Lambda
from keras.regularizers import l2
import cv2
import time
import math
import shapely
import gc
import glob
from re import split

K.set_image_dim_ordering('th')


def preprocess_image(image_paths, image_height=224, image_width=224,color_mode='rgb'):
    """resize images to the appropriate dimensions
    :param image_width:
    :param image_height:
    :param image: image
    :return: image
    """
    img_list = []
    
    for im_path in image_paths:
        image = cv2.imread(im_path)
        image = cv2.resize(image, (image_height, image_width))
    
        if color_mode == 'bgr':
            image = image.transpose((2, 0, 1))
        img_list.append(image)
        
    try:
        img_batch = np.stack(img_list, axis=0)
    except:
        print im_path
        raise ValueError('when img_size and crop_size are None, images'
                ' in image_paths must have the same shapes.')
    return img_batch


def crosschannelnormalization(alpha=1e-4, k=2, beta=0.75, n=5, **kwargs):
    """
    This is the function used for cross channel normalization in the original
    Alexnet
    """
    K.set_image_dim_ordering('th')

    def f(X):
        if K.image_dim_ordering()=='tf':
            b, r, c, ch = X.get_shape()
        else:
            b, ch, r, c = X.shape

        half = n // 2
        square = K.square(X)
        scale = k
        if K.image_dim_ordering() == 'th':
            extra_channels = K.spatial_2d_padding(K.permute_dimensions(square, (0, 2, 3, 1)), ((0,0),(half,half)))
            extra_channels = K.permute_dimensions(extra_channels, (0, 3, 1, 2))
            for i in range(n):
                scale += alpha * extra_channels[:, i:i+ch, :, :]
        if K.image_dim_ordering() == 'tf':
            extra_channels = K.spatial_2d_padding(K.permute_dimensions(square, (0, 3, 1, 2)), (half, 0))
            extra_channels = K.permute_dimensions(extra_channels, (0, 2, 3, 1))
            for i in range(n):
                scale += alpha * extra_channels[:, :, :, i:i+int(ch)]
        scale = scale ** beta
        return X / scale


    return Lambda(f, output_shape=lambda input_shape: input_shape, **kwargs)


def splittensor(axis=1, ratio_split=1, id_split=0, **kwargs):
    def f(X):
        div = K.shape(X)[axis] // ratio_split

        if axis == 0:
            output = X[id_split*div:(id_split+1)*div, :, :, :]
        elif axis == 1:
            output = X[:, id_split*div:(id_split+1)*div, :, :]
        elif axis == 2:
            output = X[:, :, id_split*div:(id_split+1)*div, :]
        elif axis == 3:
            output = X[:, :, :, id_split*div:(id_split+1)*div]
        else:
            raise ValueError("This axis is not possible")
        return output

    def g(input_shape):
        output_shape = list(input_shape)
        output_shape[axis] = output_shape[axis] // ratio_split
        return tuple(output_shape)


    return Lambda(f, output_shape=lambda input_shape: g(input_shape), **kwargs)


def AlexNet(img_shape=(3, 227, 227), n_classes=1000, l2_reg=0.,weights_path=None, lambda_mask=None):

    dim_ordering = K.image_dim_ordering()
    if dim_ordering == 'th':
        batch_index = 0
        channel_index = 1
        row_index = 2
        col_index = 3
    if dim_ordering == 'tf':
        batch_index = 0
        channel_index = 3
        row_index = 1
        col_index = 2
        
    
    inputs = Input(img_shape)

    conv_1 = Convolution2D(96, 11, 11, subsample=(4, 4), activation='relu',
                           name='conv_1', W_regularizer=l2(l2_reg))(inputs)

    if lambda_mask is not None:
        conv_1_mask  = np.reshape(lambda_mask[0:290400], (96,55,55))
    else:
        conv_1_mask = np.ones(shape=((96, 55, 55)))
    
    conv_1_mask  = K.variable(conv_1_mask)
    conv_1_lambda = Lambda(lambda x: x * conv_1_mask)(conv_1)

    conv_2 = MaxPooling2D((3, 3), strides=(2, 2))(conv_1_lambda)
    conv_2 = crosschannelnormalization(name="convpool_1")(conv_2)
    conv_2 = ZeroPadding2D((2, 2))(conv_2)
    conv_2 = merge([
        Convolution2D(128, 5, 5, activation="relu", name='conv_2_'+str(i+1),
                      W_regularizer=l2(l2_reg))(
            splittensor(axis=channel_index, ratio_split=2, id_split=i)(conv_2)
        ) for i in range(2)], mode='concat', concat_axis=channel_index, name="conv_2")

    if lambda_mask is not None:
        conv_2_mask  = np.reshape(lambda_mask[290400:477024],(256, 27, 27) )
    else:
        conv_2_mask = np.ones(shape=((256, 27, 27)))
        
    conv_2_mask = K.variable(conv_2_mask)
    conv_2_lambda = Lambda(lambda x: x * conv_2_mask)(conv_2)

    conv_3 = MaxPooling2D((3, 3), strides=(2, 2))(conv_2_lambda)
    conv_3 = crosschannelnormalization()(conv_3)
    conv_3 = ZeroPadding2D((1, 1))(conv_3)
    conv_3 = Convolution2D(384, 3, 3, activation='relu', name='conv_3',
                           W_regularizer=l2(l2_reg))(conv_3)

    if lambda_mask is not None:
        conv_3_mask  = np.reshape(lambda_mask[477024:541920],(384, 13, 13))
    else:
        conv_3_mask = np.ones(shape=((384, 13, 13)))
    
    conv_3_mask = K.variable(conv_3_mask)
    conv_3_lambda = Lambda(lambda x: x * conv_3_mask)(conv_3)

    conv_4 = ZeroPadding2D((1, 1))(conv_3_lambda)
    conv_4 = merge([
        Convolution2D(192, 3, 3, activation="relu", name='conv_4_'+str(i+1),
                      W_regularizer=l2(l2_reg))(
            splittensor(axis=channel_index, ratio_split=2, id_split=i)(conv_4)
        ) for i in range(2)], mode='concat', concat_axis=channel_index, name="conv_4")

    if lambda_mask is not None:
        conv_4_mask  = np.reshape(lambda_mask[541920:606816],(384, 13, 13))
    else:
        conv_4_mask = np.ones(shape=((384, 13, 13)))
        
    conv_4_mask = K.variable(conv_4_mask)
    conv_4_lambda = Lambda(lambda x: x * conv_4_mask)(conv_4)

    conv_5 = ZeroPadding2D((1, 1))(conv_4_lambda)
    conv_5 = merge([
        Convolution2D(128, 3, 3, activation="relu", name='conv_5_'+str(i+1),
                      W_regularizer=l2(l2_reg))(
            splittensor(axis=channel_index, ratio_split=2, id_split=i)(conv_5)
        ) for i in range(2)], mode='concat', concat_axis=channel_index, name="conv_5")

    if lambda_mask is not None:
        conv_5_mask  = np.reshape(lambda_mask[606816:650080],(256, 13, 13))
    else:
        conv_5_mask = np.ones(shape=((256, 13, 13)))
    
    conv_5_mask = K.variable(conv_5_mask)
    conv_5_lambda = Lambda(lambda x: x * conv_5_mask)(conv_5)

    dense_1 = MaxPooling2D((3, 3), strides=(2, 2), name="convpool_5")(conv_5_lambda)

    dense_1 = Flatten(name="flatten")(dense_1)
    dense_1 = Dense(4096, activation='relu', name='dense_1',
                    W_regularizer=l2(l2_reg))(dense_1)

    if lambda_mask is not None:
        dense_1_mask  = np.reshape(lambda_mask[650080:654176],(4096,))
    else:
        dense_1_mask = np.ones(shape=((4096,)))
    
    
    dense_1_mask = K.variable(dense_1_mask)
    dense_1_lambda = Lambda(lambda x: x * dense_1_mask)(dense_1)

    dense_2 = Dropout(0.5)(dense_1_lambda)
    dense_2 = Dense(4096, activation='relu', name='dense_2',
                    W_regularizer=l2(l2_reg))(dense_2)

    if lambda_mask is not None:
        dense_2_mask  = np.reshape(lambda_mask[654176:658272],(4096,))
    else:
        dense_2_mask = np.ones(shape=((4096,)))
    
    dense_2_mask = K.variable(dense_2_mask)
    dense_2_lambda = Lambda(lambda x: x * dense_2_mask)(dense_2)

    dense_3 = Dropout(0.5)(dense_2_lambda)
    if n_classes == 1000:
        dense_3 = Dense(n_classes, name='dense_3',
                        W_regularizer=l2(l2_reg))(dense_3)

    else:
        # We change the name so when loading the weights_file from a
        # Imagenet pretrained model does not crash
        dense_3 = Dense(n_classes, name='dense_3_new',
                        W_regularizer=l2(l2_reg))(dense_3)

    prediction = Activation("softmax", name="softmax")(dense_3)

    model = Model(input=inputs, output=prediction)
    if weights_path:
        model.load_weights(weights_path)

    return model


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
    print x.shape,y.shape
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
        y_bounds = ([y_sorted[int(math.floor((i/y_partitions)*len(y_sorted)))] 
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
            grid_cells.append([x_lower_bound, y_lower_bound, 
                               x_upper_bound, y_upper_bound])
    
    return grid_cells


def get_activations(model, layer, X_batch):
    """
    Code snippet needed to read activation values from each layer of the pre-trained artificial neural networks
    """
    get_activations = K.function([model.layers[0].input, K.learning_phase()], [model.layers[layer].output,])
    #The learning phase flag is a bool tensor (0 = test, 1 = train)
    activations = get_activations([X_batch,0])
    return activations
    

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

def collect_image_activations(model,image_path_list,existing_activation_ids,save_path=None):
    """
    Construct a M x N matrix, where M is the number of images and N the number of
    neurons by collecting activations from all the neurons in the network.
    """

    for j in range(len(image_path_list)):
        wid = image_path_list[j].split('/')[-1].split('.')[0].split('_')[-1]
        if wid in existing_activation_ids:
            continue
        
        im_temp = preprocess_image([image_path_list[j]],227,227, color_mode="bgr")
        print(j,image_path_list[j])
        data = np.array([],dtype=np.float32)

        i = 0
        for layer in model.layers:
            weights = layer.get_weights()
            if len(weights) > 0:
                activations = get_activations(model,i,im_temp)
                # print activations[0].shape
                temp = np.mean(activations[0], axis=0).ravel()
                print (layer.name, len(temp),len(data))
                if layer.name != 'probs':
                    data = np.append(data, temp)
            i += 1
    
        # Save as we go
        if save_path != None:
            if not os.path.isdir(os.path.join(save_path)):
                os.makedirs(os.path.join(save_path))
            with open(os.path.join(save_path,wid+'.pkl'), 'wb') as handle:
                pickle.dump([data], handle, protocol=pickle.HIGHEST_PROTOCOL)

# def collect_image_activations(model,image_path_list,n_neurons,save_path=None):
#     """
#     Construct a M x N matrix, where M is the number of images and N the number of
#     neurons by collecting activations from all the neurons in the network.
#     """

#     for j in range(len(image_path_list)):
#         im_temp = preprocess_image([image_path_list[j]],227,227, color_mode="bgr")
#         print(j,image_path_list[j])
#         data = np.array([],dtype=np.float32)

#         i = 0
#         for layer in model.layers:
#             weights = layer.get_weights()
#             if len(weights) > 0:
#                 activations = get_activations(model,i,im_temp)
#                 # print activations[0].shape
#                 temp = np.mean(activations[0], axis=0).ravel()
#                 print (layer.name, len(temp),len(data))
#                 if layer.name != 'probs':
#                     data = np.append(data, temp)
#             i += 1
#         # unit_activations.append(data)
    
#         # Save as we go
#         if save_path != None:
#             if not os.path.isdir(os.path.join(save_path)):
#                 os.makedirs(os.path.join(save_path))
#             wid = image_path_list[j].split('/')[-1].split('.')[0].split('_')[-1]
#             with open(os.path.join(save_path,wid+'.pkl'), 'wb') as handle:
#                 pickle.dump([data], handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # return unit_activations

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
        return wids

    data_path = '../data/'
    classes = ['schooner','brain_coral','junco_bird','snail','grey_whale','siberian_husky','electric_fan','bookcase','fountain_pen','toaster','balance_beam',
                'school_bus','chainlink_fence','chime','coyote','aircraft_carrier','bubble','jellyfish','marmoset','wall_clock','water_snake','Welsh_springer_spaniel',
                'Arctic_fox','football_helmet','slug','potpie','Pomeranian','Indian_cobra','beach_wagon','Italian_greyhound','European_fire_salamander','chimpanzee',
                'typewriter_keyboard','black_and_gold_garden_spider','tick','toy_terrier','switch','lighter','guillotine','otterhound','boxer','hook','jersey',
                'soap_dispenser','umbrella','tiger_beetle','cash_machine','eel','Blenheim_spaniel','clumber']
    class_wids = ['n04147183','n01917289','n01534433','n01944390','n02066245','n02110185','n03271574','n02870880','n03388183','n04442312','n02777292',
                'n04146614','n03000134','n03017168','n02114855','n02687172','n09229709','n01910747','n02490219','n04548280','n01737021','n02102177','n02120079','n03379051',
                'n01945685','n07875152','n02112018','n01748264','n02814533','n02091032','n01629819','n02481823','n04505470','n01773157','n01776313','n02087046','n04372370',
                'n03666591','n03467068','n02091635','n02108089','n03532672','n03595614','n04254120','n04507155','n02165105','n02977058','n02526121','n02086646','n02101556']
    method = 'random_10'
    
    # Using command line to process classes selectively
    args = sys.argv[1:]
    if len(args)!=0 and args[0] == '-class_to_resume':
        class_to_resume = args[1]
    else:
        class_to_resume = classes[0]

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

    #Load the ground truth labels to measure the performance
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

    # Get list of all animate and inanimate images
    im_valid_test = glob.glob(data_path+'images/*')
    im_valid_test = np.asarray(im_valid_test)

    # Make list of wids
    true_valid_wids = []
    for i in im_valid_test:
        temp1 = i.split('/')[-1]
        temp = temp1.split('.')[0].split('_')[-1]
        true_valid_wids.append(truth[int(temp)][1])
    true_valid_wids = np.asarray(true_valid_wids)


    # Load AlexNet
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model = AlexNet(weights_path=data_path+"alexnet_weights.h5",
                    lambda_mask=np.ones(shape=((658272,))))
    model.compile(optimizer=sgd, loss='mse')

    # If activations for all neurons for all images exist, grab those
    # Average them across images in each category and save them
    if os.path.isdir(os.path.join(data_path,'activations')):
        existing_activations = os.listdir(os.path.join(data_path,'activations'))
        existing_activation_ids = [split('\.',di)[0] for di in existing_activations]
    
    if len(existing_activations)<len(im_valid_test):
        collect_image_activations(model,im_valid_test,existing_activation_ids,
                                    os.path.join(data_path,'activations'))

    unit_activations = []
    for im in im_valid_test:
        wid = im.split('/')[-1].split('.')[0].split('_')[-1]
        with open(os.path.join(data_path,'activations',wid+'.pkl'), 'rb') as f:
            unit_activations.append(pickle.load(f)[0])

    unit_activations = np.row_stack(unit_activations)
    print(unit_activations.shape)

    # Calculate Baseline Ranks
    if os.path.isfile(data_path+'experiments'+'/baseline_ranks.pkl'):
        print 'opening baseline ranks file'
        with open(data_path+'experiments'+'/baseline_ranks.pkl', 'rb') as f:
            baseline_ranks = pickle.load(f)[0]
    else:
        print 'calculating baseline ranks'
        im_temp = preprocess_image(im_valid_test,227,227, color_mode="bgr")
        out = model.predict(im_temp,batch_size=64)

        predicted_valid_wids = []
        for i in range(len(im_valid_test)):
            predicted_valid_wids.append(pprint_output(out[i],1000))
        predicted_valid_wids = np.asarray(predicted_valid_wids)

        # Count errors and save baseline ranks
        count, error  = top5accuracy(true_valid_wids, predicted_valid_wids)
        baseline_ranks  = np.asarray(get_ranks(true_valid_wids,predicted_valid_wids))
        
        with open(data_path+'experiments'+'/baseline_ranks.pkl', 'wb') as handle:
            pickle.dump([baseline_ranks], handle, protocol=pickle.HIGHEST_PROTOCOL)

        print('baseline '+str(count)+' '+str(len(true_valid_wids))+' '+str(error)+' '+str(1-error))

    gc.collect()
    del model


    # For each class
        # For each layer
            # Calculate Selectivity and Magnitude
            # Grid the space
            # For each cell
                # Calculate SRD
                # Calculate SRS
    class_to_resume_seen = False
    for class_idx,c in enumerate(classes):
        if not class_to_resume_seen:
            if c == class_to_resume:
                class_to_resume_seen = True
            else:
                continue
        print 'calculating class activations'
        class_indexes = [idx for idx in range(len(true_valid_wids)) if true_valid_wids[idx]==class_wids[class_idx]]
        other_indexes = [idx for idx in range(len(true_valid_wids)) if true_valid_wids[idx]!=class_wids[class_idx]]
        if method=='random_10': # Choose 10 random classes to compare against
            random_10_classes = np.random.choice([idx for idx,cl in enumerate(classes) if cl!=c],size=10,replace=False)
            other_indexes = [idx for idx in other_indexes if true_valid_wids[idx] in np.asarray(class_wids)[random_10_classes]]
            print 'other_indexes', len(other_indexes), other_indexes
        class_activations = np.mean(unit_activations[class_indexes],axis=0)
        other_activations = np.mean(unit_activations[other_indexes],axis=0)

        X = np.column_stack((class_activations,other_activations))     

        # Layer starting and ending indexes for each layer of AlexNet in lambda mask
        layer_indexes = [0,290400,477024,541920,606816,650080,654176,658272]

        for layer_idx,layer in enumerate(['conv_1','conv_2','conv_3','conv_4','conv_5','dense_1','dense_2']):
            print c,layer

            # Magnitude and selectivity on a per layer basis
            magnitude = activation_to_magnitude(X[layer_indexes[layer_idx]:layer_indexes[layer_idx+1]])
            selectivity = activation_to_selectivity(X[layer_indexes[layer_idx]:layer_indexes[layer_idx+1]])

            print 'generating grid'
            x_partitions,y_partitions = 4,4
            cell = grid_space(magnitude,selectivity,x_partitions=x_partitions,y_partitions=y_partitions)

            with open(data_path+'experiments'+'/grid_specifications_4x4_'+c+'_'+layer+'.pkl', 'wb') as handle:
                pickle.dump([cell], handle, protocol=pickle.HIGHEST_PROTOCOL)

            # Don't have access to geopandas in python 2.7, so we're brute forcing unit assignments to cells.
            units_in_cells = dict()
            for cell_index in range(len(cell)):
                units_in_cells[cell_index] = []
                for unit_index in range(len(magnitude)):
                    network_unit_index = unit_index + layer_indexes[layer_idx] # Offset to get back the overall network index
                    if (magnitude[unit_index] >= cell[cell_index][0] and
                        selectivity[unit_index] >= cell[cell_index][1] and
                        magnitude[unit_index] <= cell[cell_index][2] and
                        selectivity[unit_index] <= cell[cell_index][3]):
                        units_in_cells[cell_index].append(network_unit_index)
                print len(units_in_cells[cell_index])

            with open(data_path+'experiments'+'/units_in_cells_4x4_'+c+'_'+layer+'.pkl', 'wb') as handle:
                pickle.dump([units_in_cells], handle, protocol=pickle.HIGHEST_PROTOCOL)

            # for each cell in grid
            print 'calculating rank deficits for each cell'
            cell_srd = {}
            for bbx in range(len(cell)):
                start = time.time()
                
                # Query indices of units in that cell, create mask and set activations to zero
                loc_new = units_in_cells[bbx]
                lambda_mask = np.ones(shape=((658272,)))

                lambda_mask[loc_new] = 0.
                print('Cell: ', bbx, ' Units: ', len(loc_new))

                sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
                model = AlexNet(weights_path=data_path+"alexnet_weights.h5",lambda_mask=lambda_mask)
                model.compile(optimizer=sgd, loss='mse')                            
                
                im_temp = preprocess_image(im_valid_test[class_indexes],227,227, color_mode="bgr")
                out = model.predict(im_temp,batch_size=64)

                predicted_valid_wids = []
                for i in range(len(im_valid_test[class_indexes])):
                    predicted_valid_wids.append(pprint_output(out[i],1000))
                predicted_valid_wids = np.asarray(predicted_valid_wids)

                # calculate ranks
                count, error  = top5accuracy(true_valid_wids[class_indexes], predicted_valid_wids)
                class_ranks  = get_ranks(true_valid_wids[class_indexes],
                                        predicted_valid_wids)
                class_mrd = mean_rank_deficit(baseline_ranks[class_indexes],class_ranks)

                print(class_mrd)
                print(c+' '+str(count)+' '+str(len(class_indexes))+' '+str(error)+' '+str(1-error))
            
                cell_srd[bbx] = [class_mrd]

                gc.collect()
                del model
                print("time : ", time.time()-start)

            # Dump SRD
            with open(data_path+'experiments'+'/cell_srd_grid_4x4_'+c+'_'+layer+'.pkl', 'wb') as handle:
                pickle.dump([cell_srd], handle, protocol=pickle.HIGHEST_PROTOCOL)

            # Create class template RDM
            class_template_RDM = np.ones((len(class_indexes)+len(other_indexes),
                                          len(class_indexes)+len(other_indexes)))
            for row in range(len(class_indexes)+len(other_indexes)):
                for column in range(len(class_indexes)+len(other_indexes)):
                    if ((row < len(class_indexes)) and (column < len(class_indexes))) or (row == column):
                        class_template_RDM[row][column] = 0

            # Semantic scores for each cell calculation
            print ('calculating semantic score for each cell')
            srs_result = {}
            cell_probe = {}
            for bbx in range(len(cell)):
                start = time.time() 
                loc_new = units_in_cells[bbx]
                print('Cell: ', bbx,'Units: ',len(loc_new))

                # All images, only units in cell
                act = np.asarray(unit_activations)[class_indexes+other_indexes][:,loc_new]
                print act.shape
                # Labels correspond to class indexes
                # y = np.asarray([1 if i in class_indexes else 0 for i in range(len(unit_activations))]) 
                y = list(np.ones(len(class_indexes)))+list(np.zeros(len(other_indexes)))
                print len(y)
                cell_RDM_pearson = 1 - np.corrcoef(act)
                # cell_RDM_euclidean = euclidean_rdm(act)    
                # srs_result[bbx]    = [stats.kendalltau(cell_RDM_pearson,class_template_RDM)[0],
                #                     stats.kendalltau(cell_RDM_euclidean,class_template_RDM)[0],
                #                     stats.spearmanr(cell_RDM_pearson,class_template_RDM,axis=None)[0],
                #                     stats.spearmanr(cell_RDM_euclidean,class_template_RDM,axis=None)[0]]
                srs_result[bbx]    = [stats.kendalltau(cell_RDM_pearson,class_template_RDM)[0],
                                    stats.spearmanr(cell_RDM_pearson,class_template_RDM,axis=None)[0]]
                print (srs_result[bbx])

                # Fit classifier to max(int(act.shape[1]/100),200) groups of 100 random neurons and record losses
                # to set expected number of times each neuron chosen to 1
                losses = []
                for _ in range(200):
                    clf = LogisticRegression(random_state=0)
                    random_indexes = np.random.choice(np.arange(act.shape[1]),
                                                        size=min(act.shape[1],100),
                                                        replace=False)
                    clf.fit(act[:,random_indexes],y)
                    losses.append(log_loss(y,clf.predict(act[:,random_indexes])))
                
                cell_probe[bbx] = [np.mean(losses)]
                print ('decoding accuracy',np.exp(-np.mean(losses)))
                print("time : ", time.time()-start)

            with open(os.path.join(data_path,'experiments','srs_result_grid_4x4_'+c+'_'+layer+'.pkl'), 'wb') as handle:
                pickle.dump([srs_result], handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(os.path.join(data_path,'experiments','cell_linear_probe_grid_4x4_'+c+'_'+layer+'.pkl'), 'wb') as handle:
                    pickle.dump([cell_probe], handle, protocol=pickle.HIGHEST_PROTOCOL)



if __name__ == "__main__":
    main()