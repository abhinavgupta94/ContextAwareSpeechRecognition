from __future__ import print_function

import re
import numpy as np
import theano
import theano.tensor as T
import lasagne
import cPickle
import time

def get_text(path):
    in_text_file = open(path, 'r')
    in_text = []
    video_features = []
    for line in in_text_file:
        line = line.strip()
        video_hash = line.split()[0]
        line = " ".join([w for w in line.split()[1:]])
        line = re.sub("[^A-Za-z1-9]"," ",line)
        line = " ".join([w for w in line.split() if w.isalpha()])
        line = line.lower().split()
        #video_features += [line]*len(line)
        video_features += [video_hash]*len(line)
        in_text += line
        in_text.append("<eos>")
        #video_features.append(line)
        video_features.append(video_hash)
    in_text = " ".join(in_text)
    in_text = in_text.decode("utf-8-sig").encode("utf-8")
    return in_text,video_features
try:
    #You can also use your own file
    #The file must be a simple text file.
    #Simply edit the file name below and uncomment the line.  
    text_data, video_data = get_text("/data/ASR5/abhinav5/train_360h/text")
except Exception as e:
    print("Please verify the location of the input file/URL.")
    print("A sample txt file can be downloaded from https://s3.amazonaws.com/text-datasets/nietzsche.txt")
    raise IOError('Unable to Read Text')


#This snippet loads the text file and creates dictionaries to 
#encode characters into a vector-space representation and vice-versa. 
ndata = text_data.split()
vocab_set = set(ndata)
words = list(vocab_set)
vocab_size = len(words)
char_to_ix = { ch:i for i,ch in enumerate(words) }
ix_to_char = { i:ch for i,ch in enumerate(words) }

place_features = cPickle.load(open("/data/ASR5/abhinav5/PlacesAlexNet_360h/place_features.p", "rb"))
object_features = cPickle.load(open("/data/ASR5/abhinav5/YTubeV2_360h/object_features.p", "rb"))

total_size = len(ndata)


#Lasagne Seed for Reproducibility
lasagne.random.set_rng(np.random.RandomState(1))

# Sequence Length
SEQ_LENGTH = 1

EMBEDDING_SIZE = 900

# Number of units in the two hidden (LSTM) layers
N_HIDDEN = 1024

# Optimization learning rate
LEARNING_RATE = .01
# All gradients above this will be clipped
GRAD_CLIP = 100

BATCH_SIZE = 128
OBJECT_FEATURES = 100
PLACE_FEATURES = 100
VIDEO_FEATURES = OBJECT_FEATURES + PLACE_FEATURES

# Number of epochs to train the net
NUM_EPOCHS = 14

batches = (SEQ_LENGTH * total_size / BATCH_SIZE) + 1
train_batches = int(0.8 * batches)
valid_batches = batches - train_batches
print(batches, train_batches, valid_batches)

def gen_data(p):

    x = np.zeros((BATCH_SIZE, SEQ_LENGTH))
    y = np.zeros(BATCH_SIZE)
    mask = np.zeros((BATCH_SIZE, VIDEO_FEATURES + SEQ_LENGTH))
    place_f = np.zeros((BATCH_SIZE, SEQ_LENGTH, PLACE_FEATURES))
    object_f = np.zeros((BATCH_SIZE, SEQ_LENGTH, OBJECT_FEATURES))
    count = 0

    while count < BATCH_SIZE:
        
        not_complete = 0

        for i in range(SEQ_LENGTH+1):
            if p + i == len(ndata):
                p = 0
                not_complete = 1
                break

        if not_complete == 1:
            break

        try:
            place_features[video_data[p]]
            object_features[video_data[p]]
        
        except KeyError:
            p+=1
            continue

        for i in range(SEQ_LENGTH):
            x[count,i] = char_to_ix[ndata[p+i]]
            mask[count, :VIDEO_FEATURES+i+1] = 1
        
        y[count] = char_to_ix[ndata[p+SEQ_LENGTH]]
        place_f[count] = (place_features[video_data[p]])*SEQ_LENGTH
        object_f[count] = (object_features[video_data[p]])*SEQ_LENGTH

        count += 1
        p += 1

    return x, y, mask, place_f.astype(theano.config.floatX), object_f.astype(theano.config.floatX), p


def createnetwork(p, no):
    print("Building network ...")
     
    # First, we build the network, starting with an input layer
    # Recurrent layers expect input of shape
    # (batch size, SEQ_LENGTH, num_features)
    symX = T.imatrix()
    mask = T.imatrix()
    symPlace = T.tensor3()
    symObject = T.tensor3()
    l_in = lasagne.layers.InputLayer(shape=(BATCH_SIZE,SEQ_LENGTH),input_var = symX)
    l_mask = lasagne.layers.InputLayer(shape=(BATCH_SIZE,VIDEO_FEATURES + SEQ_LENGTH),input_var = mask)
    l_place = lasagne.layers.InputLayer(shape=(BATCH_SIZE,SEQ_LENGTH,PLACE_FEATURES),input_var = symPlace)
    l_object = lasagne.layers.InputLayer(shape=(BATCH_SIZE,SEQ_LENGTH,OBJECT_FEATURES),input_var = symObject)
    # We now build the LSTM layer which takes l_in as the input layer
    # We clip the gradients at GRAD_CLIP to prevent the problem of exploding gradients.

    l_emb = lasagne.layers.EmbeddingLayer(l_in,input_size = vocab_size,output_size = EMBEDDING_SIZE)
    l_concat = lasagne.layers.ConcatLayer([l_place,l_object,l_emb],axis = 2)

    l_forward_1 = lasagne.layers.LSTMLayer(
        l_concat, N_HIDDEN, grad_clipping=GRAD_CLIP,
        nonlinearity=lasagne.nonlinearities.tanh,mask_input = l_mask,
        )
    
    l_backward_1 = lasagne.layers.LSTMLayer(
        l_concat, N_HIDDEN, grad_clipping=GRAD_CLIP,
        nonlinearity=lasagne.nonlinearities.tanh,mask_input = l_mask,
        backwards=True)
    
    l_concat_1 = lasagne.layers.ConcatLayer([l_forward_1, l_backward_1])
    l_drp1 = lasagne.layers.DropoutLayer(l_concat_1)

    l_forward_2 = lasagne.layers.LSTMLayer(
        l_drp1, N_HIDDEN, grad_clipping=GRAD_CLIP,
        nonlinearity=lasagne.nonlinearities.tanh,mask_input = l_mask,
        )
    
    l_backward_2 = lasagne.layers.LSTMLayer(
        l_drp1, N_HIDDEN, grad_clipping=GRAD_CLIP,
        nonlinearity=lasagne.nonlinearities.tanh,mask_input = l_mask,
        backwards=True)

    l_concat_2 = lasagne.layers.ConcatLayer([l_forward_2, l_backward_2])
    l_drp2 = lasagne.layers.DropoutLayer(l_concat_2)
    
    l_shp = lasagne.layers.SliceLayer(l_drp2, -1, 1)
    
    # The sliced output is then passed through the softmax nonlinearity to create probability distribution of the prediction
    # The output of this stage is (batch_size, vocab_size)
    l_out = lasagne.layers.DenseLayer(l_shp, num_units=vocab_size, nonlinearity=lasagne.nonlinearities.softmax)
    # Theano tensor for the targets
    target_values = T.ivector()
    # The output of the sliced layer will then be of size (batch_size, N_HIDDEN)
    
    # lasagne.layers.get_output produces a variable for the output of the net
    network_output = lasagne.layers.get_output(l_out)
    valid_output = lasagne.layers.get_output(l_out, deterministic=True)
    
    # The loss function is calculated as the mean of the (categorical) cross-entropy between the prediction and target.
    cost = lasagne.objectives.categorical_crossentropy(network_output, target_values).mean()
    valid_cost = lasagne.objectives.categorical_crossentropy(valid_output, target_values).mean()

    #l2_penalty = lasagne.regularization.regularize_layer_params(l_out, lasagne.regularization.l2)
    #l1_penalty = lasagne.regularization.regularize_network_params(l_out, lasagne.regularization.l1)

    # Retrieve all parameters from the network
    all_params = lasagne.layers.get_all_params(l_out,trainable=True)
    sh_lr = theano.shared(lasagne.utils.floatX(LEARNING_RATE))
    # Compute AdaGrad updates for training
    print("Computing updates ...")
    #newcost = cost + 0.01*l2_penalty
    updates = lasagne.updates.adagrad(cost, all_params, sh_lr)
        
    # Theano functions for training and computing cost
    print("Compiling functions ...")
    train = theano.function([l_in.input_var,l_mask.input_var,l_place.input_var,l_object.input_var,target_values], cost, updates=updates, allow_input_downcast=True)
    compute_valid_cost = theano.function([l_in.input_var,l_mask.input_var,l_place.input_var,l_object.input_var,target_values], valid_cost, allow_input_downcast=True)
    # probs = theano.function([l_in.input_var,l_mask.input_var, l_video.input_var],network_output,allow_input_downcast=True)
    
    print("Training ...")
    previous = None
    try:
        
        for it in range(NUM_EPOCHS):
            
            start_time = time.time()

            train_cost = 0
            for _ in range(train_batches):
                
                x,y,m,vp,vo,p = gen_data(p)
                train_cost += train(x,m,vp,vo,y)
                
            valid_cost = 0
            for _ in range(valid_batches):

                valid_x,valid_y,valid_m,valid_vp,valid_vo,p = gen_data(p)
                valid_cost += compute_valid_cost(valid_x,valid_m,valid_vp,valid_vo,valid_y)


            if previous == None:
                previous = valid_cost / valid_batches
            else:
                if valid_cost / valid_batches > previous:
                    current_lr = sh_lr.get_value()
                    if current_lr > 10**-7:
                        sh_lr.set_value(lasagne.utils.floatX(current_lr /2.))
                previous = valid_cost / valid_batches
            
            print("Network {} Epoch {} took {} time".format(no, it, (time.time() - start_time) / 60 ))
            print(np.exp(train_cost / train_batches))
            print(np.exp(valid_cost / valid_batches))
            '''
            x,_,m,v,_ = gen_data(0,features_dict = features_dict,data= valid_in_text,video_features = valid_video_features)
            ix = probs(x,m,v)
            print_string = ix_to_char[x[0][0]] 
            
            for v in ix:
                print_string += " " +  ix_to_char[np.argmax(v.ravel())]
            print(print_string)
            '''
    
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    p1 = 0
    p2 = total_size/5
    p3 = p2*2
    p4 = p2*3
    p5 = p2*4
    l1 = createnetwork(p1,1)
    # l2 = createnetwork(p2,2)
    # l3 = createnetwork(p3,3)
    # l4 = createnetwork(p4,4)
    # l5 = createnetwork(p5,5)
