from __future__ import absolute_import, division, print_function, unicode_literals

import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix
from sklearn.preprocessing import label_binarize
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from utils import load_img
from cnn_model import CNN_Model
from argparse import ArgumentParser

""" python train_multiclass.py --model 'axco' --num_epochs 10 --iteration 1 --result_path './results/' --check_path './saved_model/' """

def parse_args():
    """ Take arguments from user inputs."""
    parser = ArgumentParser(description='Multiclass Classification')
    #parser.add_argument('--data_path', help='Directory path for data', 
     #       default='./data/S_M_C/',  type=str)
    parser.add_argument('--dataset', help='Dataset: fourClass',
            default='fourClass', type=str)
    parser.add_argument('--model', help='Model: ax, co, sa, axco, axsa, cosa, axcosa',
            default='axcosa', type=str)
    parser.add_argument('--num_epochs', help='Maximum number of training epochs',
            default=30, type=int) 
    parser.add_argument('--batch_size', help='Batch size',
            default=32, type=int)
    parser.add_argument('--iteration', help='Number of iteration default 30',
            default=30, type=int)
    parser.add_argument('--result_path', help='Path of Result: AUC, tpr, fpr, etc',
            default='./results/', type=str)
    parser.add_argument('--check_path', help='Path of Model Checkpoint',
            default='./saved_model/', type=str)

    args = parser.parse_args()
    return args

train_loss = tf.keras.metrics.Mean(name='train_loss')
test_loss = tf.keras.metrics.Mean(name='test_loss')


def train_step(model, x, y, optimizer, loss_fn):
    with tf.GradientTape() as tape:
        pred = model(x)
        loss = loss_fn(y, pred) + tf.math.add_n(model.losses)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    train_loss(loss)
    return pred

def test_step(model, x, y, loss_fn):
    pred = model(x)
    loss = loss_fn(y, pred)

    test_loss(loss)
    return pred
    
def main(idx, imgs, epochs, batch_size, label, d, mode, result_path , multiclass=False):
    # Train and Test Data Split
    x_train, x_test, y_train, y_test = train_test_split(imgs, label, train_size=0.8, stratify=label, shuffle=True, random_state=idx)
    print(x_train.shape)
    print(x_test.shape)

    test_ds = tf.data.Dataset.from_tensor_slices((x_test,y_test)).batch(batch_size)
    del(x_test)
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(x_train.shape[0]).batch(batch_size)
    del(x_train)

    # Model
    model = CNN_Model(mode=mode, multiclass=multiclass)
    optimizer = tf.keras.optimizers.Adam()
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    
    # Default 30 epochs
    for epoch in range(epochs):
        # Training
        train_preds = []
        train_y = []
        for x_batch, y_batch in train_ds:
            train_pred = train_step(model, x_batch, y_batch, optimizer, loss_fn)
            train_preds.extend(train_pred)
            train_y.extend(y_batch.numpy())

        train_preds = np.array(train_preds)
        train_y = np.array(train_y)

        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        # Train Micro Average AUC
        fpr["micro"], tpr["micro"], _ = roc_curve(train_y.ravel(), train_preds.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        train_AUC = roc_auc['micro']

        # Test
        preds = []
        y = []
        for x_batch, y_batch in test_ds:
            pred = test_step(model, x_batch, y_batch, loss_fn)
            preds.extend(pred)
            y.extend(y_batch.numpy())
        
        preds = np.array(preds)
        y = np.array(y)

        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        # Test Micro Average AUC
        fpr["micro"], tpr["micro"], _ = roc_curve(y.ravel(), preds.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        auc_val = roc_auc['micro']

        template = 'Epoch: {}, Loss: {}, Train AUC: {}, Test Loss: {}, Test AUC: {}'
        print(template.format(epoch+1,
                                train_loss.result(),
                                train_AUC*100,
                                test_loss.result(),
                                auc_val*100))
        
        # Save Train AUC according to epoch
        t1 = open(result_path + d + '/' + mode + "/train.txt", 'a')
        t1.write('{} '.format(train_AUC))
        # Save Test AUC according to epoch
        t2 = open(result_path + d + '/' + mode + "/test.txt", 'a')
        t2.write('{} '.format(auc_val))

        final_fpr = fpr['micro']
        final_tpr = tpr['micro']

        train_loss.reset_states()
        train_AUC = .0
        test_loss.reset_states()
        
    # Train
    preds_train = []
    y_train = []
    for x_batch, y_batch in train_ds:
        pred = test_step(model, x_batch, y_batch, loss_fn)
        preds_train.extend(pred)
        y_train.extend(y_batch.numpy())

    preds_train = np.array(preds_train)
    y_train = np.array(y_train)

    t1.write('\n')
    t1.close()
    t2.write('\n')
    t2.close()

    # Save Final AUC 
    f = open(result_path + d + '/' + mode + "/auc.txt", 'a')
    f.write('{}\n'.format(auc_val))
    f.close()
    # Save Final TPR, FPR
    f = open(result_path + d + '/' + mode + "/tpr.txt", 'a')
    for i in range(final_tpr.shape[0]-1):
        f.write('{}\t'.format(final_tpr[i]))
    f.write('{}\n'.format(final_tpr[-1]))
    f.close()

    f = open(result_path + d + '/' + mode + "/fpr.txt", 'a')
    for i in range(final_fpr.shape[0]-1):
        f.write('{}\t'.format(final_fpr[i]))
    f.write('{}\n'.format(final_fpr[-1]))
    f.close()

    # Save Final Sensitivity & Specificity & Confusion Matrix


    matrix = confusion_matrix(np.argmax(y_train, axis=1), np.argmax(preds_train, axis=1))
    sns.heatmap(matrix, annot=True, cmap="YlGnBu")
    fig = plt.gcf()
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.title("eye injery")
    
    plt.tight_layout()
    plt.savefig('COAXSA_train.png')   
    plt.show()
    plt.figure()
    acc=accuracy_score(np.argmax(y_train, axis=1), np.argmax(preds_train, axis=1))  
    # precision = =tp / (tp + fp)
    ps = precision_score(np.argmax(y_train, axis=1), np.argmax(preds_train, axis=1), average='weighted') # average='macro'/'micro!!
    # recall = tp / (tp + fn)
    rc = recall_score(np.argmax(y_train, axis=1), np.argmax(preds_train, axis=1), average='weighted') 
    # F1 = 2 * (precision * recall) / (precision + recall)
    fs = f1_score(np.argmax(y_train, axis=1), np.argmax(preds_train, axis=1), average='weighted') 
    print('acc: {}, ps: {}, rc: {}, fs: {}'.format(acc, ps, rc, fs))
    
    
    matrix = confusion_matrix(np.argmax(y, axis=1), np.argmax(preds, axis=1))
    sns.heatmap(matrix, annot=True, cmap="YlGnBu")
    fig = plt.gcf()
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.title("eye injery")
    plt.tight_layout()
    plt.savefig('COAXSA_test.png')   
    plt.show()

    plt.figure()
    
    
    acc=accuracy_score(np.argmax(y, axis=1), np.argmax(preds, axis=1))  
    # precision = =tp / (tp + fp)
    ps = precision_score(np.argmax(y, axis=1), np.argmax(preds, axis=1), average='weighted') # average='macro'/'micro!!
    # recall = tp / (tp + fn)
    rc = recall_score(np.argmax(y, axis=1), np.argmax(preds, axis=1), average='weighted') 
    # F1 = 2 * (precision * recall) / (precision + recall)
    fs = f1_score(np.argmax(y, axis=1), np.argmax(preds, axis=1), average='weighted') 
    
    print('acc: {}, ps: {}, rc: {}, fs: {}'.format(acc, ps, rc, fs))

    
     
    preds = np.argmax(preds, 1)
    preds = label_binarize(preds, classes=[0,1,2,3])  
    print(confusion_matrix(y.ravel(), preds.ravel()))
    tn, fp, fn, tp = confusion_matrix(y.ravel(), preds.ravel()).ravel()
    sensitivity = tp / (tp+fn)
    specificity = tn / (fp+tn)
    print('Sensitivity: {}, Specificity: {}'.format(sensitivity, specificity))
    
    f = open(result_path + d + '/' + mode + "/sens_spec.txt", 'a')
    f.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(sensitivity, specificity, tn, fp, fn, tp))
    f.close()

    # Model save (Checkpoint)
    model.save_weights(check_path + '{}/{}/{}/my_chekpoint'.format(d, mode, idx+1))
    acc = (tn + tp) / (tn+tp+fn+fp)
    print('Test Accuracy : {}, {} dataset, {} model saved in saved_model/{}/{}/{}/my_checkpoint'.format(acc, d, mode, d, mode, idx+1))

if __name__ == '__main__':
    args = parse_args()


    d = args.dataset
    mode = args.model
    epochs = args.num_epochs
    batch_size = args.batch_size
    iteration = args.iteration # 30
    result_path = args.result_path
    check_path = args.check_path

    if not os.path.exists(result_path + d + '/' + mode):
        os.makedirs(result_path + d + '/' + mode)
    if not os.path.exists(check_path + d + '/' + mode):
        os.makedirs(check_path + d + '/' + mode)

    if 'fourClass' in d:
        multiclass = True
        print('multiclass')
    else: 
        multiclass = False

    print("Start " + d + " dataset !!")
    imgs, label = load_img(d)
    print(label.shape)

    print(" Start " + mode + " !!")
    # seed value
    seed_value = [i for i in range(30)]
    for i in range(iteration):
        ##############################################
        # 1. 'PYTHONHASHSEED'
        import os
        os.environ['PYTHONHASHSEED'] = str(seed_value[i])

        # 2. 'python' built-in pseudo-random generator
        import random
        random.seed(seed_value[i])

        # 3. 'numpy' pseudo-random generator
        np.random.seed(seed_value[i])

        # 4. 'tensorflow'
        tf.random.set_seed(seed_value[i])
        ###############################################
        print("Iteration: %d" %(i+1))
        main(idx=i,
            imgs=imgs, 
            epochs=epochs, 
            batch_size=batch_size, 
            label=label, 
            d=d, 
            mode=mode, 
            result_path=result_path,
            multiclass=multiclass)
        tf.keras.backend.clear_session()