import cv2
import numpy as np
import argparse
import time
import sys

def get_batches(x,y, batch_size):
    batches = []
    print(x.shape[0].__class__)
    for i in range(0, int(x.shape[0]/batch_size)):
        batches.append((x[i*batch_size:(i+1)*batch_size,:,:,:], y[i*batch_size:(i+1)*batch_size]))
    return batches

def load_data(npz='data/cifar_normalized.npz'):
    f = np.load(npz)
    x_train = []#f['x_train']
    y_train = []#f['y_train']
    x_test  = f['x_test']
    y_test  = f['y_test']

    return (x_train, y_train), (x_test, y_test)


def test(net, X, y, batch_size, train_or_test):
    print(batch_size.__class__)
    batches = get_batches(X, y, batch_size)
    correct_test = 0
    cum_t = 0.0
    total_time = 0
    fmt_str = 'Evaluation [%s]. Batch %d/%d (%d%%). Speed = %.2f sec/b, %.2f img/sec. Batch_precision = %.2f'
    for i, (data, label) in enumerate(batches):
        # input_data = data.transpose(0,2,3,1)
        # blob = cv2.dnn.blobFromImages(input_data, 1, (32,32), (0,0,0))
        net.setInput(data)
        t0 = time.time()
        preds = net.forward()
        t1 = time.time()
        duration = t1 - t0
        cum_t += duration
        sec_per_batch = duration
        img_per_sec   = batch_size/duration
        correct_percent = ((np.argmax(preds, axis=1)==np.argmax(label, axis=1)).sum())*100/batch_size
        correct_test += correct_percent
        if cum_t > 0.5:
            sys.stdout.write('\r' + fmt_str %(
                train_or_test, 
                i + 1, 
                len(batches),
                int((i+1)*100/len(batches)),
                sec_per_batch,
                img_per_sec,
                correct_percent
            ))
            sys.stdout.flush()
            cum_t = 0.0
    sys.stdout.write('\r' + fmt_str %(
        train_or_test, 
        i + 1, 
        len(batches),
        int((i+1)*100/len(batches)),
        sec_per_batch,
        img_per_sec,
        correct_percent
    ))
    sys.stdout.write('\n\n%s  Precision = %.2f.\n' % (
    train_or_test,
    correct_test/float(len(batches))
    ))
    # print("Average Frames Per Second : %.2f over %d images\n\n" % (((i+1)*batch_size)/(total_time), (i+1)*batch_size))

def input_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--prototxt", required=True, help="path to caffe 'deploy' prototxt file")
    ap.add_argument("-m", "--model", required=True, help="path to caffe pre-trained model")
    ap.add_argument("-d", "--data", required=True, help="path to normalized input image npz file")
    ap.add_argument("-b", "--batch_size", required=True, help="batch_size for evaluation")
    args = vars(ap.parse_args())
    return args


def main():
    args = input_arguments()
    net = cv2.dnn.readNetFromCaffe(args['prototxt'], args['model'])
    (x_train, y_train), (x_test, y_test) = load_data(npz=args['data'])
    print("==========================================>")
    print("Evaluating on Testing Data")
    test(net, x_test, y_test, int(args['batch_size']), 'TEST DATA')

if __name__ == "__main__":
    main()
