import numpy      as np
import tensorflow as tf

# Config
data_size  = 1000
test_size  = int(data_size/3)
epoch      = 5000
batch_size = 100
class_size = 10

np.random.seed(1)

def PrepareData(size):
    x   = np.asarray([[np.random.random()] for x in range(size)], dtype=np.float32)
    #print(x)
    t = []
    for y in x:
        tmp = np.zeros(class_size)
        tmp[int(y[0]*class_size)]=1
        t.append(tmp)
    y_ = np.asarray(t, dtype=np.int64)
    #print(y_)
    train_x = x[:test_size]
    #print(len(train_x))
    test_x  = x[test_size:]
    #print(len(test_x))
    train_y = y_[:test_size]
    #print(len(train_y))
    test_y  = y_[test_size:]
    #print(len(test_y))
    return (train_x, train_y, test_x, test_y)

def next_batch(train_x, train_y):
    np.random.seed()
    idx = np.random.choice([x for x in range(len(train_x))], size=batch_size)
    batch_x = []
    batch_y = []
    for i in idx:
        batch_x.append(train_x[i])
        batch_y.append(train_y[i])

    return (np.asarray(batch_x, dtype=np.float32), np.asarray(batch_y, dtype=np.int64))

# Prepare Data
train_x, train_y, test_x, test_y = PrepareData(data_size)
#print(train_x)
#print(train_y)
#exit()

batch_x, batch_y = next_batch(train_x, train_y)
#print(batch_x)
#print(batch_y)
#exit()

def inference(input_placeholder):
    # weight
    wh  = tf.Variable(tf.random_normal([1, 50], mean=0.0, stddev=0.05))
    wo  = tf.Variable(tf.random_normal([50,  class_size], mean=0.0, stddev=0.05))
    # bias
    bh  = tf.Variable(tf.zeros([50]))
    bo  = tf.Variable(tf.zeros([class_size]))
    # output
    h   = tf.nn.sigmoid(tf.matmul(input_placeholder, wh)+bh)
    y   = tf.nn.softmax(tf.matmul(h, wo)+bo)

    return (y)

def loss(output, supervisor_labels_placeholder):
    # loss function
    cross_entropy = -tf.reduce_sum(supervisor_labels_placeholder * tf.log(tf.clip_by_value(output,1e-10,1.0)))

    return (cross_entropy)

def training(loss):
    # optimize
    train_step = tf.train.AdamOptimizer(0.01).minimize(loss)

    return (train_step)

def evaluate(output, y):
    correct_prediction = tf.equal(tf.argmax(output,1), tf.argmax(y,1))

    return (correct_prediction)

def accuracy(correct_prediction):
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    return (accuracy)

def main(argv):
    # input data
    input_placeholder = tf.placeholder("float", [None, 1])
    # target data
    supervisor_labels_placeholder = tf.placeholder("float", [None, class_size])
    # feed_dict (No Batch)
    #feed_dict     = {input_placeholder: train_x, supervisor_labels_placeholder: train_y}

    # train
    sess = tf.Session()

    output        = inference(input_placeholder)
    cross_entropy = loss(output, supervisor_labels_placeholder)
    training_op   = training(cross_entropy)
    eval          = evaluate(output, supervisor_labels_placeholder)
    acc           = accuracy(eval)

    # initialize
    init = tf.global_variables_initializer()

    sess.run(init)

    print("Training...")
    for i in range(epoch):
        for j in range(int(len(train_x)/batch_size)):
            # batch
            batch_x, batch_y = next_batch(train_x, train_y)
            #print(batch_x)
            # feed_dict
            feed_dict     = {input_placeholder: batch_x, supervisor_labels_placeholder: batch_y}
            sess.run(training_op, feed_dict=feed_dict)

        if (i%100==0):
            #loss
            feed_dict     = {input_placeholder: train_x, supervisor_labels_placeholder: train_y}
            train_loss     = sess.run(cross_entropy, feed_dict=feed_dict)
            # evaluate
            train_accuracy = sess.run(acc, feed_dict=feed_dict)
            print("    step %6d: accuracy=%6.3f, loss=%6.3f" % (i, train_accuracy, train_loss))

    # test
    print("Test...")
    feed_dict     = {input_placeholder: test_x, supervisor_labels_placeholder: test_y}
    print("    accuracy = ", sess.run(acc, feed_dict=feed_dict))

if (__name__ == "__main__"):
    tf.app.run()
