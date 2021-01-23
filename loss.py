import tensorflow as tf 

def my_loss(y_true, y_pred, log=None):
    y_pred_sig = tf.clip_by_value(tf.sigmoid(y_pred),1e-15,1. -1e-15)
    y_pos_indx = tf.where(tf.greater_equal(y_true, 0.8))
    y_pos_loss1 = - tf.pow(1. - y_pred_sig, 2) * tf.math.log(y_pred_sig)
    y_pos_loss = tf.reduce_sum(tf.gather_nd(y_pos_loss1,y_pos_indx))

    y_neg_indx = tf.where(tf.less(y_true, 0.8))
    y_neg_loss1 = - tf.pow(1.- y_true, 4) * tf.pow(y_pred_sig, 2) * tf.math.log(1.-y_pred_sig)  # / (1. - y_pred_sig)
    y_neg_loss = tf.reduce_sum(tf.gather_nd(y_neg_loss1,y_neg_indx))
    
    if log:
        print(f'pos_loss: {y_pos_loss.numpy()}')
        print(f'neg_loss: {y_neg_loss.numpy()}')
    return y_pos_loss +  y_neg_loss

def my_loss1(y_true, y_pred):
    y_true = tf.where(y_true<0.8,0.,1.)
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true,logits=y_pred)
    return tf.reduce_mean(cross_ent)