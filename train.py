from one_shot_learning_network import *
from experiment_builder import ExperimentBuilder_omniglot, ExperimentBuilder_CIFAR
import tensorflow.contrib.slim as slim
import data as dataset
import tqdm
from storage import *
from gpu import define_gpu
define_gpu(1)   

tf.reset_default_graph()

train_on_img = True

classes_per_set = 5
samples_per_class = 5

batch_size = 4
 #32
fce = False

continue_from_epoch = -1 # use -1 to start from scratch
epochs = 1
# epochs = --0-200
logs_path = "MatchingNet_outputs/"

if train_on_img:
    channels = 3    
    experiment_name = logs_path + "cifar_one_shot_learning_embedding_{}_{}".format(samples_per_class, classes_per_set)
    data = dataset.CIFAR_100(batch_size=batch_size,
                                        classes_per_set=classes_per_set, samples_per_class=samples_per_class)
    experiment = ExperimentBuilder_CIFAR(data)
    one_shot_cifar, losses, c_error_opt_op, init = experiment.build_experiment(batch_size,\
        classes_per_set, samples_per_class, channels, fce)
else:
    channels = 1 
    experiment_name = logs_path + "one_shot_learning_embedding_{}_{}".format(samples_per_class, classes_per_set)
    data = dataset.OmniglotNShotDataset(batch_size=batch_size,
                                        classes_per_set=classes_per_set, samples_per_class=samples_per_class)
    experiment = ExperimentBuilder_omniglot(data)
    one_shot_omniglot, losses, c_error_opt_op, init = experiment.build_experiment(batch_size,\
        classes_per_set, samples_per_class, channels, fce)

           

total_epochs = 300
total_train_batches = 1000
total_val_batches = 100
total_test_batches = 250

save_statistics(experiment_name, ["epoch", "train_c_loss", "train_c_accuracy", "val_loss", "val_accuracy",
                                  "test_c_loss", "test_c_accuracy"])

# Experiment initialization and running
with tf.Session() as sess:
    sess.run(init)
    saver = tf.train.Saver()
    if continue_from_epoch != -1: #load checkpoint if needed
        checkpoint = "saved_models/{}_{}.ckpt".format(experiment_name, continue_from_epoch)
        variables_to_restore = []
        print("restoring variables...")
        for var in tf.get_collection(tf.GraphKeys.VARIABLES):
            print(var.name)
            variables_to_restore.append(var)
        # 
        tf.logging.info('Fine-tuning from %s' % checkpoint)

        fine_tune = slim.assign_from_checkpoint_fn(
            checkpoint,
            variables_to_restore,
            ignore_missing_vars=True)
        fine_tune(sess)

    best_val = 0.
    # with tqdm.tqdm(total=total_epochs) as pbar_e:
    #     for e in range(0, total_epochs):
    with tqdm.tqdm(total=epochs) as pbar_e:
        for e in range(0, epochs):
            total_c_loss, total_accuracy = experiment.run_training_epoch(total_train_batches=total_train_batches,
                                                                                sess=sess)
            print("Epoch {}: train_loss: {}, train_accuracy: {}".format(e, total_c_loss, total_accuracy))

            total_val_c_loss, total_val_accuracy = experiment.run_validation_epoch(
                                                                                total_val_batches=total_val_batches,
                                                                                sess=sess)
            print("Epoch {}: val_loss: {}, val_accuracy: {}".format(e, total_val_c_loss, total_val_accuracy))

            if total_val_accuracy >= best_val: #if new best val accuracy -> produce test statistics
                best_val = total_val_accuracy
                total_test_c_loss, total_test_accuracy = experiment.run_testing_epoch(
                                                                    total_test_batches=total_test_batches, sess=sess)
                print("Epoch {}: test_loss: {}, test_accuracy: {}".format(e, total_test_c_loss, total_test_accuracy))
            else:
                total_test_c_loss = -1
                total_test_accuracy = -1

            save_statistics(experiment_name,
                            [e, total_c_loss, total_accuracy, total_val_c_loss, total_val_accuracy, total_test_c_loss,
                             total_test_accuracy])

            save_path = saver.save(sess, "saved_models/{}_{}.ckpt".format(experiment_name, e))
            pbar_e.update(1)
