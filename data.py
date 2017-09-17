import numpy as np

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict

class OmniglotNShotDataset():
    def __init__(self, batch_size, classes_per_set=10, samples_per_class=1, seed=2591, queries_per_class=1):

        """
        Constructs an N-Shot omniglot Dataset
        :param batch_size: Experiment batch_size
        :param classes_per_set: Integer indicating the number of classes per set
        :param samples_per_class: Integer indicating samples per class
        e.g. For a 20-way, 1-shot learning task, use classes_per_set=20 and samples_per_class=1
             For a 5-way, 10-shot learning task, use classes_per_set=5 and samples_per_class=10
        """
        np.random.seed(seed)
        self.x = np.load("omniglot.npy")
        self.x = np.reshape(self.x, newshape=(1622, 20, 28, 28, 1))
        self.x_train, self.x_test, self.x_val = self.x[:1200], self.x[1200:1411], self.x[1411:]
        self.normalization()
        self.batch_size = batch_size
        self.n_classes = self.x.shape[0]
        self.classes_per_set = classes_per_set
        self.samples_per_class = samples_per_class
        self.queries_per_class = queries_per_class

        print("train_shape", self.x_train.shape, "test_shape", self.x_test.shape, "val_shape", self.x_val.shape)
        self.indexes = {"train": 0, "val": 0, "test": 0}
        self.datasets = {"train": self.x_train, "val": self.x_val, "test": self.x_test} #original data cached


    def normalization(self):
        """
        Normalizes our data, to have a mean of 0 and sd of 1
        """
        self.mean = np.mean(list(self.x_train)+list(self.x_val))
        self.std = np.std(list(self.x_train)+list(self.x_val))

        self.max = np.max(self.x_train)
        self.min = np.min(self.x_train)
        self.x_train = 2.0 * self.x_train - 1.0
        self.x_val = 2.0 * self.x_val - 1.0
        self.x_test = 2.0 * self.x_test - 1.0

        print("after_normalization", "mean", np.mean(self.x_train), "max", np.max(self.x_train), "min", np.min(self.x_train), "std", np.std(self.x_train))


    def get_new_batch(self, data_pack):
        """
        Collects 1000 batches data for N-shot learning
        :param data_pack: Data pack to use (any one of train, val, test)
        :return: A list with [support_set_x, support_set_y, target_x, target_y] ready to be fed to our networks
        """
        support_set_x = np.zeros((self.batch_size, self.classes_per_set, self.samples_per_class, data_pack.shape[2],
                                  data_pack.shape[3], data_pack.shape[4]), dtype=np.float32)
        support_set_y = np.zeros((self.batch_size, self.classes_per_set, self.samples_per_class), dtype=np.float32)
        target_x = np.zeros((self.batch_size, self.classes_per_set, self.queries_per_class, data_pack.shape[2], data_pack.shape[3], data_pack.shape[4]),
                            dtype=np.float32)
        target_y = np.zeros((self.batch_size, self.classes_per_set, self.queries_per_class), dtype=np.float32)

        for i in range(self.batch_size):
            classes_idx = np.arange(data_pack.shape[0])
            samples_idx = np.arange(data_pack.shape[1])
            choose_classes = np.random.choice(classes_idx, size=self.classes_per_set, replace=False)
            choose_label = np.random.choice(self.classes_per_set, size=1)

            choose_samples = np.random.choice(samples_idx, size=self.samples_per_class+self.queries_per_class, replace=False)

            x_chosen = data_pack[choose_classes]
            x_support = x_chosen[:, choose_samples[:self.samples_per_class]]
            x_target = x_chosen[:, choose_samples[self.samples_per_class:]]

            y_temp = np.arange(self.classes_per_set).reshape(self.classes_per_set, 1)

            support_set_x[i] = x_support
            support_set_y[i] = np.dot(y_temp, np.ones((1,self.samples_per_class)))
            target_x[i] = x_target
            target_y[i] = np.dot(y_temp, np.ones((1,self.queries_per_class)))

        return support_set_x, support_set_y, target_x, target_y

    def get_batch(self, dataset_name, augment=False):
        """
        Gets next batch from the dataset with name.
        :param dataset_name: The name of the dataset (one of "train", "val", "test")
        :return:
        """
        x_support_set, y_support_set, x_target, y_target = self.get_new_batch(self.datasets[dataset_name])
        if augment:
            k = np.random.randint(0, 4, size=(self.batch_size, self.classes_per_set))
            x_augmented_support_set = []
            # x_augmented_target_set = []
            for b in range(self.batch_size):
                temp_class_support = []

                for c in range(self.classes_per_set):
                    x_temp_support_set = self.rotate_batch(x_support_set[b, c], axis=(1, 2), k=k[b, c])
                    # if y_target[b] == y_support_set[b, c, 0]:
                        # x_temp_target = self.rotate_batch(x_target[b], axis=(0, 1), k=k[b, c])

                    temp_class_support.append(x_temp_support_set)

                x_augmented_support_set.append(temp_class_support)
                # x_augmented_target_set.append(x_temp_target)
            x_support_set = np.array(x_augmented_support_set)
            # x_target = np.array(x_augmented_target_set)

        "reshape and shuffle"
        n_samples = self.samples_per_class*self.classes_per_set
        n_queries = self.queries_per_class*self.classes_per_set

        x_shape = x_support_set.shape[-3:]
        x_support_set = np.reshape(x_support_set, (self.batch_size, n_samples, x_shape[0], x_shape[1], x_shape[2]))
        y_support_set = np.reshape(y_support_set, (self.batch_size, n_samples))
        shuffle_support = np.random.permutation(np.arange(n_samples))
        support_set_x = x_support_set[:, shuffle_support, :, :, :]
        support_set_y = y_support_set[:, shuffle_support]

        x_target = np.reshape(x_target, (self.batch_size, n_queries, x_shape[0], x_shape[1], x_shape[2]))
        y_target = np.reshape(y_target, (self.batch_size, n_queries))
        shuffle_target = np.random.permutation(np.arange(n_queries))

        x_target = x_target[:, shuffle_target]
        y_target = y_target[:, shuffle_target]

        return x_support_set, y_support_set, x_target, y_target

    def rotate_batch(self, x_batch, axis, k):
        # x_batch = rotate(x_batch, k*90, reshape=False, axes=axis, mode="nearest")
        return x_batch

    def get_train_batch(self, augment=False):

        """
        Get next training batch
        :return: Next training batch
        """
        return self.get_batch("train", augment)

    def get_test_batch(self, augment=False):

        """
        Get next test batch
        :return: Next test_batch
        """
        return self.get_batch("test", augment)

    def get_val_batch(self, augment=False):

        """
        Get next val batch
        :return: Next val batch
        """
        return self.get_batch("val", augment)


class CIFAR_100():
    def __init__(self, batch_size, samples_per_class=1, seed=2591, queries_per_class=1):
        """
        classes_per_set unused.
        """
        """
        Constructs an N-Shot omniglot Dataset
        :param batch_size: Experiment batch_size
        :param classes_per_set: Integer indicating the number of classes per set
        :param samples_per_class: Integer indicating samples per class
        e.g. For a 20-way, 1-shot learning task, use classes_per_set=20 and samples_per_class=1
             For a 5-way, 10-shot learning task, use classes_per_set=5 and samples_per_class=10
        """
        np.random.seed(seed)
        train_dict = unpickle('/home/weilin/OneShot/CIFAR_100/train')
        test_dict = unpickle('/home/weilin/OneShot/CIFAR_100/test')
        X_train = train_dict[b'data']
        X_test = test_dict[b'data']
        y_train = train_dict[b'fine_labels']
        y_test = test_dict[b'fine_labels']
        X = np.concatenate((X_train, X_test), axis = 0)
        y = np.concatenate((y_train, y_test), axis = 0)
        self.x, y_new = self.get_indexed_data(X, y, 100)
        self.x = np.reshape(self.x, [-1, 600, 32, 32, 3])
        shuffle_classes = np.arange(self.x.shape[0])
        np.random.shuffle(shuffle_classes)
        self.x = self.x[shuffle_classes]
        self.x_train, self.x_val, self.x_test  = self.x[:64], self.x[64:80], self.x[80:]

        # self.normalization()
        self.batch_size = batch_size
        self.n_classes = self.x.shape[0]
        self.samples_per_class = samples_per_class
        self.queries_per_class = queries_per_class

        print("train_shape", self.x_train.shape, "test_shape", self.x_test.shape, "val_shape", self.x_val.shape)
        self.indexes = {"train": 0, "val": 0, "test": 0}
        self.datasets = {"train": self.x_train, "val": self.x_val, "test": self.x_test} #original data cached


    def normalization(self):
        """
        Normalizes our data, to have a mean of 0 and sd of 1
        """
        self.mean = np.mean(list(self.x_train)+list(self.x_val))
        self.std = np.std(list(self.x_train)+list(self.x_val))

        self.max = np.max(self.x_train)
        self.min = np.min(self.x_train)
        self.x_train = 2.0 * self.x_train - 1.0
        self.x_val = 2.0 * self.x_val - 1.0
        self.x_test = 2.0 * self.x_test - 1.0

        print("after_normalization", "mean", np.mean(self.x_train), "max", np.max(self.x_train), "min", np.min(self.x_train), "std", np.std(self.x_train))

    def get_indexed_data(self, X, y, n):
        X_new = []
        y_new = []
        for i in range(n):
            idx_mask = [idx for idx, x in enumerate(y) if x==i]
            X_idx = X[idx_mask]
            y_idx = y[idx_mask]
            X_new.append(X_idx)
            y_new.append(y_idx)
        X_new = np.asarray(X_new)
        y_new = np.asarray(y_new)
        return X_new, y_new


    def get_new_batch(self, data_pack, n_classes):
        """
        Collects 1000 batches data for N-shot learning
        :param data_pack: Data pack to use (any one of train, val, test)
        :return: A list with [support_set_x, support_set_y, target_x, target_y] ready to be fed to our networks
        """
        support_set_x = np.zeros((self.batch_size, n_classes, self.samples_per_class, data_pack.shape[2],
                                  data_pack.shape[3], data_pack.shape[4]), dtype=np.float32)
        support_set_y = np.zeros((self.batch_size, n_classes, self.samples_per_class), dtype=np.float32)
        target_x = np.zeros((self.batch_size, n_classes, self.queries_per_class, data_pack.shape[2], data_pack.shape[3], data_pack.shape[4]),
                            dtype=np.float32)
        target_y = np.zeros((self.batch_size, n_classes, self.queries_per_class), dtype=np.float32)

        for i in range(self.batch_size):
            classes_idx = np.arange(data_pack.shape[0])
            samples_idx = np.arange(data_pack.shape[1])
            choose_classes = np.random.choice(classes_idx, size=n_classes, replace=False)
            choose_label = np.random.choice(n_classes, size=1)

            choose_samples = np.random.choice(samples_idx, size=self.samples_per_class+self.queries_per_class, replace=False)

            x_chosen = data_pack[choose_classes]
            x_support = x_chosen[:, choose_samples[:self.samples_per_class]]
            x_target = x_chosen[:, choose_samples[self.samples_per_class:]]

            y_temp = np.arange(n_classes).reshape(n_classes, 1)

            support_set_x[i] = x_support
            support_set_y[i] = np.dot(y_temp, np.ones((1,self.samples_per_class)))
            target_x[i] = x_target
            target_y[i] = np.dot(y_temp, np.ones((1,self.queries_per_class)))

        return support_set_x, support_set_y, target_x, target_y

    def get_batch(self, dataset_name, n_classes,  augment=False):
        """
        Gets next batch from the dataset with name.
        :param dataset_name: The name of the dataset (one of "train", "val", "test")
        :return:
        """
        x_support_set, y_support_set, x_target, y_target = self.get_new_batch(self.datasets[dataset_name], n_classes)
        if augment:
            "k = np.random.randint(0, 4, size=(self.batch_size, n_classes))"
            k = np.random.randint(0, 4, size=(self.batch_size, n_classes))
            x_augmented_support_set = []
            x_augmented_support_set = []
            # x_augmented_target_set = []
            for b in range(self.batch_size):
                temp_class_support = []

                for c in range(n_classes):
                    x_temp_support_set = self.rotate_batch(x_support_set[b, c], axis=(1, 2), k=k[b, c])
                    # if y_target[b] == y_support_set[b, c, 0]:
                        # x_temp_target = self.rotate_batch(x_target[b], axis=(0, 1), k=k[b, c])

                    temp_class_support.append(x_temp_support_set)

                x_augmented_support_set.append(temp_class_support)
                # x_augmented_target_set.append(x_temp_target)
            x_support_set = np.array(x_augmented_support_set)
            # x_target = np.array(x_augmented_target_set)

        # reshape and shuffle
        n_samples = self.samples_per_class*n_classes
        n_queries = self.queries_per_class*n_classes

        x_shape = x_support_set.shape[-3:]
        x_support_set = np.reshape(x_support_set, (self.batch_size, n_samples, x_shape[0], x_shape[1], x_shape[2]))
        y_support_set = np.reshape(y_support_set, (self.batch_size, n_samples))
        shuffle_support = np.random.permutation(np.arange(n_samples))
        support_set_x = x_support_set[:, shuffle_support, :, :, :]
        support_set_y = y_support_set[:, shuffle_support]

        x_target = np.reshape(x_target, (self.batch_size, n_queries, x_shape[0], x_shape[1], x_shape[2]))
        y_target = np.reshape(y_target, (self.batch_size, n_queries))
        shuffle_target = np.random.permutation(np.arange(n_queries))

        x_target = x_target[:, shuffle_target]
        y_target = y_target[:, shuffle_target]

        return x_support_set, y_support_set, x_target, y_target

    def rotate_batch(self, x_batch, axis, k):
        # x_batch = rotate(x_batch, k*90, reshape=False, axes=axis, mode="nearest")
        return x_batch

    def get_train_batch(self, n_classes, augment=False):

        """
        Get next training batch
        :return: Next training batch
        """
        return self.get_batch("train", n_classes, augment)

    def get_test_batch(self, n_classes, augment=False):

        """
        Get next test batch
        :return: Next test_batch
        """
        return self.get_batch("test", n_classes, augment)


    def get_val_batch(self, n_classes, augment=False):

        """
        Get next val batch
        :return: Next val batch
        """
        return self.get_batch("val", n_classes, augment)



