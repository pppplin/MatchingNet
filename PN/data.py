import numpy as np
np.random.seed(2191)  # for reproducibility

class CIFAR_100():
    def __init__(self, batch_size, classes_per_set=10, samples_per_class=1, gray = False, num_query_points = 15):

        """
        Constructs an N-Shot CIFAR100 Dataset
        :param batch_size: Experiment batch_size
        :param classes_per_set: Integer indicating the number of classes per set
        :param samples_per_class: Integer indicating samples per class
        e.g. For a 20-way, 1-shot learning task, use classes_per_set=20 and samples_per_class=1
        """
        train_dict = self.unpickle('/home/weilin/OneShot/CIFAR_100/train')
        test_dict = self.unpickle('/home/weilin/OneShot/CIFAR_100/test')
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

        temp = np.transpose(self.x, (1, 0, 2, 3, 4))
        shuffle_examples = np.arange(temp.shape[0])
        np.random.shuffle(shuffle_examples)
        temp = temp[shuffle_examples]
        self.x_query = np.transpose(temp[:15], (1, 0, 2, 3, 4))
        self.x = np.transpose(temp[15:], (1, 0, 2, 3, 4))

        self.x_train, self.x_val, self.x_test  = self.x[:64], self.x[64:80], self.x[80:]
        self.x_train_query, self.x_val_query, self.x_test_query  = self.x_query[:64], self.x_query[64:80], self.x_query[80:]

        # self.normalization() #only batch norm

        self.batch_size = batch_size
        self.n_classes = self.x.shape[0]
        self.classes_per_set = classes_per_set
        self.samples_per_class = samples_per_class

        self.indexes = {"train": 0, "val": 0, "test": 0}
        # self.datasets = {"train": self.x_train, "val": self.x_val, "test": self.x_test} #original data cached


        self.datasets = {"train": self.x_train, "val": self.x_val, "test": self.x_test} #original data cached
        self.datasets_query = {"train":self.x_train_query, "val":self.x_val_query, "test":self.x_test_query}

        # self.datasets_cache = {"train": self.load_data_cache(self.datasets["train"]),  #current epoch data cached
        #                        "val": self.load_data_cache(self.datasets["val"]),
        #                        "test": self.load_data_cache(self.datasets["test"])}
        if self.samples_per_class==1:
            self.datasets_cache = {"train": self.load_data_cache(self.datasets["train"], self.datasets_query["train"], self.samples_per_class, 3),  #current epoch data cached
                       "val": self.load_data_cache(self.datasets["val"], self.datasets_query["val"], self.samples_per_class, self.classes_per_set),
                       "test": self.load_data_cache(self.datasets["test"], self.datasets_query["test"], self.samples_per_class, self.classes_per_set)}
        else:
            self.datasets_cache = {"train": self.load_data_cache(self.datasets["train"], self.datasets_query["train"], self.samples_per_class, 3),  #current epoch data cached
                       "val": self.load_data_cache(self.datasets["val"], self.datasets_query["val"], self.samples_per_class, self.classes_per_set),
                       "test": self.load_data_cache(self.datasets["test"], self.datasets_query["test"], self.samples_per_class, self.classes_per_set)}



    def normalization(self):
        """
        Normalizes our data, to have a mean of 0 and sd of 1
        """
        self.mean = np.mean(self.x_train)
        self.std = np.std(self.x_train)
        self.max = np.max(self.x_train)
        self.min = np.min(self.x_train)
        print("train_shape", self.x_train.shape, "test_shape", self.x_test.shape, "val_shape", self.x_val.shape)
        print("before_normalization", "mean", self.mean, "max", self.max, "min", self.min, "std", self.std)
        self.x_train = (self.x_train - self.mean) / self.std
        self.x_val = (self.x_val - self.mean) / self.std
        self.x_test = (self.x_test - self.mean) / self.std
        self.mean = np.mean(self.x_train)
        self.std = np.std(self.x_train)
        self.max = np.max(self.x_train)
        self.min = np.min(self.x_train)
        print("after_normalization", "mean", self.mean, "max", self.max, "min", self.min, "std", self.std)

    def unpickle(self, file):
        import pickle
        with open(file, 'rb') as fo:
            dict = pickle.load(fo)
        return dict

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

    def load_data_cache(self, data_pack, data_pack_query, samples_per_class, classes_per_set):
        # PROBLEM HERE!!
        """
        Collects 1000 batches data for N-shot learning
        :param data_pack: Data pack to use (any one of train, val, test)
        :return: A list with [support_set_x, support_set_y, target_x, target_y] ready to be fed to our networks
        """
        # n_samples = self.samples_per_class * self.classes_per_set
        n_samples = samples_per_class * classes_per_set

        data_cache = []
        for sample in range(1000):
            support_set_x = np.zeros((self.batch_size, n_samples, 32, 32, 3))
            support_set_y = np.zeros((self.batch_size, n_samples))
            target_x = np.zeros((self.batch_size, 32, 32, 3), dtype=np.int)
            target_y = np.zeros((self.batch_size,), dtype=np.int)
            for i in range(self.batch_size):
                ind = 0
                pinds = np.random.permutation(n_samples)
                # pinds = np.arange(n_samples) 
                classes = np.random.choice(data_pack.shape[0], self.classes_per_set, False)
                x_hat_class = np.random.randint(self.classes_per_set) 

                for j, cur_class in enumerate(classes):  # each class
                    example_inds = np.random.choice(data_pack.shape[1], self.samples_per_class, False)
                    for eind in example_inds:
                        support_set_x[i, pinds[ind], :, :, :] = data_pack[cur_class][eind]
                        support_set_y[i, pinds[ind]] = j
                        ind += 1
                    if j == x_hat_class:
                        target_x[i, :, :, :] = data_pack_query[cur_class][np.random.choice(data_pack_query.shape[1])]
                        target_y[i] = j

            data_cache.append([support_set_x, support_set_y, target_x, target_y])
        return data_cache

    def get_batch(self, dataset_name):
        # ???
        """
        Gets next batch from the dataset with name.
        :param dataset_name: The name of the dataset (one of "train", "val", "test")
        :return:
        """
        if self.indexes[dataset_name] >= len(self.datasets_cache[dataset_name]):
            self.indexes[dataset_name] = 0
            if dataset_name == "train":
                if self.samples_per_class == 1:
                    self.datasets_cache[dataset_name] = self.load_data_cache(self.datasets[dataset_name], \
                        self.datasets_query[dataset_name], self.samples_per_class, 3) #should be num of train classes
                else:
                    self.datasets_cache[dataset_name] = self.load_data_cache(self.datasets[dataset_name], \
                        self.datasets_query[dataset_name], self.samples_per_class, 3)
            else:
                self.datasets_cache[dataset_name] = self.load_data_cache(self.datasets[dataset_name], \
                    self.datasets_query[dataset_name], self.samples_per_class, self.classes_per_set)

        # ???
        next_batch = self.datasets_cache[dataset_name][self.indexes[dataset_name]]
        self.indexes[dataset_name] += 1
        x_support_set, y_support_set, x_target, y_target = next_batch
        return x_support_set, y_support_set, x_target, y_target

    def get_train_batch(self):	

        """
        Get next training batch
        :return: Next training batch
        """
        return self.get_batch("train")

    def get_test_batch(self):

        """
        Get next test batch
        :return: Next test_batch
        """
        return self.get_batch("test")

    def get_val_batch(self):

        """
        Get next val batch
        :return: Next val batch
        """
        return self.get_batch("val")
