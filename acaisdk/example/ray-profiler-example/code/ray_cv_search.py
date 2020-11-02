import time
import joblib
import sys
import pandas as pd
from dask.distributed import Client
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
from ray.util.joblib import register_ray
import ray

# parser = argparse.ArgumentParser(description='Process some integers.')
# parser.add_argument("-c", "--cpu", type=int, help="number of vCPUs to use")
# parser.add_argument("-m", "--memory", type=int, help="amount of memory in MBs to use")
# parser.add_argument("-e", "--epoch", type=int, help="number of epochs")
# args = parser.parse_args()
#
# vcpus = args.cpu
# memory_limit = args.memory * 1e6   # set memory
# epochs = args.epoch



# async def f():
#     async with Client(asynchronous=True, processes=True) as client:
#         data = (X_train, y_train)
#         # scatter data
#         data_scatter = time.time()
#         scatterd = await client.scatter(data)
#         print("data scattered")
#         data_scatter_time = time.time() - data_scatter
#         # submit training job
#         submit_task = time.time()
#         future = client.submit(clf.fit, scatterd[0], scatterd[1])
#         result = await future
#         print("model trained")
#         train_time = time.time() - submit_task
#         # inference
#         inference = time.time()
#         score = result.score(X_test, y_test)
#         inf_time = time.time() - inference
#         print("CPU: %d Memory: %d Epochs: %d Scatter time: %.3f Training time: %.3f "
#               "Inference time: %.3f Score %.4f"
#               % (vcpus, memory_limit / 1e6, epochs, data_scatter_time, train_time, inf_time, score))
#
# # client = Client(processes=False)
if __name__ == '__main__':
    start_time = time.time()
    # W = int(sys.argv[1])
    # vCPUs = float(sys.argv[2])
    # Memory = int(sys.argv[3])
    # epochs = int(sys.argv[4])
    epochs = 50
    # output_file = open("output_seq.txt", "a+")
    data_path = "mnist_784_csv.csv"
    # output_file.write("Workers\tvCPUs\tMemory\tEpochs\ttrain_time")

    # Load data from https://www.openml.org/d/554
    # 784 features, 70000 instances, 10 classes
    # X_train, X_test = np.load('X_train.npy', allow_pickle=True), np.load('X_test.npy', allow_pickle=True)
    # y_train, y_test = np.load('y_train.npy', allow_pickle=True), np.load('y_test.npy', allow_pickle=True)
    # X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

    mnist = pd.read_csv(data_path)
    X, y = mnist.values[:, :-1], mnist.values[:, -1]
    random_state = check_random_state(0)
    permutation = random_state.permutation(X.shape[0])
    X = X[permutation]
    y = y[permutation]
    X = X.reshape((X.shape[0], -1))

    train_samples = 5000
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_samples, test_size=10000)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    submit_task = time.time()
    # Turn up tolerance for faster convergence
    
    clf = LogisticRegressionCV(cv=10, penalty='l1', solver='saga', tol=0.1,
                               max_iter=epochs, multi_class='auto')
    ray.init()
    register_ray()
    with joblib.parallel_backend('ray'):
        clf.fit(X_train, y_train)
    train_time = time.time() - submit_task
    # inference
    inference = time.time()
    score = clf.score(X_test, y_test)
    inf_time = time.time() - inference

    print(epochs, train_time, inf_time, score)
    # workers = c.scheduler_info()['workers']
    # # print(workers)
    # c.retire_workers(workers=[k for k, v in workers.items()])
    # print("killed all workers")
    total_time = time.time() - start_time
    # output_file.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(W, vCPUs, Memory, epochs, train_time, total_time))
    # output_file.close()
