import numpy as np
import pandas as pd
import torch as t
import matplotlib.pyplot as plt

if t.cuda.is_available():
    device = t.device('cuda')
else:
    device = t.device('cpu')
print('Device: ' + str(device))


class SGDataset(t.utils.data.Dataset):
    def __init__(self, src_df):
        xdf = src_df.drop(columns='run time').values
        ydf = src_df['run time'].values.reshape(-1, 1)  # 2-D required
        self.x_data = t.tensor(xdf, dtype=t.float32).to(device)
        self.y_data = t.tensor(ydf, dtype=t.float32).to(device)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        preds = self.x_data[idx]
        runtime = self.y_data[idx]
        return preds, runtime  # tuple of two matrices


class Net(t.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hid1 = t.nn.Linear(77, 100)  # 77-(100-100-100)-1
        self.hid2 = t.nn.Linear(100, 100)
        self.hid3 = t.nn.Linear(100, 100)
        self.oupt = t.nn.Linear(100, 1)

    def forward(self, x):
        z = t.tanh(self.hid1(x))
        z = t.tanh(self.hid2(z))
        z = t.tanh(self.hid3(z))
        z = self.oupt(z)  # no activation
        return z


def main():
    bat_size = 20
    num_epochs = 50
    num_reps = 1
    ep_log_interval = 25
    lrn_rate = 0.001
    reg_lambda = 0
    k = 3  # top-k accuracy
    n = 10  # no. of SKU data required for new workload

    df = pd.read_csv('sgim.csv')
    wkldname = ['AES', 'BZip2 Compress', 'BZip2 Decompress', 'BlackScholes', 'Blur Filter', 'DFFT', 'DGEMM', 'Dijkstra', 'JPEG Compress', 'JPEG Decompress', 'Lua', 'Mandelbrot', 'N-Body', 'PNG Compress', 'PNG Decompress', 'Ray Trace', 'SFFT', 'SGEMM', 'SHA1', 'SHA2', 'Sharpen Filter', 'Sobel', 'Stream Add', 'Stream Copy', 'Stream Scale', 'Stream Triad', 'Twofish']
    wkldmae = [0] * 27  # for storing average DNN model MAE
    wkldsd = [0] * 27  # for storing average DNN model SD
    wkldacc = [0] * 27  # for storing DNN model accuracy
    spec_df = df[df['sg'] == 0]

    # loop over 27 workloads
    for wkld in range(27):
        print("Workload : %d" % wkld)
        wkld_df = df[df.iloc[:, wkld + 51] == 1]  # the first benchmark column no. is 51
        n_correct = 0
        n_wrong = 0
        mae = np.zeros(num_reps)  # array of mae of each repetition DNN
        # 20 repetitions for each workload
        for rep in range(1, num_reps + 1):
            # gather randomly 10% data of workload for test_ds
            print("\tRepetition : %2d" % rep)
            test_df = wkld_df.sample(frac=0.1, random_state=rep)
            # sample 10 SKUs from remaining wkld_df rows and combine with rest of workloads' dataset to form train_ds
            traindf = wkld_df.drop(test_df.index).sample(n, random_state=rep)
            train_df = pd.concat([spec_df, traindf])
            test_df.reset_index(drop=True, inplace=True)
            min_rt_idx = test_df['run time'].idxmin()  # minimum runtime index in test dataset

            train_ds = SGDataset(train_df)
            test_ds = SGDataset(test_df)
            train_ldr = t.utils.data.DataLoader(train_ds, batch_size=bat_size, shuffle=True)
            test_ldr = t.utils.data.DataLoader(test_ds, batch_size=bat_size)

            net = Net().to(device)
            loss_func = t.nn.L1Loss()
            optimizer = t.optim.RMSprop(net.parameters(), lr=lrn_rate)

            net.train()  # set mode
            for epoch in range(1, num_epochs + 1):
                t.manual_seed(epoch)  # recovery reproducibility
                epoch_loss = 0  # for one full epoch
                for X, Y in train_ldr:
                    oupt = net(X)  # predicted prices
                    loss_val = loss_func(oupt, Y)  # avg per item in batch
                    epoch_loss += loss_val.item()  # accumulate avgs
                    loss_val += reg_lambda * sum(p.abs().sum() for p in net.parameters())  # L1 regularization
                    optimizer.zero_grad()  # prepare gradients
                    loss_val.backward()  # compute gradients
                    optimizer.step()  # update wts

                epoch_loss /= len(train_ldr)
                if epoch % ep_log_interval == 0 or epoch == 1:
                    print("\t\tEpoch = %4d   loss = %0.4f" % (epoch, epoch_loss))

            # path = f"./Case3/Spec-GeekLog/Wkld{wkld}_Rep{rep}.pth"
            # info_dict = {
            #    'repetition': rep,
            #    'net_state': net.state_dict(),
            #    'optimizer_state': optimizer.state_dict()
            # }
            # t.save(info_dict, path)

            net.eval()
            preds = []
            for X, Y in test_ldr:
                with t.no_grad():
                    oupt = net(X)
                    preds += oupt.tolist()  # computed runtime
                    mae[rep - 1] += loss_func(oupt, Y).item()

            mae[rep - 1] /= len(test_ldr)
            print("\tMAE of repetition %2d predictions = %0.4f" % (rep, mae[rep - 1]))
            min_rt_pred = preds[min_rt_idx]  # predicted runtime value for SKU which has minimum runtime
            preds.sort()
            print("\tRank of best SKU in test SKU predictions = %4d in %4d" % (preds.index(min_rt_pred), len(preds)))
            if preds.index(min_rt_pred) < k:  # min runtime is among top k predicted min runtimes
                n_correct += 1
            else:
                n_wrong += 1

        wkldmae[wkld] = mae.mean()
        wkldsd[wkld] = mae.std()
        wkldacc[wkld] = (n_correct * 1.0) / (n_correct + n_wrong)  # accuracy of this uarch
        print("Average MAE, SD and accuracy of %s DNN predictions = %0.4f %0.4f %0.4f" % (
            wkldname[wkld], wkldmae[wkld], wkldsd[wkld], wkldacc[wkld]))

    wkldmae, wkldsd, wkldacc, wkldname = (list(a) for a in zip(*sorted(zip(wkldmae, wkldsd, wkldacc, wkldname))))
    wkldname.append('average')
    wkldmae.append(sum(wkldmae[:27]) / 27)
    wkldsd.append(sum(wkldsd[:27]) / 27)
    wkldacc.append(sum(wkldacc[:27]) / 27)

    plt.plot(wkldname, wkldmae)
    plt.xticks(rotation=90)
    plt.ylabel('Mean Absolute Error', fontweight='bold', fontsize=10)
    plt.grid(ls=(0, (1, 4)))
    plt.tight_layout()
    plt.savefig("Geekbench wkld MAE.png")
    plt.close()

    plt.bar(wkldname, wkldacc, color='g')
    plt.xticks(rotation=90)
    plt.ylabel('Top-3 Accuracy', fontweight='bold', fontsize=10)
    plt.grid(ls=(0, (1, 4)))
    plt.tight_layout()
    plt.savefig("Geekbench wkld accuracy.png")
    plt.close()


if __name__ == "__main__":
    main()
