import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from capria_angio_to import capria_angio
from torch.utils.tensorboard import SummaryWriter


# Define a MLP
class MLP(nn.Module):
    def __init__(self, output_size, input_size=3, hidden_size=100):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.act1 = nn.Tanh()
    def forward(self, x):
        out = self.fc1(x)
        out = self.act1(out)
        out = self.fc2(out)

        return out

# define a capria_angio_dataset dataset class
class capria_angio_dataset(torch.utils.data.Dataset):
    def __init__(self, TR = 14.7e-3, tau = 1.8, T1b = 1.65, N = 144, FAMode = 'Quadratic', FAParams = [3,12], nsamples=20) -> None:
        # Set sequence parameters
        self.angio_model = capria_angio(TR, tau, T1b, N, FAMode, FAParams)
        self.delta_ts_range = [0* 1e3, 2.5*1e3]
        self.ss_range = [1, 100]
        self.ps_range = [1e-3* 1e3, 500e-3* 1e3]
        self.nsamples = nsamples

    def __len__(self):
        return 1000

    def __getitem__(self, idx):
        # Generate random parameters
        delta_ts, ss, ps = self.gen_params()
        return (np.concatenate((delta_ts, ss, ps), axis=1), self.angio_model.CAPRIAAngioSigAllRFAnalytic(delta_ts,ss,ps))
    
    def gen_params(self):
        delta_ts = np.random.uniform(self.delta_ts_range[0], self.delta_ts_range[1], (self.nsamples,))
        ss = np.random.uniform(self.ss_range[0], self.ss_range[1], (self.nsamples,))
        ps = np.random.uniform(self.ps_range[0], self.ps_range[1], (self.nsamples,))
        [delta_ts, ss, ps] = np.meshgrid(delta_ts,ss,ps, indexing='ij')
        delta_ts = delta_ts.reshape(-1,1)
        ss = ss.reshape(-1,1)
        ps = ps.reshape(-1,1)
        return delta_ts, ss, ps
    
# train the model
class trainer:
    def __init__(self, lr=1e-3, batch_size=1, logpath='.logs') -> None:
        self.model = MLP(output_size=144)
        self.dataset = capria_angio_dataset()
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optim, milestones=[350, 600, 800], gamma=0.1)
        self.writer = SummaryWriter(logpath)
        self.loss_fn = nn.MSELoss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.logpath = logpath
    def train(self, epochs):
        for epoch in tqdm(range(epochs)):
            x, y = next(iter(self.dataloader))
            self.optim.zero_grad()
            out = self.model(x.to(self.device).to(torch.float32))
            loss = self.loss_fn(out, y.to(self.device).to(torch.float32))
            loss.backward()
            self.optim.step()
            self.lr_scheduler.step()
            with torch.no_grad():
                self.writer.add_scalar('Loss/train', loss.item(), epoch)
        self.writer.close()
        torch.save(self.model, self.logpath + '/model.pt')

def collate_fn(data):
    x, y = zip(*data)
    x = torch.from_numpy(np.concatenate(x, axis=0))
    y = torch.from_numpy(np.concatenate(y, axis=0))
    return x, y


if __name__ == '__main__':
    trainer = trainer(lr=1e-2, batch_size=16, logpath='/well/okell/users/dcs094/data/dynamic_recon/train_mlp_logs/train1')
    trainer.train(1000)