import pytest
import torch
# from zmq import device
import bootcamp
import bootcamp.model
import bootcamp.dataset
import torch.cpu, torch.cuda

device = torch.device('cuda:0')

@pytest.mark.parametrize('device', 
['cpu', pytest.param('cuda', 
marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='Cuda required' ))])

def test_pytorch_sum_device(device):
    x = torch.ones(10, device=device, dtype=torch.int32)
    assert x.sum() == 10

config = bootcamp.model.PlacesTrainingConfig()
config.model.width_multiplier = 0.25

model = bootcamp.model.PlacesModel(config)
model = model.to(device)

dm = bootcamp.dataset.PlacesDataModule(16, '/places365', 1)
dm.setup()

batch = next( iter(dm.train_dataloader()) )
torch.save(batch, 'testdata/batch.pt')

batch = torch.load('testdata/batch.pt')
batch = [ v.to(device)  for v in batch ]



