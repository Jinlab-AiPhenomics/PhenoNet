import torch
from tqdm import tqdm
from torchvision import transforms
from network.lstm import lstm_network
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from network.phenovit import phenovit_network

class PhenoNet():
    def __init__(self, device, test_path, phenovit_weight_path, lstm_weight_path):
        super(PhenoNet, self).__init__()
        self.phenovit = phenovit_network().to(device)
        check_point = torch.load(phenovit_weight_path)
        self.phenovit.load_state_dict(check_point, strict=False)
        self.phenovit.eval()
        self.lstm_network = lstm_network().to(device)
        check_point = torch.load(lstm_weight_path)
        self.lstm_network.load_state_dict(check_point)
        self.lstm_network.eval()  
        self.device = device
        self.test_path = test_path

    def print_architecture(self):
        print(self.phenovit)
        print(self.lstm_network)
    
    def predict(self):
        img_tensors = self.__read_data()
        phenovit_features = []
        with torch.no_grad():
            for img in img_tensors:
                output = torch.squeeze(self.phenovit(img.to(self.device))).cpu()
                phenovit_features.append(output)
        phenovit_features = torch.stack(phenovit_features)
        phenovit_features = torch.unsqueeze(phenovit_features, dim=0)
        with torch.no_grad():
            scores = self.lstm_network(torch.as_tensor(phenovit_features, dtype=torch.float32))
            preds = torch.argmax(scores, dim=1)
        return int(preds.item())

    def __read_data(self):
        data_transform = transforms.Compose([transforms.Resize((224, 224)),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                            ])
        dataset = ImageFolder(self.test_path, transform=data_transform)
        dataloader = DataLoader(dataset, batch_size=1, num_workers=4)
        img_tensors = []
        for img, _ in tqdm(dataloader):
            img_tensors.append(img)
        return img_tensors
