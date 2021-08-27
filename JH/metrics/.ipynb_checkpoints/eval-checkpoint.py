import os
import pandas as pd
import cv2

class TestDataset(Dataset):
    def __init__(self, path, transform):
        img_list = []
        for p in tqdm(path):
            img = cv2.imread(p)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_list.append(img)
        
        self.X = img_list
        self.transform = transform

    def __len__(self):
        len_dataset = len(self.X)
        return len_dataset

    def __getitem__(self, idx):
        X = self.X[idx]
        X = self.transform(image=X)['image']  # transforms를 사용하시는 분은 X = self.transform(X)
        return X
    
class TTA:
    def __init__(self):
        self.test_dir = '/opt/ml/input/data/eval'
        self.submission = pd.read_csv(os.path.join(test_dir, 'info.csv'))
        self.image_dir = os.path.join(self.test_dir, 'images')
        self.image_paths = [os.path.join(self.image_dir, img_id) for img_id in self.submission.ImageID]
        
        
def TTA_way3(model, TTA=3, batch_size=128, threshold = 0.1):
    
    model.eval()
    avg_predictions = {}
    ans_dict = {}
    models_num = len(os.listdir("../input/your_models_folder"))
    
    for time in range(TTA):
        
        test_transformed_dataset = iMetDataset(csv_file='sample_submission.csv', 
                                      label_file="labels.csv", 
                                      img_path="test/", 
                                      root_dir='../input/imet-2019-fgvc6/',
                                      transform=transforms.Compose([
                                          #
                                          # some data augumentations here
                                          #
                                          transforms.ToTensor(),
                                          transforms.Normalize(
                                              [0.485, 0.456, 0.406], 
                                              [0.229, 0.224, 0.225])
                                      ]))

        test_loader = DataLoader(
        test_transformed_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8)

        with torch.no_grad():
            
            for i in range(models_num):

                model.load_state_dict(torch.load("../input/your_models_folder/your_model" + str(i)+ ".pth"))
   
                for batch_idx, sample in enumerate(test_loader):
     
                    image = sample["image"].to(device, dtype=torch.float)
                    img_ids = sample["img_id"]
                    predictions = model(image).cpu().numpy()
                    
                    for row, img_id in enumerate(img_ids):
                        if time == 0 and i == 0:
                            avg_predictions[img_id] = predictions[row]/(TTA*models_num)
                        else:
                            avg_predictions[img_id] += predictions[row]/(TTA*models_num)

                        if time == TTA - 1 and i == models_num -1:
                            all_class = np.nonzero(avg_predictions[img_id] > threshold)[0].tolist()
                            all_class = [str(x) for x in all_class]
                            ans_dict[img_id] = " ".join(all_class)
    
    return ans_dict