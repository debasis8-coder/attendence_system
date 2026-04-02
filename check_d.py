from facenet_pytorch import InceptionResnetV1,MTCNN
from PIL import Image
import torch
from torchvision import transforms
import os
import time


class Emb_vec:
    def __init__(self):
        pass
    
    
    def check(self):
        # If required, create a face detection pipeline using MTCNN:
        mtcnn = MTCNN(image_size=160, margin=0).to('cpu')
        
        # Create an inception resnet (in eval mode):
        resnet = InceptionResnetV1(pretrained='vggface2').to('cpu').eval()

        transform1 = transforms.Compose([transforms.ToTensor()])

        emb=[]
        clas=[]
        data_known=sorted(os.listdir('face_dataset'))
        for names in data_known:
            clas.append(names.split('.')[0])
        for k in data_known:
        
            j1=k.split('.')[0] 
            known_img=Image.open(f"face_dataset/"+f"{j1}.jpg").convert('RGB')
            known_ten=transform1(known_img)
            known_emb=resnet(known_ten.unsqueeze(0).to('cpu'))
            emb.append(known_emb)
        
        #print(emb)

        image=Image.open('test_face/test.jpg')
        img_cropped = mtcnn(image)
        ################################################

        ################################################
        # Calculate embedding (unsqueeze to add batch dimension)
        if img_cropped is not None:
            img_embedding = resnet(img_cropped.unsqueeze(0).to('cpu'))
            prob=[]
            for l in emb:
                recg=torch.cosine_similarity(img_embedding,l,dim=1).item()
                
                prob.append(recg)
            print(prob)
            val=max(prob)
            if val>0.65:
                
                file=open('check_face.txt','w')
                file.write(f"NAME:{clas[prob.index(val)]}")
                print(clas[prob.index(val)],prob)
            
        else:
            file=open('check_face.txt','w')
            file.write('')

if __name__=='__main__':

    try:
        ob=Emb_vec()
        ob.check()
            
    except:
        pass
    
   
        

    
