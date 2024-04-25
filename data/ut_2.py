from torch.utils import data
import torchvision.transforms as transforms
from torchvision.transforms.transforms import *
from torchvision.transforms.functional import *
import math
# import random
from PIL import Image
import numpy as np
import re
import os

# def count_frames_in_folder(folder_path):
#     files = os.listdir(folder_path)
#     return len(files)
# seq1 = count_frames_in_folder("/data-ext/UT_dataset/completed/images/seq1")
# seq2 = count_frames_in_folder("/data-ext/UT_dataset/completed/images/seq2")
# seq3 = count_frames_in_folder("/data-ext/UT_dataset/completed/images/seq3")
# seq4 = count_frames_in_folder("/data-ext/UT_dataset/completed/images/seq4")
# seq5 = count_frames_in_folder("/data-ext/UT_dataset/completed/images/seq5")
# seq6 = count_frames_in_folder("/data-ext/UT_dataset/completed/images/seq6")
# seq7 = count_frames_in_folder("/data-ext/UT_dataset/completed/images/seq7")
# seq8 = count_frames_in_folder("/data-ext/UT_dataset/completed/images/seq8")
# seq9 = count_frames_in_folder("/data-ext/UT_dataset/completed/images/seq9")
# seq10 = count_frames_in_folder("/data-ext/UT_dataset/completed/images/seq10")
# seq11 = count_frames_in_folder("/data-ext/UT_dataset/completed/images/seq11")
# seq12 = count_frames_in_folder("/data-ext/UT_dataset/completed/images/seq12")
# seq13 = count_frames_in_folder("/data-ext/UT_dataset/completed/images/seq13")
# seq14 = count_frames_in_folder("/data-ext/UT_dataset/completed/images/seq14")
# seq15 = count_frames_in_folder("/data-ext/UT_dataset/completed/images/seq15")
# seq16 = count_frames_in_folder("/data-ext/UT_dataset/completed/images/seq16")
# seq17 = count_frames_in_folder("/data-ext/UT_dataset/completed/images/seq17")
# seq18 = count_frames_in_folder("/data-ext/UT_dataset/completed/images/seq18")
# seq19 = count_frames_in_folder("/data-ext/UT_dataset/completed/images/seq19")
# seq20 = count_frames_in_folder("/data-ext/UT_dataset/completed/images/seq20")

action_dict={'handshake':0,'hug':1,'kick':2,'punch':3,'push':4,'others':5,
             'be kicked':5, 'be punched':5, 'be pushed':5} # all action classes

# num_frames_seqs={'seq1':seq1,'seq2':seq2,'seq3':seq3,'seq4':seq4,'seq5':seq5,'seq6':seq6,'seq7':seq7,'seq8':seq8,'seq9':seq9,
#              'seq10':seq10, 'seq11':seq11, 'seq12':seq12, 'seq13':seq13, 'seq14':seq14, 'seq15':seq15, 'seq16':seq16,
#               'seq17':seq17, 'seq18':seq18, 'seq19':seq19, 'seq20':seq20} # number of frames of each sequence

class UTDataset(data.Dataset): # create a dataset
    def __init__(self,bboxes,frames,labels_act, labels_inter, num_frames, image_size,feature_size,num_boxes=4,
                 data_augmentation=False):
        
        self.bboxes = bboxes
        self.frames = frames
        self.labels_act = labels_act
        self.labels_inter = labels_inter
        self.image_path = "/data-ext/UT_dataset/completed/UT_annotations_2/images/"
        self.image_size = image_size
        self.num_frames = num_frames
        self.feature_size = feature_size
        self.num_boxes = num_boxes
        self.data_augmentation = data_augmentation
        self.list_frames = []
        self.list_bboxes = []
        self.list_labels_inter = []
        self.list_labels_act = []

        for seq_name, frame_name in self.frames.items():
            # n_frames = len(frame_name)
            # if n_frames % self.num_frames != 0:
            #     for fid in frame_name:
            #         fid, _ = os.path.splitext(fid)
            #         frame = (seq_name, str(fid))
            #         self.list_frames.append(frame)
            #     for _ in range(self.num_frames - n_frames % self.num_frames):
            #         fid, _ = os.path.splitext(frame_name[-1])
            #         frame = (seq_name, str(fid))
            #         self.list_frames.append(frame)
            # else:
            for fid in frame_name:
                fid, _ = os.path.splitext(fid)
                frame = (seq_name, str(fid))
                self.list_frames.append(frame)

        
        for seq_name, frame_bboxes in self.bboxes.items():
            # n_frames = len(frame_bboxes)
            # if n_frames % self.num_frames != 0:
            #     for frame_id, person in frame_bboxes.items():
            #         list_people = []
            #         for pid, bbox in person.items():
            #             list_people.append((pid, bbox))
            #         self.list_bboxes.append((list_people))
            #     for _ in range(self.num_frames - n_frames % self.num_frames):
            #         frame_id = list(frame_bboxes.keys())[-1]
            #         person = frame_bboxes[frame_id]
            #         list_people = []
            #         for pid, bbox in person.items():
            #             list_people.append((pid, bbox))
            #         self.list_bboxes.append((list_people))
            # else:
            for frame_id, person in frame_bboxes.items():
                list_people = []
                for pid, bbox in person.items():
                    list_people.append((pid, bbox))
                self.list_bboxes.append((list_people))

                    

        for seq_name, frame_labels in self.labels_inter.items():
            # n_frames = len(frame_labels)
            # if n_frames % self.num_frames != 0:
            #     for frame_id, pairs in frame_labels.items():
            #         list_pairs = []
            #         for pair, label in pairs.items():
            #             list_pairs.append((pair, label))
            #         self.list_labels.append((list_pairs))
            #     for _ in range(self.num_frames - n_frames % self.num_frames):
            #         frame_id = list(frame_labels.keys())[-1]
            #         pairs = frame_labels[frame_id]
            #         list_pairs = []
            #         for pair, label in pairs.items():
            #             list_pairs.append((pair, label))
            #         self.list_labels.append((list_pairs))
            # else:
            for frame_id, pairs in frame_labels.items():
                list_pairs = []
                for pair, label in pairs.items():
                    list_pairs.append((pair, label))
                self.list_labels_inter.append((list_pairs))
        
        for seq_name, frame_labels in self.labels_act.items():
            for frame_id, person in frame_labels.items():
                list_person = []
                for pid, label in person.items():
                    list_person.append((pid, label))
                self.list_labels_act.append((list_person))


        # print(len(self.list_frames), len(self.list_bboxes), len(self.list_labels_inter), len(self.list_labels_act)) # 17745
        # exit()
        # count = 0
        # for i in self.list_frames:
        #     count += 1
        #     print(i)
        #     if count == 100:
        #         break
        # print(self.list_frames[0:3])
        # print(self.list_frames[3:6])
        # exit()
        
        
    def __len__(self):
        """
        Return the total number of samples (total number of frames of all videos in dataset)
        """
        count = int(len(self.list_frames)/self.num_frames)
        
        return count
    
    def __getitem__(self,index):
        """
        Generate one sample of the dataset
        """
        # index = int(index * self.num_frames)
        sample = self.load_samples_sequence(self.list_frames[index], self.list_bboxes[index], self.list_labels_inter[index], self.list_labels_act[index]) # Load every frame 

        # sample = self.load_samples_sequence(self.list_frames[int(index+math.ceil(self.num_frames/2))], self.list_bboxes[int(index+math.ceil(self.num_frames/2))], self.list_labels[int(index+math.ceil(self.num_frames/2))]) # Load every frame 
        # print(index, index+self.num_frames)
        # sample = self.load_samples_sequence(self.list_frames[index:index+self.num_frames], self.list_bboxes[index:index+self.num_frames], self.list_labels[index:index+self.num_frames]) # Load every frame 
        return sample

    def load_samples_sequence(self,select_frame, select_bboxes, select_labels_inter, select_labels_act):
        
        list_select_frame = []
        list_select_bboxes = []
        list_select_labels_inter = []
        list_select_labels_act = []
        list_select_frame.append(select_frame)
        list_select_bboxes.append(select_bboxes)
        list_select_labels_inter.append(select_labels_inter)
        list_select_labels_act.append(select_labels_act)
        select_frame = list_select_frame
        select_bboxes = list_select_bboxes
        select_labels_inter = list_select_labels_inter
        select_labels_act = list_select_labels_act

        # print(select_frame)    # ('seq1', '47') [('0', (0.878377, 0.516776, 0.16989, 0.547253)), ('1', (0.0292947, 0.643608, 0.047536, 0.449234))] [('0-1', 6)]
        # print("----------------")
        

        random_factor = np.random.randint(0, 2) # for augmentation
        OH, OW = self.feature_size
        images, bboxes, normalized_bboxes = [], [], []
        interactions = []
        actions = []
        bboxes_num = []

        for numf in range(len(select_frame)):
            seq_name, fid = select_frame[numf]
            label_set_inter = select_labels_inter[numf]
            label_set_act = select_labels_act[numf]
            num_people = len(select_bboxes[numf])
            img = Image.open(self.image_path + seq_name + '/' + fid + '.jpg')

            temp_boxes = []
            temp_normalized_boxes, temp_interactions, temp_actions = [], [], []
            list_id, list_pairs = [], []
            if self.data_augmentation == True:
                if random_factor == 0:# scaling + random color jittering
                    minx = 1
                    miny = 1
                    maxx = 0
                    maxy = 0
                    for pid in range(num_people):
                        _, box = select_bboxes[pid]
                        x_center, y_center, width, height = box # normalized bbox coordinates
                        x1 = x_center - width / 2
                        y1 = y_center - height / 2
                        x2 = x_center + width / 2
                        y2 = y_center + height / 2

                        minx = min(minx,x1)
                        miny = min(miny,y1)
                        maxx = max(maxx,x2)
                        maxy = max(maxy,y2)
                    maxx = 1 - maxx
                    maxy = 1 - maxy
                    lim = min(minx,miny,maxx,maxy) * 0.95
                    lower = max((1-lim),0.7)
                    upper = min((1+lim),1.3)
                    rnd = np.random.randint(0,100)*(upper-lower)*0.01+lower # get a random number in [lower, upper]
                    if rnd <= 1:# zoom in and crop
                        lim = 1 - rnd
                        i = lim*img.size[0] # random value to ensure not being cut
                        j = lim*img.size[1]
                        img = crop(img, j, i, img.size[1]-2*j, img.size[0]-2*i)
                        for box in self.anns[seq_name][fid]['bboxes']:
                            y1,x1,y2,x2 = box
                            y1 = (y1-0.5)*0.5/(0.5-lim)+0.5
                            x1 = (x1-0.5)*0.5/(0.5-lim)+0.5
                            y2 = (y2-0.5)*0.5/(0.5-lim)+0.5
                            x2 = (x2-0.5)*0.5/(0.5-lim)+0.5
                            w1,h1,w2,h2 = x1*OW, y1*OH, x2*OW, y2*OH  
                            temp_boxes.append((w1,h1,w2,h2))
                            temp_normalized_boxes.append((x1,y1,x2,y2))
                    else: # zoom out and interpolation
                        lim = rnd - 1
                        img = Pad((int((rnd-1)*img.size[0]),int((rnd-1)*img.size[1])), fill=0, padding_mode='reflect')(img)
                        for box in self.anns[seq_name][fid]['bboxes']:
                            y1,x1,y2,x2 = box
                            y1 = (y1-0.5)*0.5/(0.5+lim) + 0.5
                            x1 = (x1-0.5)*0.5/(0.5+lim) + 0.5
                            y2 = (y2-0.5)*0.5/(0.5+lim) + 0.5
                            x2 = (x2-0.5)*0.5/(0.5+lim) + 0.5
                            w1,h1,w2,h2 = x1*OW, y1*OH, x2*OW, y2*OH
                            temp_boxes.append((w1,h1,w2,h2))
                            temp_normalized_boxes.append((x1,y1,x2,y2))
                    img = ColorJitter(brightness = 0.4, contrast=0.25, saturation = 0.2)(img)

                elif random_factor == 1: # horizontal flipping + scaling + random color jittering
                    minx = 1
                    miny = 1
                    maxx = 0
                    maxy = 0
                    for box in self.anns[seq_name][fid]['bboxes']:
                        y1,x1,y2,x2 = box
                        minx = min(minx,x1)
                        miny = min(miny,y1)
                        maxx = max(maxx,x2)
                        maxy = max(maxy,y2)
                    maxx = 1-maxx
                    maxy = 1-maxy
                    lim = min(minx,miny,maxx,maxy)*0.95
                    lower = max((1-lim),0.7)
                    upper = min((1+lim),1.3)
                    rnd = np.random.randint(0,100)*(upper-lower)*0.01+lower
                    if rnd <= 1: # zoom in and crop
                        lim = 1 - rnd
                        i = lim*img.size[0]
                        j = lim*img.size[1]
                        img = crop(img, j, i, img.size[1]-2*j, img.size[0]-2*i)
                        for box in self.anns[seq_name][fid]['bboxes']:
                            y1,x1,y2,x2 = box
                            tmp1 = x1
                            tmp2 = x2
                            x1 = 1-tmp2
                            x2 = 1-tmp1
                            y1 = (y1-0.5)*0.5/(0.5-lim)+0.5
                            x1 = (x1-0.5)*0.5/(0.5-lim)+0.5
                            y2 = (y2-0.5)*0.5/(0.5-lim)+0.5
                            x2 = (x2-0.5)*0.5/(0.5-lim)+0.5
                            w1,h1,w2,h2 = x1*OW, y1*OH, x2*OW, y2*OH  
                            temp_boxes.append((w1,h1,w2,h2))
                            temp_normalized_boxes.append((x1,y1,x2,y2))
                    else: # zoom out and interpolation
                        lim=rnd-1
                        img = Pad((int((rnd-1)*img.size[0]),int((rnd-1)*img.size[1])), fill=0, padding_mode='reflect')(img)
                        for box in self.anns[seq_name][fid]['bboxes']:
                            y1,x1,y2,x2 = box
                            y1 = (y1-0.5)*0.5/(0.5+lim)+0.5
                            x1 = (x1-0.5)*0.5/(0.5+lim)+0.5
                            y2 = (y2-0.5)*0.5/(0.5+lim)+0.5
                            x2 = (x2-0.5)*0.5/(0.5+lim)+0.5
                            tmp1 = x1
                            tmp2 = x2
                            x1 = 1-tmp2
                            x2 = 1-tmp1
                            w1,h1,w2,h2 = x1*OW, y1*OH, x2*OW, y2*OH
                            temp_boxes.append((w1,h1,w2,h2))
                            temp_normalized_boxes.append((x1,y1,x2,y2))
                    img = ColorJitter(brightness=0.4,contrast=0.25, saturation=0.2)(img)
                    img = transforms.RandomHorizontalFlip(p=1)(img)
            else:
                for p in range(num_people):
                    pid, box = select_bboxes[numf][p]
                    x_center, y_center, width, height = box
                    x1 = x_center - width / 2
                    y1 = y_center - height / 2
                    x2 = x_center + width / 2
                    y2 = y_center + height / 2
                    # y1,x1,y2,x2=box # normalized bbox coordinates
                    w1,h1,w2,h2 = x1*OW, y1*OH, x2*OW, y2*OH    # convert to feature map coordinates [OH,OW]=][27,48] output size after passing through backbone
                    temp_boxes.append((w1,h1,w2,h2))
                    temp_normalized_boxes.append((x1,y1,x2,y2))
                    list_id.append(pid)

            img = transforms.functional.resize(img,self.image_size)
            img = np.array(img)

            # H,W,3 -> 3,H,W
            img = img.transpose(2,0,1)
            images.append(img) # original image and augmented images
            
            for i in list_id:
                for j in list_id:
                    if int(i)!=int(j):
                        pair = str(i) + '-' + str(j)
                        if pair not in list_pairs and str(j) + '-' + str(i) not in list_pairs:
                            list_pairs.append(pair)
            
            for p in list_pairs:
                for label in label_set_inter:
                    pl, l = label
                    if pl == p or pl == p[::-1]:
                        temp_interactions.append(int(l))
                        break
            
            bboxes_num.append(len(temp_boxes))

            for p in label_set_act:
                p_id, l = p
                temp_actions.append(int(l))

        # align the input data
            while len(temp_boxes) != self.num_boxes:
                temp_boxes.append((0,0,0,0))
                temp_normalized_boxes.append((0,0,0,0))
                temp_actions.append(-1)

            while len(temp_interactions) != int(self.num_boxes*(self.num_boxes-1)/2):
                temp_interactions.append(-1)

            bboxes.append(temp_boxes)
            normalized_bboxes.append(temp_normalized_boxes)
            actions.append(temp_actions)
            interactions.append(temp_interactions)
        
        images = np.stack(images)
        bboxes_num = np.array(bboxes_num, dtype=np.int32)
        bboxes = np.array(bboxes,dtype=float).reshape(-1,self.num_boxes,4)
        normalized_bboxes = np.array(normalized_bboxes,dtype=float).reshape(-1,self.num_boxes,4)
        actions = np.array(actions,dtype=np.int32).reshape(-1,self.num_boxes)
        interactions = np.array(interactions,dtype=np.int32).reshape(-1,int(self.num_boxes*(self.num_boxes-1)/2))
        
        #convert to pytorch tensor
        images = torch.from_numpy(images).float()
        bboxes = torch.from_numpy(bboxes).float()
        normalized_bboxes = torch.from_numpy(normalized_bboxes).float()
        actions = torch.from_numpy(actions).long().squeeze(0)  
        interactions = torch.from_numpy(interactions).long().squeeze(0)
        bboxes_num = torch.from_numpy(bboxes_num).int()
        # print(actions.size(), interactions.size()) # [15] and [105]
        
        
        # print(images.size(), bboxes.size(), interactions.size(), bboxes_num.size()) # torch.Size([1, 3, 540, 960]) torch.Size([1, 4, 4]) torch.Size([6]) torch.Size([1])
        
        # return images, bboxes, bboxes_num, interactions, normalized_bboxes, select_frame
        return images, bboxes, actions, bboxes_num, interactions, seq_name, fid, normalized_bboxes

def UT_get_bboxes(seqs):
    data = {}
    data_path = "/data-ext/UT_dataset/completed/UT_annotations_2/bounding_boxes/"
    for sname in seqs:
        list_bboxes_annos = os.listdir(data_path + sname)
        list_bboxes_annos.sort(key=lambda x: int(re.findall(r'\d+', x)[0]))  # Sort file names in ascending order
        frames = {}
        for f in list_bboxes_annos:
            f_name, _ = os.path.splitext(f)
            list_people = {}
            with open(os.path.join(data_path + sname, f), 'r') as file:
                for line in file:
                    id, x_center, y_center, width, height = line.split()
                    list_people[id] = (float(x_center), float(y_center), float(width), float(height))
                    # print(f"id: {id}, x_center: {x_center}, y_center: {y_center}, width: {width}, height: {height}")
            frames[f_name] = list_people
        data[sname] = frames
    return data

def UT_get_frames(seqs): 
    data = {}
    data_path = "/data-ext/UT_dataset/completed/UT_annotations_2/images/"
    for sname in seqs:
        list_frames = os.listdir(data_path + sname)
        list_frames.sort(key=lambda x: int(re.findall(r'\d+', x)[0]))  # Sort frame names in ascending order
        data[sname] = list_frames
    return data  # dict with key as seq name and value as list of frame names

def UT_get_labels_inter(seqs):
    data = {}
    data_path = "/data-ext/UT_dataset/completed/UT_annotations_2/annotations_inter/"
    count_0, count_1 = 0, 0
    count = 0
    for sname in seqs:
        list_labels_annos = os.listdir(data_path + sname)
        list_labels_annos.sort(key=lambda x: int(re.findall(r'\d+', x)[0]))
        frames = {}
        for f in list_labels_annos:
            f_name, _ = os.path.splitext(f)
            list_pairs = {}
            with open(os.path.join(data_path + sname, f), 'r') as file:
                for line in file:
                    pair, label = line.split()
                    list_pairs[pair] = int(label)
                    count += 1
                    if label == "0":
                        count_0 += 1
                    elif label == "1":
                        count_1 += 1
            frames[f_name] = list_pairs
        data[sname] = frames
    # print("count_inter: ", count)
    return data, count_0, count_1

def UT_get_labels_act(seqs):
    data = {}
    data_path = "/data-ext/UT_dataset/completed/UT_annotations_2/annotations_action/"
    count_0, count_1, count_2, count_3, count_4, count_5 = 0, 0, 0, 0, 0, 0
    count = 0
    for sname in seqs:
        list_labels_annos = os.listdir(data_path + sname)
        list_labels_annos.sort(key=lambda x: int(re.findall(r'\d+', x)[0]))
        frames = {}
        for f in list_labels_annos:
            f_name, _ = os.path.splitext(f)
            list_actions = {}
            with open(os.path.join(data_path + sname, f), 'r') as file:
                for line in file:
                    elements = line.split()
                    person_id = elements[0]
                    labels = elements[1:]
                    if labels == []:
                        labels = "others"
                    else:
                        labels = ' '.join(labels)
                    if labels == "be kicked" or labels == "be punched" or labels == "be pushed" or labels == "others":
                        count_5 += 1
                    elif labels == "handshake":
                        count_0 += 1
                    elif labels == "hug":
                        count_1 += 1
                    elif labels == "kick":
                        count_2 += 1
                    elif labels == "punch":
                        count_3 += 1
                    elif labels == "push":
                        count_4 += 1
                    labels = action_dict[labels] # get int label
                    list_actions[person_id] = int(labels)
                    # print(f"action: {action_dict[line.strip()]}")
                    count += 1
            frames[f_name] = list_actions
        data[sname] = frames
    # print(count)
    return data, count_0, count_1, count_2, count_3, count_4, count_5
# def build_ut_test(cfg, seq_num):
#     tst_seq = ['seq' + str(seq_num)]
#     test_bboxes = UT_get_bboxes(tst_seq)  # a dict
#     test_frames = UT_get_frames(tst_seq)  # a dict
#     test_labels = UT_get_labels(tst_seq)  # a dict

#     validation_set = UTDataset(test_bboxes,
#                                 test_frames,
#                                 test_labels,
#                                 cfg.num_frames,
#                                 cfg.image_size,
#                                 cfg.out_size,
#                                 num_boxes=cfg.num_boxes,
#                                 data_augmentation=False)
#     validation_loader=data.DataLoader(validation_set,
#                                       batch_size=cfg.test_batch_size,
#                                       shuffle=False,
#                                       num_workers=cfg.num_workers,
#                                       )
#     return validation_loader


def build_ut(cfg):
    tst_seq = ['seq' + str(i) for i in range(11, 21)]
    trn_seq = ['seq' + str(i) for i in range(1, 11)]

    train_bboxes = UT_get_bboxes(trn_seq)  # a dict with keys as seq names and values as frame id dicts. In frame id dict, key is frame id and value is a dict with key as person id and value as bbox
    train_frames = UT_get_frames(trn_seq)  # a dict with key as seq name "seq1" and value as list of frame names
    train_labels_inter, count_0_inter, count_1_inter = UT_get_labels_inter(trn_seq)  # a dict with keys as seq names and values as frame id dicts. In frame id dict, key is frame id and value is a dict with key as human pair ("0-1") and value as interaction label
    train_labels_act, count_0, count_1, count_2, count_3, count_4, count_5 = UT_get_labels_act(trn_seq)  # a dict with keys as seq names and values as frame id dicts. In frame id dict, key is frame id and value is a dict with key as person id and value as action label
    # print(f"count_0_inter: {count_0_inter}, count_1_inter: {count_1_inter}")
    # print(f"count_0: {count_0}, count_1: {count_1}, count_2: {count_2}, count_3: {count_3}, count_4: {count_4}, count_5: {count_5}")
    # exit()

    test_bboxes = UT_get_bboxes(tst_seq)  # a dict
    test_frames = UT_get_frames(tst_seq)  # a dict
    test_labels_inter, count_0_inter, count_1_inter = UT_get_labels_inter(tst_seq)  # a dict
    test_labels_act, count_0, count_1, count_2, count_3, count_4, count_5 = UT_get_labels_act(tst_seq)  # a dict
    

    # build train and test sets
    training_set = UTDataset(train_bboxes,
                              train_frames,
                              train_labels_act,
                              train_labels_inter,
                              cfg.num_frames,
                              cfg.image_size,
                              cfg.out_size,
                              num_boxes=cfg.num_boxes,
                              data_augmentation=False)

    validation_set = UTDataset(test_bboxes,
                                test_frames,
                                test_labels_act,
                                test_labels_inter,
                                cfg.num_frames,
                                cfg.image_size,
                                cfg.out_size,
                                num_boxes=cfg.num_boxes,
                                data_augmentation=False)

    training_loader=data.DataLoader(training_set,
                                     batch_size=cfg.batch_size,
                                     shuffle=False,
                                    #  num_workers=cfg.num_workers,
                                     )

    validation_loader=data.DataLoader(validation_set,
                                      batch_size=cfg.test_batch_size,
                                      shuffle=False,
                                    #   num_workers=cfg.num_workers,
                                      )

    print('Build dataset finished...')
    print('%d train samples' % len(train_frames))
    print('%d test samples' % len(test_frames))

    return training_loader,validation_loader

if __name__ == "__main__":
    # A = BIT_read_annotations('BIT', 'kick_0035')
    # print(A)
    build_ut()
