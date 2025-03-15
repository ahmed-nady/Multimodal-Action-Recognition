
import cv2
import os
import numpy as np
import concurrent.futures
def split_vid(vid_path,vid_actions):

    vid_cap = cv2.VideoCapture(vid_path)
    vid_name = vid_path.split('/')[-1][:-4]
    frm_num = -1
    fps = vid_cap.get(cv2.CAP_PROP_FPS)
    width = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  
    saved_actions=0
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
  
    action_idx = 0
    action_info = vid_actions[action_idx]
    action_label,start_frm,end_frm=action_info[0],action_info[1],action_info[2]
    action_frms =[]
    while True:
        flag,frame = vid_cap.read()
        if not flag:
            print('vid is completed',vid_name)
            break
        frm_num +=1
        if frm_num >= start_frm and frm_num <= end_frm:
            action_frms.append(frame)
        if frm_num==end_frm:
            #===save vid action and then get next one===#
            save_vid_path = os.path.join(dataset_vids_actions_dir,vid_name+f"-A{action_label:02}-N{action_idx+1:02}"+'.mp4')
            vid_out = cv2.VideoWriter(save_vid_path, fourcc, fps, (width, height))
            for frm in action_frms:
                vid_out.write(frm)
            vid_out.release()

            action_frms.clear()
            saved_actions += 1
            #==get next action details==#
            action_idx += 1
            if action_idx < len(vid_actions):

                action_info = vid_actions[action_idx]
                action_label, start_frm, end_frm = action_info[0], action_info[1], action_info[2]
            else:
                #===exit ===#
                vid_cap.release()
                assert len(vid_actions)==saved_actions,f"{vid_name}: num_actions {len(vid_actions)}, saved_actions:{saved_actions}"
                break

def split_vid_iterating_actions(vid_path,vid_actions):

    vid_cap = cv2.VideoCapture(vid_path)
    vid_name = vid_path.split('/')[-1][:-4]
    frm_num = -1
    fps = 20.0#vid_cap.get(cv2.CAP_PROP_FPS)
    width = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    saved_actions=0
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    for action_idx in range(len(vid_actions)):
        action_info = vid_actions[action_idx]
        action_label, start_frm, end_frm = action_info[0], action_info[1], action_info[2]
        # ===save vid action and then get next one===#
        save_vid_path = os.path.join(dataset_vids_actions_dir,
                                     vid_name + f"-A{action_label:02}-N{action_idx + 1:02}" + '.mp4')
        vid_out = cv2.VideoWriter(save_vid_path, fourcc, fps, (width, height))
        while True:
            flag,frame = vid_cap.read()
            if not flag:
                #print('vid is completed',vid_name)
                break
            frm_num +=1
            if frm_num >= start_frm and frm_num <= end_frm:
                vid_out.write(frame)
            if frm_num==end_frm:
                vid_out.release()
                saved_actions += 1
                break
    vid_out.release()
    vid_cap.release()

    if len(vid_actions)!=saved_actions:
        print(f'vid:{vid_name} has gt_actions:{len(vid_actions)},saved ones: {saved_actions},frm_num:{frm_num}')
    #assert len(vid_actions)==saved_actions,f'vid:{vid_name} has gt_actions:{len(vid_actions)},saved ones: {saved_actions}'
def parse_vid_labels(vid_label):
    vid_actions =[]
    with open(vid_label,'r') as f:
        action_lst= f.readlines()
        for action_data in action_lst:
            action_data = action_data.strip().split(',')[:-1]
            vid_actions.append([int(action_data[0]),int(action_data[1]),int(action_data[2])])
    vid_actions = sorted(vid_actions, key=lambda x: x[1])
    return vid_actions

def test_vid():
    vid_label = '/ActionData/PKU-MMD/Train_Label_PKU_final/0043-L.txt'
    vid_actions = parse_vid_labels(vid_label)
    vid_path = os.path.join(dataset_vid_dir,'0043-L.avi')
    split_vid_iterating_actions(vid_path,vid_actions)

if __name__=='__main__':


    dataset_label_dir = '/ActionData/PKU-MMD/Train_Label_PKU_final/'
    dataset_vid_dir = '/ActionData/PKU-MMD/RGB_VIDEO/'
    dataset_vids_actions_dir ='/ActionData/PKU-MMD/RGB_VIDEO_ACTIONS'
    vid_labels = os.listdir(dataset_label_dir)
    datset_actions =[]

    multi_threading = False
    if not multi_threading:
        for i in range(len(vid_labels)):
            #print(f"processing vid: {vid_labels[i][:-4]}")
            vid_label = os.path.join(dataset_label_dir,vid_labels[i])
            vid_actions = parse_vid_labels(vid_label)
            datset_actions.extend(vid_actions)
            continue
            vid_path = os.path.join(dataset_vid_dir,vid_labels[i][:-3]+'avi')
            split_vid(vid_path,vid_actions)

        print(len(datset_actions))
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=30) as executor:
            for i in range(len(vid_labels)):
                #print(f"processing vid: {vid_labels[i][:-4]}")
                vid_label = os.path.join(dataset_label_dir, vid_labels[i])
                vid_actions = parse_vid_labels(vid_label)

                vid_path = os.path.join(dataset_vid_dir, vid_labels[i][:-3] + 'avi')
                executor.submit(split_vid_iterating_actions,vid_path, vid_actions)

