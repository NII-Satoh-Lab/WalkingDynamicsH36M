#####################################################################################################################
#### Human3.6M joint representation
#####################################################################################################################

# for human36m. Dataset gives 32 jionts, with duplicates
JOINTS_DICT_H36M_32 = {0: "Hip",
                       1: "RHip", 2: "RKnee", 3: "RAnkle", 4: "RFoot", 5: "RToes",
                       6: "LHip", 7: "LKnee", 8: "LAnkle", 9: "LFoot", 10: "LToes",
                       11: "Hip",
                       12: "Torso", 13: "Neck", 14: "Nose", 15: "Head",
                       16: "Nose",
                       17: "LShoulder", 18: "LElbow", 19: "LWrist",
                       20: "LHand", 21: "LSmallFinger", 22: "LCenterFinger", 23: "LThumb",
                       24: "Nose",
                       25: "RShoulder", 26: "RElbow", 27: "RWrist",
                       28: "RHand", 29: "RSmallFinger", 30: "RCenterFinger", 31: "RThumb", }

# representation without duplicates. 25 joints. Duplicates are described in notebook
CONVERSION_IDX_H36M_32TO25 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19, 21, 22, 25, 26, 27, 29, 30]
JOINTS_DICT_H36M_25 = {0: "Hip",
                       1: "RHip", 2: "RKnee", 3: "RAnkle", 4: "RFoot", 5: "RToes",
                       6: "LHip", 7: "LKnee", 8: "LAnkle", 9: "LFoot", 10: "LToes",
                       11: "Torso", 12: "Neck", 13: "Nose", 14: "Head",
                       15: "LShoulder", 16: "LElbow", 17: "LWrist",
                       18: "LSmallFinger", 19: "LThumb",
                       20: "RShoulder", 21: "RElbow", 22: "RWrist",
                       23: "RSmallFinger", 24: "RThumb"}
CENTER_JOINT_H36M_25 = 0

# dict_h36m_25_to_openpose = [8, 9, 10, 11, 24, 22, 12, 13, 14, 21, 19, 8, 1, 0, 0, 5, 6, 7, 7, 7, 2, 3, 4, 4, 4]
COLORS_H36M_25 = [[255,   0,   0],  [  0, 255,  85], [  0, 255, 170],
                   [  0, 255, 255], [  0, 255, 255], [  0, 255, 255],
                   [  0, 170, 255], [  0,  85, 255], [  0,   0, 255],
                   [  0,   0, 255], [  0,   0, 255], [255,   0,   0], [255,   0,   0],
                   [255,   0,  85], [255,   0,  85],  [170, 255,   0],
                   [ 85, 255,   0],  [  0, 255,   0],  [  0, 255,   0],
                   [  0, 255,   0], [255,  85,   0], [255, 170,   0],
                   [255, 255,   0], [255, 255,   0], [255, 255,   0]]
LIMBSEQ_H36M_25 = [[0, 1], [0,6], [0,11],
                   [1,2], [2,3], [3,4], [4,5],# right foot
                   [6,7], [7,8], [8,9], [9,10],# left foot
                   [11,12], [12,13], [13,14], # head
                   [12, 15], [12,20],
                   [15,16], [16,17], [17,18], [17,19], # left hand
                   [20,21], [21,22], [22,23], [22,24] # right hand
                   ]

# removing feet and hands results in a 17 representation, which DIFFERS from other 17-joints representations (COCO, openpose)
CONVERSION_IDX_H36M_32TO17 = [0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27]
JOINTS_DICT_H36M_17 =  {0: "Hip",
                       1: "RHip", 2: "RKnee", 3: "RAnkle",
                       4: "RFoot ", 5: "LHip", 6: " LKnee",
                       7: "LAnkle", 8: "LFoot", 9: "Neck", 10: "Head",
                       11: "LShoulder", 12: "LElbow", 13: "LWrist",
                       14: "RShoulder", 15: "RElbow", 16: "RWrist"}
CENTER_JOINT_H36M_17 = 0
LIMBSEQ_H36M_17 = [[0, 1], [0,5],
                   [1,2], [2,3], [3,4], # right foot
                   [5,6], [6,7], [7,8],# left foot
                   [0,9], [9,10], [9,11], [9,14], # head
                   [11,12], [12,13], # left hand
                    [14,15], [15,16] # right hand
                   ]

assert len(JOINTS_DICT_H36M_32) == 32
assert len(JOINTS_DICT_H36M_25) == 25
assert len(CONVERSION_IDX_H36M_32TO25) == 25
assert len(JOINTS_DICT_H36M_17) == 17
assert len(CONVERSION_IDX_H36M_32TO17) == 17

#####################################################################################################################
#### BODY25 joint representation from openpose
#####################################################################################################################
# --> IDEA: You can compute torso and head from coordinates. Then you have more coordinates for 17joints. Compute also hands and feet
JOINTS_DICT_OPENPOSE = {0: "Nose", 1: "Neck",
                        2: "RShoulder", 3: "RElbow", 4: "RWrist",
                        5: "LShoulder", 6: "LElbow", 7: "LWrist",
                        8: "MidHip", 9: "RHip", 10: "RKnee", 11: "RAnkle", 12: "LHip", 13: "LKnee", 14: "LAnkle",
                        15: "REye", 16: "LEye", 17: "REar", 18: "LEar",
                        19: "LBigToe", 20: "LSmallToe", 21: "LHeel", 22: "RBigToe", 23: "RSmallToe", 24: "RHeel",
                        25: "Background"}
CENTER_JOINT_OPENPOSE = 8
assert len(JOINTS_DICT_OPENPOSE) == 25+1

COLORS_OPENPOSE = [[255, 0, 85], [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0],  # yellow
                   [170, 255, 0], [85, 255, 0], [0, 255, 0],  # green
                   [255, 0, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255],  # cyan
                   [0, 170, 255], [0, 85, 255], [0, 0, 255],  # blue
                   [255, 0, 170],  # 15
                   [170, 0, 255],  # 16
                   [255, 0, 255],  # 17
                   [85, 0, 255],  # 18
                   [0, 0, 255], [0, 0, 255], [0, 0, 255],  # right foot
                   [0, 255, 255], [0, 255, 255], [0, 255, 255]  # left foot
                   ]
LIMBSEQ_OPENPOSE = [[0, 1], [0, 15], [0, 16], [1, 2], [1, 5], [1, 8], [15, 17], [16, 18],  # head
                    [2, 3], [3, 4],  # left arm
                    [5, 6], [6, 7],  # right arm
                    [8, 9], [9, 10], [10, 11], [11, 24], [11, 22], [22, 23],
                    [8, 12], [12, 13], [13, 14], [14, 21], [14, 19], [19, 20]]

assert len(COLORS_OPENPOSE) == 25
#####################################################################################################################
#### Unified representation for HUman3.6M and BODY25 from openpose. With 17 joints.
#### It is possible to add another joint (torso) between hip and neck. H36m already has it, for openpose it
#### can be computed from middle hip and neck coordinates. But to do that, you need to have hip and neck of all poses
#####################################################################################################################

CONVERSION_IDX_H36M_32TO17_UNIFIED = [0, 1, 2, 3, 5, 6, 7, 8, 10, 13, 14, 17, 18, 19, 25, 26, 27]
CONVERSION_IDX_OPENPOSE_TO17_UNIFIED = [8, 9, 10, 11, 22, 12, 13, 14, 19, 1, 0, 5, 6, 7, 2, 3, 4]

JOINTS_DICT_17_UNIFIED = {0: "Hip",
                          1: "RHip", 2: "RKnee", 3: "RAnkle", 4: "RToes",
                          5: "LHip", 6: "LKnee", 7: "LAnkle", 8: "LToes",
                          9: "Neck", 10: "Nose",
                          11: "LShoulder", 12: "LElbow", 13: "LWrist",
                          14: "RShoulder", 15: "RElbow", 16: "RWrist"}
CENTER_JOINT_17_UNIFIED = 0
assert len(JOINTS_DICT_17_UNIFIED) == 17
assert len(CONVERSION_IDX_OPENPOSE_TO17_UNIFIED) == 17
assert len(CONVERSION_IDX_H36M_32TO17_UNIFIED) == 17

COLORS_17_UNIFIED = [[255, 0, 0],
                     [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 255, 255],  # right leg
                     [0, 170, 255], [0, 85, 255], [0, 0, 255], [0, 0, 255],  # left leg
                     [255, 0, 0], [255, 0, 85],
                     [170, 255, 0], [85, 255, 0], [0, 255, 0],  # left arm
                     [255, 85, 0], [255, 170, 0], [255, 255, 0]]  # right arm

LIMBSEQ_17_UNIFIED = [[0, 1], [0, 5], [0, 9], [1, 2], [2, 3], [3, 4], [5, 6], [6, 7], [7, 8],
                      [9, 10], [9, 11], [9, 14], [11, 12], [12, 13], [14, 15], [15, 16]]

#####################################################################################################################
### 3D Poses in the Wild 2D dataset with 18 body joints
#####################################################################################################################
# Same sequence as BODY25 joint representation from openpose, without middle hip, fingers and toes

JOINTS_DICT_3DPW = {0: 'Nose', 1: 'Neck',
                     2: 'RShoulder', 3: 'RElbow', 4: 'RWrist',
                     5: 'LShoulder', 6: 'LElbow', 7: 'LWrist',
                     8: 'RHip', 9: 'RKnee', 10: 'RAnkle',
                     11: 'LHip', 12: 'LKnee', 13: 'LAnkle',
                     14: 'REye', 15: 'LEye', 16: 'REar', 17: 'LEar'}

LIMBSEQ_3DPW = [[0, 1], [0, 14], [0, 15], [1, 2], [1, 5], [14, 16], [15, 17], # head
                 [2, 3], [3, 4], # left arm
                 [5, 6], [6, 7], # right arm
                 [2,8], [5,11], [8, 9], [9, 10], [11, 12], [12, 13]] # legs

COLORS_3DPW = [[255, 0, 85], [255, 0, 0],
                [255, 170, 0], [255, 255, 0], [255, 85, 0],
                [85, 255, 0], [0, 255, 0], [170, 255, 0],
                [0, 255, 170], [0, 255, 255], [0, 255, 85],
               [0, 85, 255], [0, 0, 255], [0, 170, 255],
               [255, 0, 170],   [170, 0, 255],  [255, 0, 255],  [85, 0, 255]]

#####################################################################################################################
### 3D Poses in the Wild 3D dataset with 24 body joints
#####################################################################################################################
# Same sequence as BODY25 joint representation from openpose, without middle hip, fingers and toes

JOINTS_DICT_SMPL_24  = {
                0: 'Pelvis', 1: "LHip", 2: "RHip", 
                3: 'Waist', 6: 'Torax', 9: 'Chest', 
                4: 'LKnee', 5: 'RKnee',
                7: 'LAnkle', 8: 'RAnkle',
                10: 'LToes', 11: 'RToes', 
                12: "Neck", 
                13: "LClavicle", 14: "RClavicle",
                15: 'Nose',
                16: "LShoulder", 17: "RShoulder", 18: "LElbow", 19: "RElbow",
                20: "LWrist", 22: "LHand",
                21: "RWrist", 23: "RHand",
                }

JOINTS_DICT_3DPW_3D = JOINTS_DICT_SMPL_24 # joint name and correspondence for 3DPW, 3D coordinates
CONVERSION_IDX_SMPL24_TO_SOMOF = [1, 2, 4, 5, 7, 8, 12, 16, 17, 18, 19, 20, 21]

#####################################################################################################################
### SoMoF PoseTrack dataset with 14 body joints
#####################################################################################################################
JOINTS_DICT_SOMOF_PT = {0: 'Neck', 1: 'Head',
                     2: 'LShoulder', 3: 'RShoulder',
                     4: 'LElbow', 5: 'RElbow', 6: 'LWrist', 7: 'RWrist',
                     8: 'LHip', 9: 'RHip',
                     10: 'LKnee',  11: 'RKnee', 12: 'LAnkle', 13: 'RAnkle'}

COLORS_SOMOF_PT = [[255, 0, 0], [255, 0, 0],
                [170, 255, 0], [255, 85, 0],
                [85, 255, 0], [255, 170, 0], [0, 255, 0], [255, 255, 0],
                [85, 0, 255], [255, 0, 255],  # 18 # 17
#                 [0, 170, 255], [0, 255, 85],  # let them have colrs similar as 17: "REar", 18: "LEar",
                [0, 85, 255], [0, 255, 170], [0, 0, 255], [0, 255, 255]]

LIMBSEQ_SOMOF_PT = [[0, 1], [0, 2], [0, 3], #neck and shoulders
                 [2, 4], [4, 6], # left arm
                 [3, 5],  [5, 7], #right arm
                 [8, 9], # hips
                 [2, 8], [8, 10], [10, 12], #left leg
                 [3, 9], [9, 11], [11, 13]] #right leg
#####################################################################################################################
## Utilities for creating new constant configuaritions quickly
#####################################################################################################################
# 1) Try to find kpts_dict with
#     fig, ax = plt.subplots()
#     fig.set_size_inches(18, 12)
#     ax.imshow(plt.imread(join(...)))
#     ax.scatter(coords[:, 0], coords[:,1])
#     red_idx = 2
#     ax.scatter(coords[red_idx, 0], coords[red_idx,1], c='r')
#
#     for i, txt in enumerate(kpts_dict.values()):
#         if i <= red_idx:
#             ax.annotate(txt, (int(coords[i, 0]), int(coords[i, 1])), c='k')
# 2) Try to find correspondences with other dicts to have it easier with colors for example
#
# limbs_seq_vanilla = [('Neck', 'Head'),( 'Neck', 'LShoulder'), ('Neck', 'RShoulder'),
#         ('LShoulder', 'LElbow'), ('LElbow', 'LWrist'),
#         ('RShoulder', 'RElbow'), ('RElbow', 'RWrist'),
#          ('LHip', 'RHip'),
#         ('LShoulder', 'LHip'), ('LHip', 'LKnee'), ('LKnee', 'LAnkle'),
#         ('RShoulder', 'RHip'), ('RHip', 'RKnee'), ('RKnee', 'RAnkle')]
# JOINTS_DICT_inverted = {v:k for k,v in JOINTS_DICT.items()}
# LIMBSEQ = [[JOINTS_DICT_inverted[tup[0]],JOINTS_DICT_inverted[tup[1]]]  for tup in limbs]
#
#####################################################################################################################

def get_joint_representation(dataset='unified'):
    if dataset == 'unified':
        return LIMBSEQ_17_UNIFIED, JOINTS_DICT_17_UNIFIED, COLORS_17_UNIFIED
    elif dataset == 'openpose':
        return LIMBSEQ_OPENPOSE, JOINTS_DICT_OPENPOSE, COLORS_OPENPOSE
    elif dataset == 'h36m':
        return LIMBSEQ_H36M_25, JOINTS_DICT_H36M_25, COLORS_H36M_25
    elif dataset == 'h36m-32':
        return LIMBSEQ_H36M_32, JOINTS_DICT_H36M_32, COLORS_H36M_32
    elif dataset == 'somof_pt':
        return LIMBSEQ_SOMOF_PT, JOINTS_DICT_SOMOF_PT, COLORS_SOMOF_PT
    elif dataset == '3dpw':
        return LIMBSEQ_3DPW, JOINTS_DICT_3DPW, COLORS_3DPW
    else:
        assert 0, "Not implemented"

def get_center_joint_idx(dataset='unified'):
    if dataset == 'unified':
        return CENTER_JOINT_17_UNIFIED
    elif dataset == 'openpose':
        return CENTER_JOINT_OPENPOSE
    elif dataset == 'h36m':
        return CENTER_JOINT_H36M_17 # is always first joint regardless of n of ojints
    else:
        assert 0, "Not implemented"

def get_neck_hip_idx(dataset='h36m', n_joint=17):
    if dataset == 'unified':
        assert n_joint == 17
        return 9, 0 #("Neck", "Hip")
    elif dataset == 'openpose':
        assert n_joint == 25
        return 8,1 # ("Neck",  "MidHip")
    elif dataset == 'h36m':
        if n_joint == 17:
            return 9, 0  # ("Neck", "Hip") # return from unified
        elif n_joint == 25:
            assert 0, 'Check wheather to return torso or neck idx'
        else:
            assert 0, "Not implemented"
    elif dataset == 'somof_pt':
        assert n_joint == 14
        return 0,8 # Neck, left hip # NOte tha this is not extremely correct!
    elif dataset == '3dpw':
        assert n_joint == 18
        return 1, 11 # Neck, left hip # NOte tha this is not extremely correct!
    else:
        assert 0, "Not implemented"

def get_right_left_hip_idxs(dataset='h36m', n_joint=17):
    if dataset == 'unified':
        assert n_joint == 17
        return 1, 5  # ("RHip", "LHip")
    elif dataset == 'openpose':
        assert n_joint == 25
        assert 0, "Not implemented"
    elif dataset == 'h36m':
        if n_joint == 17:
            return 1, 5  # ("RHip", "LHip") # return from unified
        elif n_joint == 25:
            assert 0, "Not implemented"
        else:
            assert 0, "Not implemented"
    elif dataset == 'somof_pt':
        assert n_joint == 14
        return 9, 8
    elif dataset == '3dpw':
        assert n_joint == 18
        return 8, 11
    else:
        assert 0, "Not implemented"

"""
In the following functions
- joints have shape (..., 25, 2) where 2 stand for x,y
- heatmaps have shape (25, ...) for example (25, H, W)
where 25 is the initial number of joints
"""

def convert_openpose_joints_to17(joints):
    return joints[..., CONVERSION_IDX_OPENPOSE_TO17_UNIFIED, :]


def convert_openpose_heatmaps_to17(heatmaps):
    assert heatmaps.shape[0] == 25
    return heatmaps[CONVERSION_IDX_OPENPOSE_TO17_UNIFIED, :]


def convert_h36m_joints_to17(joints):
    return joints[..., CONVERSION_IDX_H36M_32TO17_UNIFIED, :]


def convert_h36m_joints_to25(joints):
    return joints[..., CONVERSION_IDX_H36M_32TO25, :]
