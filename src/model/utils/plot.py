import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from functools import partial

from .numpy import get_fig_as_nparray, numpy_img_to_tensor

RADIUS = np.array([1,1,1])*0.025 #in m

def create_pose(ax, plots, vals, limbseq, left_right_limb, pred=True, update=False):

    # Start and endpoints of our representation
    I = np.array([touple[0] for touple in limbseq])
    J = np.array([touple[1] for touple in limbseq])
    # Left / right indicator
    LR = np.array([left_right_limb[a] or left_right_limb[b] for a, b in limbseq])
    if pred:
        lcolor = "#9b59b6"
        rcolor = "#2ecc71"
    else:
        lcolor = "#8e8e8e"
        rcolor = "#383838" # dark grey

    for i in np.arange(len(I)):
        x = np.array([vals[I[i], 0], vals[J[i], 0]])
        z = np.array([vals[I[i], 1], vals[J[i], 1]])
        y = np.array([vals[I[i], 2], vals[J[i], 2]])
        if not update:

            if i == 0:
                plots.append(ax.plot(x, y, z, c=rcolor if LR[i] else lcolor,
                                     label=['GT' if not pred else 'Pred'])) #lw=2, linestyle='--',
            else:
                plots.append(ax.plot(x, y, z,  c=lcolor if LR[i] else rcolor)) #lw=2, linestyle='--',

        elif update:
            plots[i][0].set_xdata(x)
            plots[i][0].set_ydata(y)
            plots[i][0].set_3d_properties(z)
            plots[i][0].set_color(lcolor if LR[i] else rcolor)

    return plots


def center_around_hip(vals, ax):
    global RADIUS
    rx, ry, rz = RADIUS[0], RADIUS[1], RADIUS[2]
    # remember that y and z are switched in the plot
    xroot, zroot, yroot = vals[0,0], vals[0,1], vals[0,2]
    ax.set_xlim3d([-rx+xroot, rx+xroot])
    ax.set_ylim3d([-ry+yroot, ry+yroot])
    ax.set_zlim3d([-rz+zroot, rz+zroot])

def update(num,data_gt,plots_gt,fig,ax, data_pred=None, center_pose=True, return_img=False, **kwargs):
    
    gt_vals=data_gt[num]
    
#     pred_vals=data_pred[num]
    plots_gt=create_pose(ax,plots_gt,gt_vals,pred=False,update=True, **kwargs)
    if data_pred is not None:
        vals=data_pred[num]
        plots_gt=create_pose(ax,plots_gt,vals,pred=True,update=True, **kwargs)
#     plots_pred=create_pose(ax,plots_pred,pred_vals,pred=True,update=True)

    if center_pose:
        center_around_hip(gt_vals, ax)
    # ax.set_title('pose at time frame: '+str(num))
    # ax.set_aspect('equal')
    if return_img:
        img = get_fig_as_nparray(fig)
        return img
 
    return plots_gt

# def plot_moving_3d_projection(poses_reshaped3d, out_file='./images/human_viz.gif', center_pose=True, units="mm"):
#     assert len(poses_reshaped3d.shape) == 3
#     assert units in ["mm", "bins"]
#     vals = poses_reshaped3d.copy()
#     timesteps = vals.shape[0]
#     fig = plt.figure(figsize=(12, 12))

#     ax = plt.axes(projection='3d')
#     # vals[:,:,0] = poses_reshaped3d[:,:,2].copy()
#     # vals[:,:,2] = poses_reshaped3d[:,:,0].copy()
#     if units == "mm":
#         vals /= 1000 # from mm to meters
#     vals*= np.array([1,-1,1]) #invert point on vertical axis(body height), because the axis points downwards in plt

#     gt_plots=[]
#     pred_plots=[]

#     gt_plots=create_pose(ax,gt_plots,vals[0],pred=False,update=False)

#     ax.set_xlabel("x")
#     ax.set_ylabel("z") 
#     ax.set_zlabel("y")

#     # ax.set_xlim3d([-1, 3])
#     # ax.set_ylim3d([0, 4])
#     # ax.set_zlim3d([-1, 3])
#     global RADIUS 
#     RADIUS = vals[0].max(axis=0)//2 # symmetric radius distance from body center
#     RADIUS[RADIUS==0] = 1
#     center_around_head(vals[0], ax)


#     line_anim = animation.FuncAnimation(fig, partial(update, center_pose=center_pose), 
#                                         timesteps, fargs=(vals,gt_plots,
#                                                                        fig,ax),interval=4000//timesteps, blit=False)
#     plt.show()
#     line_anim.save(out_file,writer='pillow')
#     return line_anim


def get_np_frames_3d_projection(poses_reshaped3d, limbseq, left_right_limb, data_pred=None, xyz_range=None, is_range_sym=False, center_pose=False, units="mm", 
                                as_tensor=False, orientation_like=None, title=None, center_like=None):
    assert len(poses_reshaped3d.shape) == 3, poses_reshaped3d.shape
    assert units in ["mm", "bins"]
    assert orientation_like is None or orientation_like in ["motron", "h36m"]
    vals = poses_reshaped3d.copy()
    if xyz_range is not None:
        xyz_range = xyz_range.clone()
    timesteps = vals.shape[0]
    fig = plt.figure(figsize=(8, 8))

    ax = plt.axes(projection='3d')
    if title is not None:
        ax.set_title(title)
    # vals[:,:,0] = poses_reshaped3d[:,:,2].copy()
    # vals[:,:,2] = poses_reshaped3d[:,:,0].copy()
    if units == "mm":
        vals /= 1000 # from mm to meters
        
    if data_pred is not None:
        data_pred = data_pred.copy()
        if units == "mm":
            data_pred/= 1000
    
    gt_plots=[]
    pred_plots=[]

    gt_plots=create_pose(ax,gt_plots,vals[0],pred=False,update=False, limbseq=limbseq, left_right_limb=left_right_limb)

    ax.set_xlabel("x")
    ax.set_ylabel("z") 
    ax.set_zlabel("y")

    global RADIUS 
    if xyz_range is not None:
        assert center_pose == False
        if units=="mm":
            xyz_range /= 1000
        if not is_range_sym:
            ax.set_xlim3d([0, xyz_range[0]])
            ax.set_ylim3d([0, xyz_range[2]])
            ax.set_zlim3d([0, xyz_range[1]])
        else: 
            ax.set_xlim3d([-xyz_range[0]/2, xyz_range[0]/2])
            ax.set_ylim3d([-xyz_range[2]/2, xyz_range[2]/2])
            ax.set_zlim3d([-xyz_range[1]/2, xyz_range[1]/2])
    elif center_like is not None:
        if units=="mm":
            RADIUS = (center_like[0].max(axis=0)/1000)//2 # symmetric radius distance from body center
            RADIUS[RADIUS==0] = 1
            center_around_hip(center_like[0]/1000, ax)
        else:
            RADIUS = center_like[0].max(axis=0)//2 # symmetric radius distance from body center
            RADIUS[RADIUS==0] = 1
            center_around_hip(center_like[0], ax)
    else:   
        RADIUS = vals[0].max(axis=0)//2 # symmetric radius distance from body center
        RADIUS[RADIUS==0] = 1
        center_around_hip(vals[0], ax)
    
    # if (vals[...,1]<0.).all():
    #     ax.invert_zaxis()
    ax.set_box_aspect([1,1,1])

    if orientation_like is not None:
        if orientation_like =="h36m":
            ax.view_init(20, -70) # View angle from cameras in h36m
        elif orientation_like =="motron":
            ax.view_init(12, 48) # motron h36m view angle
    
    if as_tensor:
        frames = [numpy_img_to_tensor(update(t,vals,gt_plots,fig,ax, data_pred=data_pred, center_pose=center_pose, return_img=True, limbseq=limbseq, left_right_limb=left_right_limb)) for t in range(len(vals))]
    else:
        frames = [update(t,vals,gt_plots,fig,ax, data_pred=data_pred, center_pose=center_pose, return_img=True, limbseq=limbseq, left_right_limb=left_right_limb) for t in range(len(vals))]
    plt.close()
    return frames
