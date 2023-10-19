import numpy as np

def get_pose_mat(pose):
    position, orientation = pose
    if len(orientation) == 4:
        from transforms3d.quaternions import quat2mat
        x, y, z, w = orientation
        quat_new = [w, x, y, z]
        mat_R = quat2mat(quat_new)
    elif len(orientation) == 3:
        from transforms3d.euler import euler2mat
        x, y, z = orientation
        mat_R = np.dot(np.dot(np.array(euler2mat(x, 0, 0)), np.array(euler2mat(0, y, 0))), euler2mat(0, 0, z))
    else:
        raise NotImplementedError

    pose = np.eye(4, 4)
    pose[:3, :3] = mat_R
    pose[:3, 3] = position
    return pose