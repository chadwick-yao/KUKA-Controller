import numpy as np
import math
import scipy.spatial.transform as st


def pose_euler2quat(pose):
    if not isinstance(pose, np.ndarray):
        pose = np.array(pose)
    assert pose.shape == (6,) or pose.shape == (7,)
    if pose.shape == (6,):
        tmp_pose = np.zeros(7)
        tmp_pose[:3] = pose[:3]
        tmp_pose[3:] = st.Rotation.from_euler("zyx", pose[3:]).as_quat()
    else:
        tmp_pose = np.zeros(6)
        tmp_pose[:3] = pose[:3]
        tmp_pose[3:] = st.Rotation.from_quat(pose[3:]).as_euler("zyx")
        
    return tmp_pose


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
        mat_R = np.dot(
            np.dot(np.array(euler2mat(x, 0, 0)), np.array(euler2mat(0, y, 0))),
            euler2mat(0, 0, z),
        )
    else:
        raise NotImplementedError

    pose = np.eye(4, 4)
    pose[:3, :3] = mat_R
    pose[:3, 3] = position
    return pose


def rotation_matrix(angle, direction, point=None):
    """
    Returns matrix to rotate about axis defined by point and direction.

    E.g.:
        >>> angle = (random.random() - 0.5) * (2*math.pi)
        >>> direc = numpy.random.random(3) - 0.5
        >>> point = numpy.random.random(3) - 0.5
        >>> R0 = rotation_matrix(angle, direc, point)
        >>> R1 = rotation_matrix(angle-2*math.pi, direc, point)
        >>> is_same_transform(R0, R1)
        True

        >>> R0 = rotation_matrix(angle, direc, point)
        >>> R1 = rotation_matrix(-angle, -direc, point)
        >>> is_same_transform(R0, R1)
        True

        >>> I = numpy.identity(4, numpy.float32)
        >>> numpy.allclose(I, rotation_matrix(math.pi*2, direc))
        True

        >>> numpy.allclose(2., numpy.trace(rotation_matrix(math.pi/2,
        ...                                                direc, point)))
        True

    Args:
        angle (float): Magnitude of rotation
        direction (np.array): (ax,ay,az) axis about which to rotate
        point (None or np.array): If specified, is the (x,y,z) point about which the rotation will occur

    Returns:
        np.array: 4x4 homogeneous matrix that includes the desired rotation
    """
    sina = math.sin(angle)
    cosa = math.cos(angle)
    direction = unit_vector(direction[:3])
    # rotation matrix around unit vector
    R = np.array(
        ((cosa, 0.0, 0.0), (0.0, cosa, 0.0), (0.0, 0.0, cosa)), dtype=np.float32
    )
    R += np.outer(direction, direction) * (1.0 - cosa)
    direction *= sina
    R += np.array(
        (
            (0.0, -direction[2], direction[1]),
            (direction[2], 0.0, -direction[0]),
            (-direction[1], direction[0], 0.0),
        ),
        dtype=np.float32,
    )
    M = np.identity(4)
    M[:3, :3] = R
    if point is not None:
        # rotation not around origin
        point = np.array(point[:3], dtype=np.float32, copy=False)
        M[:3, 3] = point - np.dot(R, point)
    return M


def unit_vector(data, axis=None, out=None):
    """
    Returns ndarray normalized by length, i.e. eucledian norm, along axis.

    E.g.:
        >>> v0 = numpy.random.random(3)
        >>> v1 = unit_vector(v0)
        >>> numpy.allclose(v1, v0 / numpy.linalg.norm(v0))
        True

        >>> v0 = numpy.random.rand(5, 4, 3)
        >>> v1 = unit_vector(v0, axis=-1)
        >>> v2 = v0 / numpy.expand_dims(numpy.sqrt(numpy.sum(v0*v0, axis=2)), 2)
        >>> numpy.allclose(v1, v2)
        True

        >>> v1 = unit_vector(v0, axis=1)
        >>> v2 = v0 / numpy.expand_dims(numpy.sqrt(numpy.sum(v0*v0, axis=1)), 1)
        >>> numpy.allclose(v1, v2)
        True

        >>> v1 = numpy.empty((5, 4, 3), dtype=numpy.float32)
        >>> unit_vector(v0, axis=1, out=v1)
        >>> numpy.allclose(v1, v2)
        True

        >>> list(unit_vector([]))
        []

        >>> list(unit_vector([1.0]))
        [1.0]

    Args:
        data (np.array): data to normalize
        axis (None or int): If specified, determines specific axis along data to normalize
        out (None or np.array): If specified, will store computation in this variable

    Returns:
        None or np.array: If @out is not specified, will return normalized vector. Otherwise, stores the output in @out
    """
    if out is None:
        data = np.array(data, dtype=np.float32, copy=True)
        if data.ndim == 1:
            data /= math.sqrt(np.dot(data, data))
            return data
    else:
        if out is not data:
            out[:] = np.array(data, copy=False)
        data = out
    length = np.atleast_1d(np.sum(data * data, axis))
    np.sqrt(length, length)
    if axis is not None:
        length = np.expand_dims(length, axis)
    data /= length
    if out is None:
        return data


def String2Double(message: str, size: int):
    """Convert string type to double type"""

    strVals = message.split("_")
    assert size <= len(
        strVals
    ), f"You're trying to obtain {size} numbers which is out of range."

    try:
        doubleVals = [float(strVals[idx]) for idx in range(size)]
        return doubleVals
    except ValueError as e:
        raise e("Unsupported value to convert.")
