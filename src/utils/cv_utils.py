import cv2

def rescale_frame(frame, scale=1):
    """
    Scale a frame by a given percentage
    """
    if scale == 1:
        return frame
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
