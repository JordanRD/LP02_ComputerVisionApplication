import numpy as np
import cv2

def corner_score(image, u=5, v=5, window_size=(5, 5)):
    """
    Menghitung Sum of Squared Differences (SSD) untuk deteksi sudut.
    Menggunakan zero-padding untuk menangani nilai window di luar batas citra.
    """
    H, W = image.shape
    shifted_image = np.zeros_like(image, dtype=np.float32)
    
    start_y, end_y = max(0, v), min(H, H + v)
    start_x, end_x = max(0, u), min(W, W + u)
    src_start_y, src_end_y = max(0, -v), min(H, H - v)
    src_start_x, src_end_x = max(0, -u), min(W, W - u)
    
    shifted_image[start_y:end_y, start_x:end_x] = image[src_start_y:src_end_y, src_start_x:src_end_x]
    
    # Menghitung perbedaan kuadrat
    squared_diff = (shifted_image - image.astype(np.float32)) ** 2
    
    # Mengaplikasikan window dengan box filter
    output = cv2.boxFilter(squared_diff, -1, window_size, normalize=False, borderType=cv2.BORDER_CONSTANT)
    
    return output

def compute_image_gradients(image):
    """
    Menghitung gradien citra Ix, Iy dan perkaliannya (Ix^2, Iy^2, IxIy)
    menggunakan operator Sobel untuk persiapan matriks Harris.
    """
    Ix = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    
    Ixx = Ix ** 2
    Iyy = Iy ** 2
    Ixy = Ix * Iy
    
    return Ix, Iy, Ixx, Iyy, Ixy
