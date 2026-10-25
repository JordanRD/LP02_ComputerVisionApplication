import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np

# ===================== Soal 1 (Dicky) =====================


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

    shifted_image[start_y:end_y, start_x:end_x] = image[
        src_start_y:src_end_y, src_start_x:src_end_x
    ]

    # Menghitung perbedaan kuadrat
    squared_diff = (shifted_image - image.astype(np.float32)) ** 2

    # Mengaplikasikan window dengan box filter
    output = cv2.boxFilter(
        squared_diff, -1, window_size, normalize=False, borderType=cv2.BORDER_CONSTANT
    )

    return output


def compute_image_gradients(image):
    """
    Menghitung gradien citra Ix, Iy dan perkaliannya (Ix^2, Iy^2, IxIy)
    menggunakan operator Sobel untuk persiapan matriks Harris.
    """
    Ix = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    Ixx = Ix**2
    Iyy = Iy**2
    Ixy = Ix * Iy

    return Ix, Iy, Ixx, Iyy, Ixy


# ===================== Soal 2 (Raka & Naufal) =====================


def compute_structure_tensor(
    Ixx, Iyy, Ixy, window_size=(5, 5), use_gaussian=True, sigma=1.0
):
    """
    Menghitung Struktur Matriks Harris (M) per piksel.

    Input:
        Ixx, Iyy, Ixy : produk gradien (hasil dari Soal 1)
        window_size   : ukuran window (tuple), contoh (5,5)
        use_gaussian  : True = Gaussian window, False = Box filter
        sigma         : sigma untuk Gaussian (hanya dipakai jika use_gaussian=True)

    Output:
        A, B, C       : hasil windowing dari Ixx, Iyy, Ixy
                        sehingga M = [[A, C], [C, B]] di setiap piksel
    """
    if use_gaussian:
        # Pastikan kernel size ganjil
        ksize = window_size[0] if window_size[0] % 2 == 1 else window_size[0] + 1
        A = cv2.GaussianBlur(Ixx, (ksize, ksize), sigma)
        B = cv2.GaussianBlur(Iyy, (ksize, ksize), sigma)
        C = cv2.GaussianBlur(Ixy, (ksize, ksize), sigma)
    else:
        # Box filter (sum dalam window, tanpa normalize)
        A = cv2.boxFilter(Ixx, -1, window_size, normalize=False)
        B = cv2.boxFilter(Iyy, -1, window_size, normalize=False)
        C = cv2.boxFilter(Ixy, -1, window_size, normalize=False)

    return A, B, C


def harris_detector(image, window_size=(5, 5), k=0.04, use_gaussian=True, sigma=1.0):
    """
    Menghitung skor Harris R = det(M) - k * trace(M)^2 untuk semua piksel.
    Output berupa citra HxW.
    """
    if image.dtype != np.float32 and image.dtype != np.float64:
        image = image.astype(np.float32)

    # Gradien dari Soal 1
    _, _, Ixx, Iyy, Ixy = compute_image_gradients(image)

    # Struktur tensor M
    A, B, C = compute_structure_tensor(Ixx, Iyy, Ixy, window_size, use_gaussian, sigma)

    # Corner response R
    det_M = A * B - C * C
    trace_M = A + B
    response = det_M - k * (trace_M**2)

    return response


def extract_harris_corners(response, threshold_ratio=0.01, nms_size=3):
    """
    Thresholding dan Non-Maximum Suppression (NMS) pada response R.
    Output: koordinat sudut (x, y) dan mask sudut.
    """
    threshold_value = threshold_ratio * response.max()

    kernel = np.ones((nms_size, nms_size), np.uint8)
    local_max = cv2.dilate(response, kernel)

    corner_mask = (response == local_max) & (response > threshold_value)

    corners_y_x = np.argwhere(corner_mask)
    corners_x_y = np.fliplr(corners_y_x)

    return corners_x_y, corner_mask


# ===================== Soal 3 (Fergiant) =====================


def draw_corners(image_bgr, corners, radius=3, color=(0, 0, 255)):
    """
    Menggambar titik sudut (x, y) pada salinan citra BGR.
    """
    output = image_bgr.copy()
    for x, y in corners:
        cv2.circle(output, (int(x), int(y)), radius, color, -1)
    return output


def visualize(gray, score, response, corner_img, u, v, output_path="harris_result.png"):
    """
    Menampilkan citra asli, skor SSD E(u,v), response Harris R, dan hasil deteksi sudut.
    """
    plt.figure(figsize=(16, 4))

    plt.subplot(1, 4, 1)
    plt.title("Citra Asli (Grayscale)")
    plt.imshow(gray, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.title(f"Corner Score E(u={u}, v={v})")
    plt.imshow(score, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.title("Harris Response R")
    plt.imshow(response, cmap="hot")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.title("Sudut Terdeteksi")
    plt.imshow(cv2.cvtColor(corner_img, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.show()


def main(filename):
    u, v = 5, 5
    W = (5, 5)

    img = cv2.imread(filename)
    if img is None:
        raise FileNotFoundError(f"Citra tidak ditemukan: {filename}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # save score
    score = corner_score(gray, u, v, W)

    # save Harris Corner Detector
    response = harris_detector(gray, W)
    corners, _ = extract_harris_corners(response, threshold_ratio=0.05, nms_size=5)

    print("Shape citra      :", gray.shape)
    print("Shape score      :", score.shape)
    print("Shape response   :", response.shape)
    print("Jumlah sudut     :", len(corners))

    corner_img = draw_corners(img, corners)
    cv2.imwrite("harris_corners.png", corner_img)
    visualize(gray, score, response, corner_img, u, v)

    return score, response, corners


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "building.png")
