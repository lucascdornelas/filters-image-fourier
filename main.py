import cv2
from matplotlib import pyplot as plt
import numpy as np

if __name__ == '__main__':

    img_path = "new_data/c2.jpg"
    img = cv2.imread(img_path, 0) # load an image

    # A saída é uma matriz complexa 2D. 1º canal real e 2º imaginário
    # Para fft in opencv, a imagem de entrada precisa ser convertida para float32
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)

    # Reorganiza uma transformada de Fourier X mudando a frequência zero
    # componente ao centro da matriz.
    # Caso contrário, começa no topo esquerdo do corenr da imagem (array)
    dft_shift = np.fft.fftshift(dft)

    # Magnitude da função é 20.log (abs (f))
    # Para valores que são 0, podemos acabar com valores indeterminados para log.
    # Portanto, podemos adicionar 1 ao array para evitar ver um warning.
    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

    # Máscara circular HPF, o círculo central é 0, permanecendo todos uns
    # Pode ser usado para detecção de borda porque as frequências baixas no centro são bloqueadas
    # e apenas altas frequências são permitidas. As bordas são componentes de alta frequência.

    # Amplifica o ruído.
    # FIGURA 6:
    # rows, cols = img.shape
    # crow, ccol = int(rows / 2), int(cols / 2)
    # mask = np.ones((rows, cols, 2), np.uint8)
    # r = 20
    # center = [crow, ccol]
    # x, y = np.ogrid[:rows, :cols]
    # mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
    # mask[mask_area] = 0

    # Máscara LPF circular, o círculo central é 1, permanecendo todos os zeros
    # Permite apenas componentes de baixa frequência - regiões suaves
    # Pode suavizar o ruído, mas desfoca as bordas.
    # FIGURA 7:
    rows, cols = img.shape
    crow, ccol = int(rows / 2), int(cols / 2)
    mask = np.zeros((rows, cols, 2), np.uint8)
    r = 40
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
    mask[mask_area] = 1

    # Filtro Passa-Banda - Máscara de círculo concêntrico, apenas os pontos que vivem em um círculo concêntrico são uns
    # rows, cols = img.shape
    # crow, ccol = int(rows / 2), int(cols / 2)
    # mask = np.zeros((rows, cols, 2), np.uint8)
    # r_out = 80
    # r_in = 10
    # center = [crow, ccol]
    # x, y = np.ogrid[:rows, :cols]
    # mask_area = np.logical_and(((x - center[0]) ** 2 + (y - center[1]) ** 2 >= r_in ** 2),
    #                         ((x - center[0]) ** 2 + (y - center[1]) ** 2 <= r_out ** 2))
    # mask[mask_area] = 1

    # aplicar máscara e DFT inverso: Multiplique a imagem transformada de Fourier (valores)
    # com os valores da máscara.
    fshift = dft_shift * mask

    # Obtenha o espectro de magnitude (apenas para fins de plotagem)
    fshift_mask_mag = 20 * np.log(cv2.magnitude(fshift[:, :, 0], fshift[:, :, 1]))

    # Inverter deslocamento para mudar a origem de volta para o canto superior esquerdo.
    f_ishift = np.fft.ifftshift(fshift)

    #Inverse DFT to convert back to image domain from the frequency domain. 
    #Will be complex numbers
    img_back = cv2.idft(f_ishift)

    #Magnitude spectrum of the image domain
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    fig = plt.figure(figsize=(12, 12))
    ax1 = fig.add_subplot(2,2,1)
    ax1.imshow(img, cmap='gray')
    ax1.title.set_text('Input Image')
    ax2 = fig.add_subplot(2,2,2)
    ax2.imshow(magnitude_spectrum, cmap='gray')
    ax2.title.set_text('FFT of image')
    ax3 = fig.add_subplot(2,2,3)
    ax3.imshow(fshift_mask_mag, cmap='gray')
    ax3.title.set_text('FFT + Mask')
    ax4 = fig.add_subplot(2,2,4)
    ax4.imshow(img_back, cmap='gray')
    ax4.title.set_text('After inverse FFT')
    plt.show()