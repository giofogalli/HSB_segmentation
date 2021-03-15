import sys, os, re
import time
from skimage import morphology, measure, restoration, segmentation, util, color, feature
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import signal, fftpack, interpolate, ndimage, stats
np.set_printoptions(suppress=True, linewidth=100)

def p(img):
	plt.imshow(img)
	plt.show()

def spread_hist(img): #stretch image histogram from 0 to 255
	img = np.int64(img)
	maxx = np.max(img)
	minn = np.min(img)
	out = ((img - minn)/(maxx - minn)) * 255
	return(np.uint8(out))
'''
def kullback_leibler_divergence(p, q):
    p = np.asarray(p)
    q = np.asarray(q)
    filt = np.logical_and(p != 0, q != 0)
    return np.sum(p[filt] * np.log2(p[filt] / q[filt]))
'''

# OPENCV CONVEX HULL IMAGE
def cv_convex_hull_image(binary):
    binary = np.uint8(binary)
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    hull_list = []
    for i in range(len(contours)):
        hull = cv2.convexHull(contours[i])
        hull_list.append(hull)
    #drawing = np.zeros((binary.shape[0], binary.shape[1], 3), dtype=np.uint8)
    drawing = np.zeros(binary.shape, dtype=np.uint8)
    for i in range(len(contours)):
        #color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
        #color1 = (0,255,255)
        #color2 = (255,255,0)
        #cv2.drawContours(drawing, contours, i, color1, thickness=5) # ONLY CONTOURS
        cv2.drawContours(drawing, hull_list, i, 255, thickness=-1) # REAL HULL
    return(drawing)

'''
# Vertical equalization (VLINES)
rows, cols = image.shape
equalized = np.zeros_like(image)
for i in range(5,cols-6):
	thin = np.median(cv2.equalizeHist(image[:,i-5:i+6]), axis=1)
	equalized[:,i] = thin

fig, ax = plt.subplots(1,2)
ax[0].imshow(image)
ax[1].imshow(equalized)
plt.show()

#image = equalized
'''
    
def auto_segment(img, kernel=(50,150), blur_ratio=200, plot=False):    
    '''
    Automated segmentation of Arrieta's filtered image.
    
    img : array_like
        Arrieta's filtered image (one channel).
    kernel : tuple
        Shape of kernel used for bluring. The first value is rows and the second columns.
        The kernel shape is related to segmented area shape. 
        Recommended kernel[0]/kernel[1] ratio: same as tooth shape [1/3].
        Default is (50, 150).
    blur_ratio : int
        Ratio used for resize intermediate images. Recommended values in the range [100,300].
        LOW blur_ratio -> more circular area, more padding from neat bands
        HIGH blur_ratio -> more adjusted to neat bands, lose recoverable areas
        Default is 200.
    plot: bool
        Plot segmented image and a image for segmentation visual analysis.

    Return: array_like
        Segmented image.
    '''

    r,c = img.shape[:2]
    kernel1, kernel2 = kernel

    reach_size = kernel1*kernel2*blur_ratio
    rc_ratio = r/c
    newc = np.sqrt(reach_size/rc_ratio)
    newr = rc_ratio * newc

    image = cv2.resize(img, (int(newc), int(newr)))
    
    print(f'''original: {img.shape}, {img.size}
                blur_ratio: {img.size/(kernel1*kernel2)}
    ''') 
    print(f'''reduced: {image.shape}, {image.size}
                blur_ratio: {image.size/(kernel1*kernel2)}
    ''')
 
    # Applying sobel operator
    vert = cv2.Sobel(image,-1,0,1) #vertical
    horiz = cv2.Sobel(image,-1,1,0)
    
    # Bluring sobel image
    v = vert.copy()
    kernel = (kernel1,kernel2)#(50,150)
    v1 = cv2.blur(v, kernel)
    v2 = cv2.blur(v1, kernel)
    v3 = cv2.blur(v2, kernel)
    #plt.imshow(v3)
    #plt.show()

    h = horiz.copy()
    #kernel = (50,150)
    h1 = cv2.blur(h, kernel)
    h2 = cv2.blur(h1, kernel)
    h3 = cv2.blur(h2, kernel)
    #plt.imshow(h3)
    #plt.show()
    
    subvh = np.clip(np.int32(v3)-np.int32(h3), 0,255)

    thresh_list_descent = np.arange(subvh.max()*2//3,subvh.max()//3-1,-1)
    ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(51,51)) # ellipse kernel for mask branches removal
    output_mask = np.zeros_like(img) # the original one
    decision = 'loop end'
    for i in thresh_list_descent:
        mask = np.uint8(subvh>=i)
        n, labels = cv2.connectedComponents(mask)
        hsb_lab = labels[subvh==subvh.max()][0] # pinpoint the right HSB label
        labels[labels!=hsb_lab] = 0 #only HSB label remains
        labels = cv2.morphologyEx(np.uint8(labels), cv2.MORPH_OPEN, kernel=ellipse) # remove branches of label mask
        #redo labelling before exclude possible cut off branches
        n, labels = cv2.connectedComponents(np.uint8(labels>0))
        hsb_lab = labels[subvh==subvh.max()][0] # pinpoint the right HSB label
        labels[labels!=hsb_lab] = 0 
        #
        hull = cv_convex_hull_image(labels)
        # Performing IoU 
        uni_hull, counts_hull = np.unique(hull, return_counts=True)
        uni_I, counts_I = np.unique(np.abs(hull//255 - (labels>0)*1), return_counts=True)
        if len(counts_hull) < 2:
            print('No background or foreground exist after convex hull processing.')
            os._exit(1)
        elif len(counts_I) < 2:
            print('CONVEX HULL is axactly the same as HSB mask.')
            counts_I = np.append(counts_I, 0)
        iou =  counts_I[1] / counts_hull[1] # Intersection over Union of hull and HSB mask
        # VISUALIZATION ONLY (make a darker area as background)
        '''
        im = vert.copy()
        im = np.int32(im)
        #im[subvh<i]-=80 # threshold mask
        im[hull==0] -=80 # hull mask
        im = np.clip(im, 0,255)
        '''
        # PLOT IMAGE IN LOOP
        #fig, ax = plt.subplots(1,2)
        #ax[0].imshow(labels+hull//255)
        #ax[0].set_title(f'>={i},{subvh.max()}->0, iou={iou}')        
        #ax[1].imshow(im)
        #plt.show()
        if iou > 0.1333: # means 20000 of intersection (without division by hull area)
            decision = 'IoU limit break'
            break
        else:
            output_mask = hull

    im = vert.copy()
    im = np.int32(im)
    #im[subvh<i]-=80 # threshold mask
    im[output_mask==0] -=80 # hull mask
    im = np.clip(im, 0,255)
    output_mask = cv2.resize(output_mask, (c,r))
    img[output_mask==0]=0
    print(f'threshold: {i}; decision: {decision}')
    
    if plot:
        fig, ax = plt.subplots(1,2)
        ax[0].imshow(img)
        ax[1].imshow(im)
        plt.show()
    return(img, im, i, decision)
    
if __name__ == '__main__':


    input_dir = '/home/sergio/py-venvs/matching_set/fotos_zuli/pipeline/internship/dentes_giovani/store_DE'
    output_dir = '/home/sergio/py-venvs/matching_set/fotos_zuli/pipeline/internship/dentes_giovani/store_analyse'
    #os.chdir('/home/sergio/py-venvs/matching_set/fotos_zuli/pipeline/internship/arrieta_imgs')
    
    os.chdir(input_dir)

    list_imgs = os.listdir()#['5sDE.jpg']#[i for i in os.listdir() if re.search('\An.*\.jpg$',i)] #['n149a.jpg']
    list_imgs = sorted(list_imgs, key=lambda x: int(re.search('\d+',x).group()))
    
    from filtering2 import horizontal_bands_filter

    for img_name in list_imgs:

        print(img_name)
        img = cv2.imread(img_name, 0)
        segmented, im, threshold, decision = auto_segment(img, kernel=(50,150), blur_ratio=200, plot=False)
        a = horizontal_bands_filter(segmented, name=img_name, save=False, pos_and_neg=False, metadata=None)
        
        os.chdir(output_dir)
        
        fig, ax = plt.subplots(1,3)
        ax[0].imshow(segmented)
        ax[0].set_title(f'threshold: {threshold} decision: {decision}')
        ax[1].imshow(im)
        ax[2].imshow(a)
        plt.tight_layout()
        #plt.savefig(f'{img_name[:-4]}A.png', dpi=300)
        plt.close(fig)

        os.chdir(input_dir)
        #p(a)







# DISCRETE FOURIER TRANSFORM 2D
'''
img = image
dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
dft_shift+=1
magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1]))

rows, cols = img.shape
crow, ccol = rows//2, cols//2
mask = np.ones((rows, cols, 2), np.uint8)
rout = 800
rin = 500
center = [crow, ccol]
x,y = np.ogrid[:rows, :cols]
mask_out = (x - center[0])**2 + (y-center[1])**2 <= rout**2
mask_in = (x - center[0])**2 + (y-center[1])**2 <= rin**2
mask[mask_out==False] = 0
mask[mask_in==True] = 0

fshift = dft_shift * mask
fshift_mask_mag = 2000 * np.log(cv2.magnitude(fshift[:,:,0]+1, fshift[:,:,1]+1))
f_ishift = np.fft.ifftshift(fshift)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:,:,0], img_back[:,:,1])

fig, ax = plt.subplots(2,2)
ax[0,0].imshow(img)
ax[0,1].imshow(magnitude_spectrum)
ax[1,0].imshow(fshift_mask_mag)
ax[1,1].imshow(img_back)
plt.show()
'''
# END OF DFT 2D
