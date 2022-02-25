from globalSettings import *

def resize_image(image, pad = 0):
    im = np.array(image)
    im = im.copy()
    
    min_dim = (BaseModelConfig.patch_dim * 3) + (BaseModelConfig.gap * 2) + 2 + (BaseModelConfig.jitter * 2) + pad * 2
    
    if im.shape[0] < min_dim or im.shape[1] < min_dim:
        # resize image
        ratio1 = float(min_dim)/float(im.shape[0])
        ratio2 = float(min_dim)/float(im.shape[1])
        
        ratio = max(ratio1, ratio2)
        im = cv2.resize(im,(int(im.shape[0]*ratio), int(im.shape[1]*ratio)))
    
    immax=float(im.shape[0]*im.shape[1])
    
    if immax > 450000:
        return im
    
    maxdim=random.randint(immax,450000 + 1)
    ratio=float(maxdim)/float(immax)
    im = cv2.resize(im,(int(im.shape[0]*np.sqrt(ratio)), int(im.shape[1]*np.sqrt(ratio))))
    return im

def plot_image(img,text=None,should_save=False):
    plt.figure(figsize=(10, 10))
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()  

def show_plot(iteration,loss,fname):
    plt.plot(iteration,loss)
    plt.savefig(fname)
    plt.show()
    
def display_canvas(patch_bucket, coordinates, title):
    x = np.linspace(8, 8, 100)
    y = np.cos(x)

    fig, ax = plt.subplots()

    idx = 0
    for i in reversed(range(3)):
        for j in range(0, 9, 3):
            ax.imshow(patch_bucket[idx], extent=[coordinates[i+j][0], 
                                          coordinates[i+j][0]+BaseModelConfig.patch_dim, 
                                          coordinates[i+j][1], 
                                          coordinates[i+j][1]+BaseModelConfig.patch_dim], aspect='auto')
            idx += 1
    ax.plot(x, y)
    plt.title(title)
    plt.show()

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor
    