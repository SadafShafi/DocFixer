from PIL import Image
import tensorflow as tf
import cv2
import numpy as np
from skimage import io


# Python 3.8.10


class DocOrientationFixer():

  def __init__(self):

    '''
    This code is responsible for loading a pre-trained Keras model and its weights.

    model_path: String, path to the saved model weights.
    model_architecture: String, path to the saved model architecture.

    The function uses model_from_json from the tf.keras.models module to instantiate a model from the saved architecture.
    It then uses load_weights method to load the saved weights.
    Finally, the model is compiled with optimizer 'adam', loss function 'binary_crossentropy', and evaluation metric 'accuracy'.
    '''

    self.model_path = 'Model_100L_1.0_acc/best_weights_with_validation_big_dataset100_acc_100.h5' 
    self.model_architecture = 'Model_100L_1.0_acc/model_100_lay_100_acc.json' 
    self.model = tf.keras.models.model_from_json(open(self.model_architecture).read())
    self.model.load_weights(self.model_path)
    self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])




  def orientation(self,image):

      """
        Predict the orientation of an image using a pre-trained model.
        
        Parameters:
        image (np.array): Input image in numpy array format.
        
        Returns:
        pred (np.array): Prediction score with shape (1,2) for each orientation class, where
            pred[0,0] is the score for 'Not Upside Down' class, and
            pred[0,1] is the score for 'Upside Down' class.
        
        """

      image = Image.fromarray(image)
      image = image.resize((224,224))
      image = np.expand_dims(image, axis=0)
      pred = self.model.predict(image/255.0,verbose = False)
      return pred

  def addPadding(self,img):
    """
      Adds padding to an image to make it square.
      
      Parameters
      ----------
      img : numpy array
          The input image as a numpy array.
          
      Returns
      -------
      result : numpy array
          The padded image as a numpy array.
          
      Raises
      ------
      None
      
      Examples
      --------
      >>> addPadding(np.array([[1, 2], [3, 4]]))
      array([[255, 255, 255, 255, 255, 255],
            [255, 255, 255, 255, 255, 255],
            [255, 255,   1,   2, 255, 255],
            [255, 255,   3,   4, 255, 255],
            [255, 255, 255, 255, 255, 255],
            [255, 255, 255, 255, 255, 255]], dtype=uint8)
      """

    if img.shape[2] > 3:
        img = img[...,:3]
    # print(img.shape)
    old_image_height, old_image_width, channels = img.shape
    # print(img.shape)
    # create new image of desired size and color (blue) for padding
    toAdd = abs(old_image_height - old_image_width)

    if old_image_width >= old_image_height:
      new_image_width = old_image_width  
      new_image_height = old_image_height + toAdd
    else:
      new_image_width = old_image_width  + toAdd
      new_image_height = old_image_height 

    if new_image_height < 1000 :new_image_height = 1000
    if new_image_width < 1200:new_image_width = 1200

    color = (255) * channels
    result = np.full((new_image_height,new_image_width, channels), color, dtype=np.uint8)

    # compute center offset
    x_center = (new_image_width - old_image_width) // 2
    y_center = (new_image_height - old_image_height) // 2

    # copy img image into center of result image
    result[y_center:y_center+old_image_height, 
          x_center:x_center+old_image_width] = img
    # print(result.shape)
    return result


  def deskew(self,im, max_skew=10):

      """
        Deskews an input image and returns the rotated image.

        Parameters:
        im (ndarray): Input image in numpy array format
        max_skew (int, optional): Maximum skew angle for filtering outliers. Default is 10.

        Returns:
        ndarray: Deskewed image in numpy array format

        """


      im = self.addPadding(im)

      height, width , _= im.shape
      # Create a grayscale image and denoise it
      im_gs = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
      
      im_gs = cv2.fastNlMeansDenoising(im_gs, h=3)
      # print(im_gs.shape)
      
      
      # Create an inverted B&W copy using Otsu (automatic) thresholding
      im_bw = cv2.threshold(im_gs, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
      # cv2_imshow(im_bw)
      # Detect lines in this image. Parameters here mostly arrived at by trial and error.
      lines = cv2.HoughLinesP(
          im_bw, 1, np.pi / 180, 200, minLineLength=width / 12, maxLineGap=width / 150
      )
      # print(lines)
      # Collect the angles of these lines (in radians)
      angles = [] 
      for line in lines:
          x1, y1, x2, y2 = line[0]
          angles.append(np.arctan2(y2 - y1, x2 - x1))
        #  print("X1,y1", x1,y1,x2,y2)
      # If the majority of our lines are vertical, this is probably a landscape image
      landscape = np.sum([abs(angle) > np.pi / 4 for angle in angles]) > len(angles) / 2
      # Filter the angles to remove outliers based on max_skew
      if landscape:
          angles = [
              angle for angle in angles if np.deg2rad(90 - max_skew) < abs(angle) < np.deg2rad(90 + max_skew)
          ]
      else:
          angles = [angle for angle in angles if abs(angle) < np.deg2rad(max_skew)]
      if len(angles) < 5:
          # Insufficient data to deskew
          return im
      # Average the angles to a degree offset
      angle_deg = np.rad2deg(np.median(angles))
      # If this is landscape image, rotate the entire canvas appropriately
      if landscape:
          # print(angle_deg)
          if angle_deg < 0:
              im = cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)
              angle_deg += 90
          elif angle_deg > 0 and angle_deg<60:
              im = cv2.rotate(im, cv2.ROTATE_90_COUNTERCLOCKWISE)
              angle_deg -= 90
          elif angle_deg>60 and angle_deg<88.5:
            im = cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)
            angle_deg -= 90
          elif angle_deg>88.5 and angle_deg<100:
            im = cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)
            angle_deg += 90  
      # Rotate the image by the residual offset
      M = cv2.getRotationMatrix2D((width / 2, height / 2), angle_deg, 1)
      im = cv2.warpAffine(im, M, (width, height), borderMode=cv2.BORDER_REPLICATE)

      isUpsideDown = self.orientation(im)[0][0]


      if(isUpsideDown>=0.5):
        print("Don't rotate the image")
      else:
        print("Rotate the image")
        im = cv2.rotate(im,cv2.ROTATE_180)

      print(isUpsideDown)
      return im





  def Fix_Doc_Orientation(self,path):
    """
      Fixes the orientation of a Doc image.
      
      Parameters
      ----------
      path : str
          The file path of the Doc image.
          
      Returns
      -------
      img : numpy array
          The fixed orientation of the Doc image as a numpy array.
          
      Raises
      ------
      FileNotFoundError
          If the specified `path` does not exist.
          
      Examples
      --------
      >>> Fix_Doc_Orientation('/path/to/Doc.jpg')
      array([...], dtype=uint8)
      """

    image = Image.open(path)
    img = self.deskew(np.asarray(image))

    return img




############ Run the following function to get an image as numpy array
myinstance = DocOrientationFixer()

image_path = 'sample.jpg'
print(myinstance.Fix_Doc_Orientation(image_path))