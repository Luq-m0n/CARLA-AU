from PIL import Image, ImageDraw
import xml.etree.ElementTree as ET

sample_image = Image.open('/home/aashiqueluqman/vista/Carla-Object-Detection-Dataset-master/train/Town02_002160.png')


tree = ET.parse('/home/aashiqueluqman/vista/Carla-Object-Detection-Dataset-master/train/Town02_002160.xml')
root = tree.getroot()

bbox_coordinates = []
for member in root.findall('object'):
    class_name = member[0].text # class name
        
    # bbox coordinates
    xmin = int(member[4][0].text)
    ymin = int(member[4][1].text)
    xmax = int(member[4][2].text)
    ymax = int(member[4][3].text)
    # store data in list
    bbox_coordinates.append([class_name, xmin, ymin, xmax, ymax])


sample_image_annotated = sample_image.copy()

img_bbox = ImageDraw.Draw(sample_image_annotated)

for bbox in bbox_coordinates:
    print(bbox)
    img_bbox.rectangle(bbox[1:], outline="red")
    img_bbox.text((bbox[3],bbox[4]),str(bbox[0]),align='right')
   
sample_image_annotated.show()
sample_image_annotated.save('Town02_002160an.png','PNG')
