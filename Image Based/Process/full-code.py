from ultralytics import YOLO
model = YOLO(r"D:\OneDrive\Bangkit Academy\Project\Image Based\1\runs\segment\train3\weights\best.pt")

image_path = "test1.jpg" 
image_cloth_path = "cloth3.jpg"

results = model.predict(image_path)
boxes = results[0].boxes.xyxy.tolist()
result = results[0]
bounding_height = int(boxes[0][3] - boxes[0][1])
bounding_width = int(boxes[0][2] - boxes[0][0])
from PIL import Image
masks = result.masks
mask1 = masks[0]
polygon = mask1.xy[0]
mask = mask1.data[0].numpy()
mask_img = Image.fromarray(mask,"I")
mask_img = mask_img.convert('RGB')

image = Image.open(image_path)

# Mendapatkan ukuran citra
width, height = image.size
resized_mask_img = mask_img.resize((width, height), Image.ANTIALIAS)
width, height = resized_mask_img.size
resized_mask_img_pixels = resized_mask_img.load()
white_area = (int(boxes[0][0]), int(boxes[0][1]), int(boxes[0][2]), int(boxes[0][3]))
scale_factor = 1.05

# Calculate the expansion size
expand_width = int(((white_area[2] - white_area[0]) * (scale_factor - 1)) / 2)
expand_height = int(((white_area[3] - white_area[1]) * (scale_factor - 1)) / 2)

# Adjust the coordinates
left = (white_area[0] - expand_width)
top = (white_area[1] - expand_height)
right =(white_area[2] + expand_width)
bottom = (white_area[3] + expand_height)

# Crop the adjusted white area
white_part = resized_mask_img.crop(box=(white_area[0], white_area[1], white_area[2], white_area[3]))
white_part_resized = white_part.resize((right-left,bottom-top), Image.ANTIALIAS)

# Create a new blank image that will show changes
new_img = Image.new('RGB', resized_mask_img.size, (0, 0, 0))
new_img.paste(white_part_resized, (left, top))
pixels = new_img.load()

threshold_value = 128
for y in range(height):
    for x in range(width):
        # Ambil nilai piksel (dalam kasus RGB, ambil salah satu komponen saja karena awalnya adalah grayscale)
        r, g, b = pixels[x, y]  # Semua nilai r, g, b sama karena asalnya grayscale
        if r < threshold_value:
            pixels[x, y] = (0, 0, 0)  # Set pixel to black
        else:
            pixels[x, y] = (255, 255, 255)

image2 = Image.open(image_cloth_path)
adjust_cloth = 150
cloth_resized= image2.resize((bounding_width + adjust_cloth, bounding_height + adjust_cloth), Image.ANTIALIAS)
mask_pixels = new_img.load()
width_mask, height_mask = new_img.size
cloth_resized_pixels = cloth_resized.load()
width_cloth,height_cloth = cloth_resized.size
start_x = int(boxes[0][0]) - int(adjust_cloth / 2)
end_x = int(boxes[0][2]) + int(adjust_cloth / 2)
start_y = int(boxes[0][1]) - int(adjust_cloth / 2)
end_y = int(boxes[0][3]) + int(adjust_cloth / 2)
for y in range(start_y, end_y-1):
    for x in range(start_x, end_x-1):
        current_pixel = mask_pixels[x, y]
        if current_pixel == (0, 0, 0):
            mask_pixels[x, y] = mask_pixels[x,y]  
        else:
            mask_pixels[x, y] = cloth_resized_pixels[x-start_x,y-start_y]
            
final_masking = new_img.copy()
final_masking_pixels = final_masking.load()

final_image_ori = Image.open(image_path)
final_image_ori_pixels = final_image_ori.load()

for y in range(height):
    for x in range(width):
        current_pixel_2 = final_masking_pixels[x,y]
        if current_pixel_2 == (0,0,0):
            final_masking_pixels[x,y] = final_image_ori_pixels[x,y]

final_masking.show()