import numpy
import cv2

from PIL import Image
from handlers import ImageHandler

image_handler = ImageHandler()

# TODO:
#   - Improve responsiveness of buttons
#   - Make tracking algorithm better (it could always be better)

# Resolution of final height
WIDTH, HEIGHT = 1100, 960

# Single mode only adds a suit to a single face but is more stable
single_mode = False

# References the filenames of the suit images to overlay
suits = image_handler.load_images()
suits_alpha = image_handler.generate_alpha_masks(suits)
suit_sizes = [2.8, 3.5, 4.5, 3.1, 3.2, 2.9, 4.5]

# Acts as an offset based on a given multiplier of the height of the detecyed face
vertical_offsets = [-0.3, -0.1, -0.1, 0.2, 0.1, 0.05, 0.1]

# Ensure all lists are of equal size
assert len(suits) == len(suits_alpha) == len(suit_sizes) == len(vertical_offsets)

suit_index = 0
suit_img = suits[suit_index]
suit_alpha_mask = suits_alpha[suit_index]

# Create cv2 objects
window_name = "E-Drip demo"

cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, WIDTH, HEIGHT)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
video_stream = cv2.VideoCapture(0)

# Main loop
while(True):

    # Receive frame from video stream
    _, frame = video_stream.read()

    # Create greyscale version of frame
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Use face cascade to detect faces within the frame
    faces = face_cascade.detectMultiScale(grey, 1.1, 4)

    # Convert image to PIL format
    frame_PIL = Image.fromarray(frame)

    # If there are any faces
    if len(faces) > 0:

        # Only draw a single suit
        if single_mode:
            largest_face_index = numpy.argmax([i[2] for i in faces])  # Get largest face by bounding box width
            x, y, w, h = faces[largest_face_index]  # Extract values from face

            # Scale face (and alpha mask)
            scaled_width  = int(w*suit_sizes[suit_index])
            scaled_height = int(scaled_width * suit_img.height / suit_img.width)
            suit_alpha_mask_resized = suit_alpha_mask.resize((scaled_width,scaled_height), Image.ANTIALIAS)
            suit_img_resized = suit_img.resize((scaled_width,scaled_height), Image.ANTIALIAS)

            # Convert BGRA -> RGBA
            b, g, r, a = suit_img_resized.split()
            suit_img_resized = Image.merge("RGBA", (r, g, b, a))

            # Place image on top of the frame in correct location
            frame_PIL.paste(
                suit_img_resized,
                (
                    int(x + (0.5*w) - (0.5*scaled_width) ),
                    int(y + h * (1-vertical_offsets[suit_index]) )
                ),
                mask=suit_alpha_mask_resized
            )
        
        else:

            # Sort faces by largest size first.
            # This means the smaller faces (which are likely not faces and false positives)
            # are drawn first so they will likely be covered up by any larger faces
            # which are drawn last.
            faces = sorted(faces,key=lambda l:l[2])

            for face in faces:
                x, y, w, h = list(face)  # Extract values from face

                # Scale face (and alpha mask)
                scaled_width  = int(w*suit_sizes[suit_index])
                scaled_height = int(scaled_width * suit_img.height / suit_img.width)
                suit_alpha_mask_resized = suit_alpha_mask.resize((scaled_width,scaled_height), Image.ANTIALIAS)
                suit_img_resized = suit_img.resize((scaled_width,scaled_height), Image.ANTIALIAS)
                
                # Convert BGRA -> RGBA
                b, g, r, a = suit_img_resized.split()
                suit_img_resized = Image.merge("RGBA", (r, g, b, a))

                # Place image on top of the frame in correct location
                frame_PIL.paste(
                    suit_img_resized,
                    (
                        int(x + (0.5*w) - (0.5*scaled_width) ),
                        int(y + h * (1-vertical_offsets[suit_index]) )
                    ),
                    mask=suit_alpha_mask_resized
                )

    # Convert image back to cv2 format           
    frame = numpy.array(frame_PIL)

    # Show image on window
    cv2.imshow(window_name, frame)
    
    # If x is pressed, rotate suit_index and change suit
    if cv2.waitKey(1) & 0xFF == ord('x'):
        if suit_index == len(suits)-1:
            suit_index = 0
        else:
            suit_index += 1

        suit_img = suits[suit_index]
        suit_alpha_mask = suits_alpha[suit_index]
    
    # If q is pressed, beak of out loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video stream
video_stream.release()
# Destroy all the windows
cv2.destroyAllWindows()