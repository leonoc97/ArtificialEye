# bounding_box_position.py

def get_position(x1, y1, x2, y2, frame_width, frame_height):
    frame_center_x = frame_width / 2
    frame_center_y = frame_height / 2

    box_center_x = (x1 + x2) / 2
    box_center_y = (y1 + y2) / 2

    if box_center_x < frame_center_x:
        horizontal_position = "links"
    elif box_center_x > frame_center_x:
        horizontal_position = "rechts"
    else:
        horizontal_position = "Mitte"

    if box_center_y < frame_center_y:
        vertical_position = "oben"
    elif box_center_y > frame_center_y:
        vertical_position = "unten"
    else:
        vertical_position = "Mitte"

    return horizontal_position, vertical_position
