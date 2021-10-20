import csv 

def save_data(file, header, data):
    """
    Save measured data in a CSV file
    """
    with open(file, 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in data:
            writer.writerow(row)


def get_projected_corners(dst):
    """
    Extract the coordinates of the four reference surface corners' and return
    them as a list. The mapping between list elements and corner coordinates is
    depicted in the diagram below.

     --------------
    |(0,1) -- (2,3)|
    |         /    |
    |        /     |
    |       /      |
    |      /       |
    |     /        |
    |    /         |
    |(4,5) -- (6,7)|
     --------------

    """
    # tl, bl, br, tr
    tl_x = dst[0][0][0]
    tl_y = dst[0][0][1]
    bl_x = dst[1][0][0]
    bl_y = dst[1][0][1]
    br_x = dst[2][0][0]
    br_y = dst[2][0][1]
    tr_x = dst[3][0][0]
    tr_y = dst[3][0][1]
    return [tl_x, tl_y, tr_x, tr_y, bl_x, bl_y, br_x, br_y]