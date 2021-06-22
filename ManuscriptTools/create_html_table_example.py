from utils.DirTools import DirTools
from utils.TableHTML import TableHTML
import pathlib
import numpy as np
import os

if __name__ == "__main__":
    # we set the manuscript path we want to work with
    manuscript1_path = "manuscripts_demo/P2/illustration"
    manuscript2_path = "manuscripts_demo/P3/illustration"

    pathlib.Path("example").mkdir(parents=False,
                                  exist_ok=True)  # we create an example folder where the web page will be stored

    # we create a list of the path to the images in the two previous manuscripts
    manuscript1_images_path = DirTools.list_folder_images(manuscript1_path)
    manuscript2_images_path = DirTools.list_folder_images(manuscript2_path)

    table = TableHTML('Title of the HTML Table!')  # we instantiate an object table HTML with the title of the page

    table.add_head(["column1 title", "2nd column", "3rd column"])  # we set the column names for the table

    # we will here create a table with 4 rows

    for i in range(4):
        # we select here 3 random image index in the P2 and P3 manuscripts
        p2_idx = np.random.randint(len(manuscript1_images_path))
        p3_idx = np.random.randint(len(manuscript2_images_path))
        p2_idx_2 = np.random.randint(len(manuscript1_images_path))

        # an image is a tuple (i, path)
        # where i is the index (can be anything) of the image,
        # and path, the path to the image
        image1 = (p2_idx, manuscript1_images_path[p2_idx])
        image2 = ("put a title here!", manuscript1_images_path[p2_idx_2])
        image3 = ("", manuscript2_images_path[p3_idx])

        # a row is a list in which each element is a column
        # a column can be whether an image, whether a list of images
        row = [[image1, image2], image3, image2]
        table.add_row(row)  # we add the row to the table

    html_table_path = os.path.join("example",
                                   "table.html")  # we set the path of the table as being ./example/table.html
    table.save(html_table_path)  # we save the table at that path
    exit()
