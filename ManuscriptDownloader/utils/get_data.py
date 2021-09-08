from .ManuscriptDownloader import ManuscriptDownloader
from .morgan_folio_downloader import download_themorgan_manuscript, download_file_url
from .extract_images_from_pdf import pdf_2_images
from .ExtractIllustrations import ExtractIllustrations
from os.path import join
from os import remove
from PIL import Image


def download_p1_manuscript(manuscripts_path):
    manuscript_downloader = ManuscriptDownloader()
    dest_path = join(manuscripts_path, "P1")
    manuscript_downloader.download_manuscript("MSS_Barb.gr.438", "vatlib", width=-1,
                                              dest_path=dest_path,
                                              rgb=False, prefix="image--", pixels_to_crop=107)
    image = Image.new('RGB', (1515, 2019))
    image.save(join(dest_path, "image--000.jpg"))


def download_p2_manuscript(manuscripts_path):
    manuscript_downloader = ManuscriptDownloader()
    dest_path = join(manuscripts_path, "P2")
    manuscript_downloader.download_manuscript("MSS_Ott.gr.354", "vatlib",  dest_path=dest_path,
                                              pixels_to_crop=0, with_real_image_name=True, divide_width_by=2)


def download_p3_manuscript(manuscripts_path):
    #downloading pdf
    dest_path = join(manuscripts_path, "P3")
    P3_PDF_url = "http://www.internetculturale.it/jmms/objdownload?id=oai%3A193.206.197.121%3A18%3AVE0049%3ACSTOR.241.10734&teca=marciana&resource=img&mode=all"
    pdf_2_images(P3_PDF_url, dest_path=dest_path)


def download_d1_manuscript(manuscripts_path):
    dest_path = join(manuscripts_path, "D1")
    D1_url = "http://ica.themorgan.org/manuscript/thumbs/143825"
    download_themorgan_manuscript(D1_url, dest_path=dest_path)
    download_file_url("http://ica.themorgan.org/icaimages/6/m652.219a.jpg", dest_path)
    download_file_url("http://ica.themorgan.org/icaimages/6/m652.381ra.jpg", dest_path)
    remove(join(dest_path, 'm652.219r.jpg'))


def download_d2_manuscript(manuscripts_path):
    manuscript_downloader = ManuscriptDownloader()
    dest_path = join(manuscripts_path, "D2")
    manuscript_downloader.download_manuscript(10690, "wdl", dest_path=dest_path)


def download_d3_manuscript(manuscripts_path):
    manuscript_downloader = ManuscriptDownloader()
    dest_path = join(manuscripts_path, "D3")
    manuscript_downloader.download_manuscript("MSS_Chig.F.VII.159", "vatlib", dest_path=dest_path,
                                              pixels_to_crop=54)


def download_data(manuscripts_path):
    download_func_dict = {"P1": download_p1_manuscript,
                          "P2": download_p2_manuscript,
                          "P3": download_p3_manuscript,
                          "D1": download_d1_manuscript,
                          "D2": download_d2_manuscript,
                          "D3": download_d3_manuscript}

    for manuscript_name, download_manuscript in download_func_dict.items():
        print("Downloading {} ...".format(manuscript_name))
        download_manuscript(manuscripts_path)


def extract_illustrations(manuscripts_path, annotations_path):
    manuscripts_list = ["P1", "P2", "P3", "D1", "D2", "D3"]
    extract_illustrations = ExtractIllustrations()

    for manuscript in manuscripts_list:
        print("Extracting illustrations for manuscript {}".format(manuscript))
        json_file = join(annotations_path, manuscript + ".json")
        manuscript_path = join(manuscripts_path, manuscript)
        extract_illustrations.save_annotations(json_file, dir_path=manuscript_path)


def set_up_data(manuscripts_path, annotations_path):
    print("="*10 + " Downloading folios " + "="*10)
    download_data(manuscripts_path)
    print("="*10 + " Extracting illustrations " + "="*10)
    extract_illustrations(manuscripts_path, annotations_path)


if __name__ == "__main__":
    MANUSCRIPTS_PATH = "manuscripts"
    ANNOTATIONS_PATH = "../annotations"
    set_up_data(MANUSCRIPTS_PATH, ANNOTATIONS_PATH)


