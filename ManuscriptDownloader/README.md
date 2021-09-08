# ManuscriptDownloader
## Presentation

This repository contains a script that will download the manuscripts' folios for you, and extract the illustrations from them.

## Prerequisites

To run the script, some additional packages are needed:
```
pip install tqdm
pip install Pillow
pip install bs4
pip install requests
pip install fitz
pip install PyMuPDF
```

## Command Line
There are two commands: 

- **download** : Downloads the manuscripts' folios.
- **extract** : Extracts illustrations from manuscript folios. The instruction about how the illustrations have to be extracted are stored in .json files. The .json files folder is located in the root of the repository under the name "annotations".
In each manuscript folder, two directories will be created "illustration" and "annotation". The directory "illustration" contains the extracted illustration and the directory "annotation" contains the manuscripts folios with rectangles showing for each illustration, how it has been defined.

To download the folios use:
```
python main.py download -p path
```
**args**
- `-p, --path` directory where manuscripts will be downloaded.

To extract the illustrations from the folios use:
```
python main.py extract -p path -a annotation_path
```
**args**
- `-p, --path` directory where manuscripts are stored.
- `-a, --annotation_path` directory where the JSON annotation files are stored. If this parameter isn't set it will use the "annotations" folder, included in this directory. 


## Manuscripts

Some information about the manuscripts:

| Alias  | Name | Nb folios | Nb illustrations | Link |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| P1  | Barb.gr.438  | 109  | 51  | https://digi.vatlib.it/mss/detail/Barb.gr.438  |
| P2  | Ott.gr.354   | 176  | 51  | https://digi.vatlib.it/mss/detail/Ott.gr.354  |
| P3  | Gr. IV. 35  | 188  | 52  | http://www.internetculturale.it/jmms/iccuviewer/iccu.jsp?id=oai%3A193.206.197.121%3A18%3AVE0049%3ACSTOR.241.10734&mode=all&teca=marciana  |
| D1  | De Materia Medica  | 557  | 816  | https://www.themorgan.org/manuscript/143825  |
| D2  | Of Medical Substances | 351  | 405  | https://www.wdl.org/en/item/10690/  |
| D3  | Chig.F.VII.159  | 512  | 839  | https://digi.vatlib.it/view/MSS_Chig.F.VII.159  |
