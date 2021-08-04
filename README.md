# Image Collation

(ICDAR 2021) Pytorch implementation of Paper "Image Collation: Matching illustrations in manuscripts"

[[PDF](http://imagine.enpc.fr/~shenx/ImageCollation/ICDAR2021_ImageCollation_Paper.pdf) [[Project website](http://imagine.enpc.fr/~shenx/ImageCollation/)] [[1min Video] (https://youtu.be/yXe7JHSJDUs)] [[Slides](http://imagine.enpc.fr/~shenx/ImageCollation/ICDAR_ImageCollation_Slides.pdf)]  

<p align="center">
<img src="https://github.com/Rykoua/ImageCollation/blob/main/img/teaser.JPG" width="800px" alt="teaser">
</p>

The project is an extension work to [ArtMiner](http://imagine.enpc.fr/~shenx/ArtMiner/) and [Historical Watermark Recognition](http://imagine.enpc.fr/~shenx/Watermark). If our project is helpful for your research, please consider citing : 

```
@inproceedings{kaoua2021imagecollation,
  title={Image Collation: Matching illustrations in manuscripts},
  author={Kaoua, Ryad and Shen, Xi and Durr-Lazaris, Alexandra and Lazaris, Stavros and Picard, David and Aubry, Mathieu},
  booktitle={International Conference on Document Analysis and Recognition (ICDAR)},
  year={2021}
}
```

## Table of Content
* [Installation](#installation)
* [Download Manuscript](#download-manuscript)
* [Match Illustration](#match-illustration)
* [Visualize Matches](#visualize-matches)



## Installation


Code is tested under **Pytorch > 1.0 + Python 3.6** environment. To install all dependencies : 

``` Bash
bash requirement.sh
```

## Download Manuscript

[ManuscriptDownloader](https://github.com/Rykoua/ManuscriptDownloader) contains a script that will download the manuscripts' folios for you, and extract the illustrations from them. 

## Match Illustration
[IllustrationMatcher](https://github.com/Rykoua/IllustrationMatcher) will run our algorithm to match two manuscripts. The results can be visualised with [ManuscriptTools](https://github.com/Rykoua/ManuscriptTools)

## Visualize Matches

Given two manuscripts, and their respective illustrations extracted, [ManuscriptTools](https://github.com/Rykoua/ManuscriptTools) will generate a set of web pages. 
