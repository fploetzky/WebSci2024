# WebSci'24 -- Event Semantics for Knowledge Graphs

This repository contains the code for our paper **Lost in Recursion: Mining Rich Event Semantics in Knowledge Graphs** published at the Web Science Conference 2024.

For citations you may use:

```bibtex
@inproceedings{ploetzky2024receventsemantics,
  author       = {Florian Pl{\"{o}}tzky and
                  Niklas Kiehne and
                  Wolf{-}Tilo Balke},
  title        = {Lost in Recursion: Mining Rich Event Semantics in Knowledge Graphs},
  booktitle    = {ACM Web Science Conference (Websci '24), May 21--24, 2024, Stuttgart, Germany},
  publisher    = {{ACM}},
  year         = {2024},
  doi          = {10.1145/3614419.3644001},
}
```

## Working with the code

Please be aware that due to copyright reasons we can **not** publish the full texts of the newspaper articles used in the proof of concept. 
That is, you need to download the articles and parse them on your own.
To do that please successively load the ```json``` files in the subfolders  ```newspaper_articles```, download the newspaper article behind the respective in the keys, use [newspaper3k](https://newspaper.readthedocs.io/en/latest/) to extract the text from the article and populate the text attribute for each entry in the json files.

To mine the narratives and resolve the recursive nodes refer to the ```run_extraction.ipynb``` notebook.