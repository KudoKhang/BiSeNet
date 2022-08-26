```bash
sh download_data.py
```


# Dataset structure
```bash
dataset---train---images---0.jpg
       |        |      |___1.jpg
       |        |
       |        |__masks---0.png
       |        |      |___1.png
       |        |
       |___test___images__...
       |        |
       |        |__masks__...
       |
       |___val___images___...
                |
                |_masks___...

train/test/val = 90/5/5        
```
# Augument datasets
- [x] Helen
- [x] Celeb-HQ
- [x] LaPa
- [x] IBug
----
Merge 4 dataset augmented → CeHePaIb 
