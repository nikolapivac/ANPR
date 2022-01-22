# ANPR

This Automatic Number Plate Recognition application was made as a part of a bachelor thesis in July 2021. It was made following a tutorial from Adrian Rosebrock (`pyimagesearch.com`) 
and modified to recognize Croatian license plates.

The thesis deals with the problem of recognizing license plates on cars. The methods of image processing using the OpenCV library and the methods of optical character recognition (OCR) 
using the Tesseract system are explained in detail. 
The algorithm was run on 30 photographs classified into three groups of ten photographs, sorted by conditions in which they were taken. Group 1 contains photographs taken in best condition,
Group 2 contains medium quality photos, and Group 3 contains photographs taken in the worst conditions.

To run the application, type the following command in Command Prompt:

```bash
>> python   ocr_license_plate.py   --i  image_folder  --c 1
```

