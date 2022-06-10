


# Gradio with MONAI and UNET/UNETR for multi-organ segmentation 

*[Work in progress]*

This repo explores using [Gradio](https://gradio.app/) with medical imaging models to drag and drop medical imaging volumes and get an automatic multi-organ segmentation. 

- Run app
- Drag and drop a nifti image
- Get an automatic multi-organ segmentation
- 3D mesh rendering of the segmentated organs

![](images/screenshot1.png)
*Data from the [Medical Image Decathlon](http://medicaldecathlon.com/) (CC-BY-SA)*

## Run

- Create python environment
- Install requirements: `pip install -r requirements.txt`




## TODO
- [ ] Include some sort of progress bar in gradio
- [ ] Include UNETR
- [ ] Preview of nifti once loaded