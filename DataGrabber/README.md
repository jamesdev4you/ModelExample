<h1>ANIMAL DATA GRABBER</h1>  
<h3>Great for gathering pictures of animals, and using these pictures as inputs for machine learning!</h3>

<h4>
This was built mostly from Claude. This is version 1.0 and I have not thoroughly went through this code. Please use with discresion.
The goal is to build an app that uses neural networks to analyze the crediablilty of images. ( Classify images. i.e: Is this, or is this not a White Ibis? ).
</h4>

To build a n.n. we must first gather quality data. This is said "quality data" for animals. It works fantastic on windows 11 and Ubuntu.  MacOS will eventually be done.

This is an GBIF API Data Grabber that uses https://www.inaturalist.org recommended way to gather data.

It is also a data splitter that takes data from iNaturalist and organizes them into credible pictures that can be used for machine learning.

From a big picture perspective, it uses the GBIF API to gather openly-licensed occurrence images, most of which are hosted on iNaturalist.

Split dataset automatically splits the data in a way that may be used for machine learning purposes. 

It's a convienent way to get data on iNaturalist, without manually downloading a bunch of photos and giving credit where its due. Credit is DUE btw - in data/licenses.csv

To run it:
1) get your terminal to the correct directory
2) pip install requests 
3) python download_gbif_images.py 
4) python split_dataset.py     
5) ba da bing ba da boom. For more info please read comments in the two files or upload directory to claude.
