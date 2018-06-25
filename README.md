# GrabCut

My implementation of *[Grabcut: Interactive foreground extraction using iterated graph cuts, ACM SIGGRAPH 2004](https://cvg.ethz.ch/teaching/cvl/2012/grabcut-siggraph04.pdf)*, without border matting.

## Dependencies

* Python 3.6
* OpenCV
* NumPy
* scikit-learn
* python-igraph

## File desctiptions

* `GMM.py` - Gaussian mixture model
* `grabcut.py` - Core implementation of the algorithm.
* `grabcut_opencv.py` - A copy of [the official OpenCV sample](https://github.com/opencv/opencv/blob/master/samples/python/grabcut.py).
* `messi5.jpg` - A copy of [OpenCV's sample image](https://github.com/opencv/opencv/blob/master/samples/data/messi5.jpg).

## Usage

```
python grabcut.py <filename>
```

If `filename` is not specified, the program will pick `messi5.jpg` to segment. For more details, please refer to instructions printed to the console after run the program.