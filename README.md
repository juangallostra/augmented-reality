# augmented-reality
Augmented reality card based application with Python, numpy and OpenCV

## Usage

* Place the image of the surface to be tracked inside the `reference` folder.
* On line 36 of `src/ar_main.py` replace `'model.jpg'` with the name of the image you just copied inside the `reference` folder.
* On line 40 of `src/ar_main.py` replace `'fox.obj'` with the name of the model you want to render.
* Open a terminal session inside the project folder and run `python src/ar_main.py`

### Command line arguments

* `--rectangle`, `-r`: Draws the projection of the reference surface on the video frame as a blue rectangle.
* `--matches`, `-m`: Draws matches between reference surface and video frame.


## Troubleshooting

**If you get the message**:

```
Unable to capture video
```
printed to your terminal, the most likely cause is that your OpenCV installation has been compiled without FFMPEG support. Pre-built OpenCV packages such as the ones downloaded via pip are not compiled with FFMPEG support, which means that you have to build it manually.

**If you get the error**:

```
Traceback (most recent call last):
File "src/ar_main.py", line 174, in
main()
File "src/ar_main.py", line 40, in main
obj = OBJ(os.path.join(dir_name, 'models/fox.obj'), swapyz=True)
File "[...]/augmented-reality/src/objloader_simple.py", line 16, in init
v = v[0], v[2], v[1]
TypeError: 'map' object is not subscriptable
```
The most likely cause is that you are trying to execute the code with Python 3 and the code is written in Python 2. The `map` function in Python 3 returns an iterable object of type map, and not a subscriptible list. To fix it, change the calls to `map()` by `list(map())` on lines 14, 19 and 24 of `src/objloader_simple.py`. 

## Explanation

See this blog entries for an in-depth explanation of the logic behind the code:

* [Part 1](https://bitesofcode.wordpress.com/2017/09/12/augmented-reality-with-python-and-opencv-part-1/)
* [Part 2](https://bitesofcode.wordpress.com/2018/09/16/augmented-reality-with-python-and-opencv-part-2/)

## Results

* [Mix](https://www.youtube.com/watch?v=YVJSFcUbIoU)
* [Fox](https://www.youtube.com/watch?v=V13VE6UJ-1g)
* [Ship](https://www.youtube.com/watch?v=VDwfW75f3Xo)
* [Rat](https://www.youtube.com/watch?v=Bb7pYthMM64)
* [Cow](https://www.youtube.com/watch?v=f0fNzXP3ku8)
* [Fox II](https://www.youtube.com/watch?v=_fozNTdql6U)
* [Fox III](https://www.youtube.com/watch?v=FGKkIr_IIy4)
