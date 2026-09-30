# SYDE 572

```
index.html              landing page
assets/style.css        stylesheet
1/index.html            Assignment 1 page (built by tools/build.py)
1/media/                plots and the scanned handwritten pages
1/code/part1.py         shortest distance from a point to a function
1/code/part2.py         fitting a line and a parabola
1/code/out_part1.txt    console output, shown on the page
1/code/out_part2.txt
tools/build.py          rebuilds 1/index.html
```

## Running the code

```
pip install numpy matplotlib
cd 1/code
python part1.py
python part2.py
```

Both print their tables and save their plots into `1/media`.

## Handwritten pages

`scan1.jpg` and `scan2.jpg` are Part 1, `scan3.jpg` is Part 2. To replace one,
overwrite the file and rebuild.

## Rebuilding the page

```
cd 1/code && python part1.py > out_part1.txt && python part2.py > out_part2.txt
cd ../.. && python tools/build.py
```
