This is a groupproject completed during my Masterstudies at Imperial College submitted the 13/12/2019.
The inital commit of this repository is the submitted version. Any later commits have been added after submission.

Contributers to initial commit:

Sotiris Gkoulimaris,
Helen Situ,
Jiabo Wang,
Qing Ma,
Nikolas Vornehm,

# ACSE-4-armageddon

Asteroids entering Earth’s atmosphere are subject to extreme drag forces that decelerate, heat and disrupt the space rocks. The fate of an asteroid is a complex function of its initial mass, speed, trajectory angle and internal strength. 

[Asteroids](https://en.wikipedia.org/wiki/Asteroid) 10-100 m in diameter can penetrate deep into Earth’s atmosphere and disrupt catastrophically, generating an atmospheric disturbance ([airburst](https://en.wikipedia.org/wiki/Air_burst)) that can cause [damage on the ground](https://www.youtube.com/watch?v=tq02C_3FvFo). Such an event occurred over the city of [Chelyabinsk](https://en.wikipedia.org/wiki/Chelyabinsk_meteor) in Russia, in 2013, releasing energy equivalent to about 520 [kilotons of TNT](https://en.wikipedia.org/wiki/TNT_equivalent) (1 kt TNT is equivalent to 4.184e12 J), and injuring thousands of people ([Popova et al., 2013](http://doi.org/10.1126/science.1242642); [Brown et al., 2013](http://doi.org/10.1038/nature12741)). An even larger event occurred over [Tunguska](https://en.wikipedia.org/wiki/Tunguska_event), an unpopulated area in Siberia, in 1908. 

This tool predicts the fate of asteroids entering Earth’s atmosphere for the purposes of hazard assessment.

### Installation Guide

After cloning this repository, please install software requirements by running
```
pip install -r requirements.txt
```

### User instructions

Module can be imported with
```
>>> import armageddon
```
The core functionality is to simulate an asteroid entering the atmosphere. 
It can be called in the following example format:
```
>>> earth = armageddon.Planet()
>>> results, outcomes = earth.impact(radius=,velocity=,density=,strength=,angle=)
```
Where the specified parameter values can be filled in as desired. 

Please refer to the documentation found in docs_build/index.html for more detailed descriptions of the functions. 

### Documentation

The code includes [Sphinx](https://www.sphinx-doc.org) documentation. On systems with Sphinx installed, this can be built by running

```
python -m sphinx docs html
```

then viewing the `index.html` file in the `html` directory in your browser.

For systems with [LaTeX](https://www.latex-project.org/get/) installed, a manual pdf can be generated by running

```
python -m sphinx  -b latex docs latex
```

Then following the instructions to process the `Armageddon.tex` file in the `latex` directory in your browser.

### Testing

The tool includes several tests, which you can use to checki its operation on your system. With [pytest](https://doc.pytest.org/en/latest) installed, these can be run with

```
python -m pytest armageddon
```
