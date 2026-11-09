
Test subject 1:

Normal arm – with calibration, the ML model is better at distinguishing open from fist/thumbs up. The NN is more sure on all the others, is faster and is more certain compared to the ML model, but it mistakes open for thumbs up. Both can be confused when the user uses hard thumb flexion.

NN – right: 100%; left: 100%; thumbs up: 93% (mistaken for open or fist); fist: 95% (mistaken for thumbs up, especially when the fist gesture is aggressive); rest: 100%.

ML – slower and not as certain. It confuses rest and open in 5% of cases, and fist and thumbs up have a 10% error rate. Smaller changes can alter the result due to uncertainty.