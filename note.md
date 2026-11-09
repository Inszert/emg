
Test subject 1:

Normal arm, with calibration: the ML model is better at distinguishing open from fist/thumbs up. The NN is more sure on all the others, is faster, and is more certain than the ML model, but it mistakes open for thumbs up. Both can be confused when the user uses hard thumb flexion.

NN: right – 100%; left – 100%; thumbs up – 93% (mistaken for open or fist); fist – 95% (mistaken for thumbs up, especially when the fist gesture is aggressive); rest – 100%.
ML: slower, less certain; mistakes rest and open in 5% of cases; fist and thumbs up 10% error rate; a smaller change can change the result due to uncertainty.

Test subject 2

Left arm

ML: rest okay, left and right ok. Fist and thumb confused. Left-handed, normal arm width. Open is confused but gets it 50/50.

NN: other than  rest, chaos.

Right arm

ML: fist and thumbs up missed 2x; the others seemed right.

NN: rest – thumbs-up mix-up; open, thumb and fist react based on the calibration on a left-hander.
Rest limit needs to be increased; recalibrating rest makes it accurate.



Test subject 3


Tomas

Thin arm. Right and left hand correctly predicted, but other movements that need small electromagnetic changes were not visible; everything was possibly taken as rest.
Solution: thinner armband, wider arm. The NN could only see right and left; everything else was rest or none, decisive. Sometimes fist and hard thumbs up showed up.



Test subject 4


David
had a wider arm would say an averga male arm, the contast were firm. 
Thumbs up and fist mixed. NN worked beautifully; ML mixed fist and thumbs up.
