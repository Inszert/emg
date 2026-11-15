# Preliminary Testing Report – Neural Network (NN) vs. Traditional ML

## 1. Testing Overview

Testing was conducted on 10 subjects with varying genders, arm circumferences, and muscle profiles. None of the subjects were included in the training dataset, meaning that all tests were performed on previously unseen subjects.

The armband was positioned on the right arm throughout all training and testing sessions, with:
- The blue charging marker oriented towards the user.
- The main unit positioned on the upper side of the brachioradialis muscle.

The initial results indicate that the Neural Network (NN) generally performs better than the traditional ML model. However, performance depends heavily on calibration quality, arm circumference, and how consistently the armband maintains contact with the skin.

An important distinction was made between subjects for whom the armband could be fitted properly and subjects with particularly thin arms, for whom the current armband did not provide suitable fit.

## 2. Individual Test Results

### Test Subject 1 – Samo
**Arm type:** Normal arm

The ML model performed better when distinguishing between open-hand and fist/thumbs-up gestures.

The NN was faster, more confident, and generally more accurate across the other gestures. However, it sometimes incorrectly classified an open hand as a thumbs-up gesture.

Both models could experience difficulties when the user applied strong thumb flexion.

**NN results:**
- Right: 100% observed recognition
- Left: 100% observed recognition
- Thumbs up: 93%
- Fist: 95%
- Rest: 100%

**ML results:**
- Generally slower and less confident than the NN.
- Rest and open-hand gestures had approximately a 5% error rate.
- Fist and thumbs-up gestures had approximately a 10% error rate.
- Small changes in signal quality could affect the classification result.

### Test Subject 2 – Klain
**Arm type:** Normal arm width  
**Additional characteristic:** Left-handed user

**Left arm:**

ML:
- Rest, left, and right gestures were recognized correctly.
- Fist and thumbs-up gestures were frequently confused.
- Open-hand recognition was inconsistent, with approximately 50% accuracy.

NN:
- Apart from the rest gesture, predictions were highly inconsistent.

**Right arm:**

ML:
- Fist and thumbs-up gestures were missed twice.
- The remaining gestures were generally recognized correctly.

NN:
- Rest was occasionally confused with thumbs up.
- Open-hand, thumbs-up, and fist recognition appeared to be strongly affected by calibration settings.
- Increasing the rest detection threshold improved performance. Recalibrating the rest position resulted in more accurate predictions.

### Test Subject 3 – Tomas
**Arm type:** Thin arm  
**Armband fit:** Not suitable

Both models correctly recognized left and right movements. However, gestures requiring smaller electromagnetic signal changes were not reliably detected. Most movements were classified as rest or no movement.

Occasionally, fist and strong thumbs-up gestures were detected.

The NN was decisive in recognizing left and right movements but struggled to distinguish the remaining gestures.

Potential improvements:
- Investigate a thinner armband.
- Evaluate alternative sensor positioning.
- Improve contact between the sensors and the skin.

### Test Subject 4 – David
**Arm type:** Wider, average male arm

The subject had a relatively wide arm, with firm muscle contractions.

NN:
- Performed very well overall.

ML:
- Frequently confused fist and thumbs-up gestures.

The main difficulty was distinguishing between fist and thumbs-up movements.

### Test Subject 5 – Oti
**Arm type:** Very wide arm

NN:
- Generally performed well.

ML:
- Had difficulties recognizing thumbs-up and open-hand gestures.
- Frequently confused fist and thumbs-up.

Both models experienced some uncertainty, potentially due to differences in arm circumference and sensor positioning.

### Test Subject 6 – Kata
**Arm type:** Normal arm

NN:
- Performed reasonably well.

ML:
- Was less accurate than the NN.

The results further suggest that calibration quality and gesture strength have a significant influence on classification accuracy.

### Test Subject 7 – Julka
**Arm type:** Thin arm  
**Armband fit:** Not suitable

NN:
- Fist and thumbs-up gestures were frequently confused.
- Predictions fluctuated rapidly, with very high uncertainty in some cases.
- Some prediction changes occurred within approximately one second.

ML:
- Gestures other than open hand were generally recognized correctly.

The results indicate potential difficulties with signal consistency and gesture differentiation.

### Test Subject 8 – Terka
**Arm type:** Thin arm  
**Armband fit:** Not suitable

- Left and right movements were recognized correctly.
- Rest was recognized correctly.
- Fist recognition was approximately 50%.
- The remaining gestures produced inconsistent or uncertain predictions.

The main difficulty was distinguishing gestures requiring relatively small signal changes.

### Test Subject 9 – Berti
**Arm type:** Normal, slightly thinner than average

NN:
- Demonstrated very consistent and confident classification, with estimated performance of up to approximately 95%.

ML:
- Predictions were occasionally uncertain.

Despite having a slightly thinner-than-average arm, the NN demonstrated strong performance.

### Test Subject 10 – Name not provided
**Arm type:** Normal, wider and muscular

NN:
- Demonstrated very consistent performance, with estimated accuracy of up to approximately 95%.

ML:
- Predictions were less confident, with confidence reportedly below 75% in some cases.
- Nevertheless, the ML model performed better than it did for Test Subject 9.

The NN again demonstrated superior consistency and prediction confidence.

## 3. Armband Fit and Classification Accuracy

For a more meaningful comparison, subjects with particularly thin arms and unsuitable armband fit were excluded from the armband-fit performance assessment.

**Excluded subjects:**
- Tomas (Subject 3)
- Julka (Subject 7)
- Terka (Subject 8)

These subjects remain part of the overall testing results but are not included in the assessment of performance under suitable armband-fit conditions.

Berti (Subject 9) remains included because his arm was within the normal range, although slightly thinner than average.

This leaves 7 subjects in the armband-fit assessment.

### Estimated accuracy under different conditions

| Testing conditions | NN | Traditional ML |
|---|---:|---:|
| Very good fit and strong signal | 90–95% | 80–85% |
| Reasonably good fit and signal | ~84% | 70–80% |
| Acceptable fit, some signal inconsistency | 75–83% | 60–70% |
| Poor fit or unsuitable armband size | 50–74% | 40–60% |

These are preliminary engineering estimates, not measured group averages.

### Fit-adjusted performance assessment

For the 7 subjects with normal or wider arm characteristics and a more suitable armband fit, the estimated performance is:

- **NN: approximately 84–95%.**
- **Traditional ML: approximately 70–85%.**

The three subjects excluded due to unsuitable armband fit demonstrated substantially more difficulties, particularly with gestures requiring smaller electromagnetic signal changes.

The distinction suggests that physical armband fit is an important factor influencing the performance of both models.

## 4. Main Findings and Challenges

1. **NN generally outperforms ML:** The NN demonstrated faster predictions, higher confidence, and better overall classification performance in most test scenarios.

2. **Calibration is critical:** Incorrect calibration, particularly of the rest position, can significantly reduce classification accuracy. Recalibration can considerably improve performance.

3. **Sensor-to-skin contact:** Armband fit and consistent skin contact appear to be among the most important factors influencing signal quality.

4. **Arm circumference:** Particularly thin arms can introduce challenges because the current armband may not maintain sufficient sensor contact. Very wide arms may also require attention to positioning.

5. **Thumb strength:** The strength and extent of thumb flexion significantly influence recognition accuracy.

6. **Fist vs. thumbs up:** This remains one of the most frequent classification problems across both models.

7. **Prediction stability:** The NN generally produces more decisive predictions, while the ML model is more likely to show uncertainty or fluctuate between classifications.

8. **Individual anatomical differences:** Arm circumference alone does not determine performance. Muscle activation, arm shape, gesture execution, and sensor contact also contribute significantly.

## 5. Recommendations and Conclusions

The NN appears to be the more promising approach for gesture classification, particularly when tested on previously unseen subjects.

Further development should focus on:

- Improving and standardizing calibration procedures, particularly for the rest position.
- Developing alternative armband sizes for users with thinner arms.
- Improving sensor-to-skin contact.
- Improving differentiation between fist and thumbs-up gestures.
- Investigating the influence of thumb strength and gesture execution intensity.
- Improving prediction stability for subjects with weaker or smaller electromagnetic signal changes.
- Evaluating performance across a larger and more diverse group of subjects.

Overall, the NN demonstrates better generalization and more consistent performance than the traditional ML model.

The estimated NN accuracy is approximately 84–95% under suitable armband-fit conditions, while the traditional ML model is estimated at approximately 70–85%.

When the armband does not fit properly, performance can decrease considerably. This indicates that improving the physical design and sizing of the armband could be just as important as further improving the classification algorithms.

## 6. Statistical Limitations

The available observations do not provide enough information to calculate statistically validated overall accuracy percentages.

The current accuracy figures are engineering estimates based on observed behaviour and individual gesture results.

For a statistically reliable comparison, future testing should record:
- Total repetitions per gesture and subject.
- Number of correct and incorrect predictions.
- Actual armband-fit classification for every subject.
- Calibration quality.
- Prediction confidence and stability.

This would allow the calculation of separate accuracy measurements for suitable and unsuitable armband-fit conditions, as well as a reliable comparison between NN and traditional ML.