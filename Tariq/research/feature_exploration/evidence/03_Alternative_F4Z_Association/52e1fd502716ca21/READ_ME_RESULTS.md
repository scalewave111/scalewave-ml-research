# Alternative F4Z association ranking
Target: BTCUSDT forward 48-hour signed return strictly above +500 log bps.
Ranking: descending 5th-percentile weekly-block-resampled, quarter-adjusted risk difference.
Bands learned before the ranking period; reserved validation never changes the ordering.
Features examined: 16. Eligible: 14. Reported: 10.

## Frozen results
### 1. f4z__flow__synthetic_15_rz__1h__value
Condition: -2.8187673 <= value < 2.8443652
Ranking effect: 79.538 percentage points; score: 77.018.
Reserved-period effect: 83.162 points; positive observed association.
Evidence: positive lower resampling bound; still selection-biased

### 2. f4z__flow__synthetic_1_rz__1h__value
Condition: 1.5299432 <= value < +infinity
Ranking effect: 74.927 percentage points; score: 72.336.
Reserved-period effect: 73.141 points; positive observed association.
Evidence: positive lower resampling bound; still selection-biased

### 3. f4z__flow__synthetic_0_rz__1h__value
Condition: 1.5577524 <= value < +infinity
Ranking effect: 72.566 percentage points; score: 70.002.
Reserved-period effect: 75.455 points; positive observed association.
Evidence: positive lower resampling bound; still selection-biased

### 4. f4z__flow__synthetic_2_rz__1h__value
Condition: 1.5429132 <= value < +infinity
Ranking effect: 71.730 percentage points; score: 69.348.
Reserved-period effect: 71.806 points; positive observed association.
Evidence: positive lower resampling bound; still selection-biased

### 5. f4z__flow__synthetic_4_rz__1h__value
Condition: 1.4830948 <= value < +infinity
Ranking effect: 70.486 percentage points; score: 68.137.
Reserved-period effect: 68.340 points; positive observed association.
Evidence: positive lower resampling bound; still selection-biased

### 6. f4z__flow__synthetic_3_rz__1h__value
Condition: 1.5413846 <= value < +infinity
Ranking effect: 71.016 percentage points; score: 67.866.
Reserved-period effect: 70.292 points; positive observed association.
Evidence: positive lower resampling bound; still selection-biased

### 7. f4z__flow__synthetic_5_rz__1h__value
Condition: 1.4912458 <= value < +infinity
Ranking effect: 69.165 percentage points; score: 66.310.
Reserved-period effect: 67.388 points; positive observed association.
Evidence: positive lower resampling bound; still selection-biased

### 8. f4z__flow__synthetic_6_rz__1h__value
Condition: 1.4763542 <= value < +infinity
Ranking effect: 67.718 percentage points; score: 65.128.
Reserved-period effect: 68.636 points; positive observed association.
Evidence: positive lower resampling bound; still selection-biased

### 9. f4z__flow__synthetic_9_rz__1h__value
Condition: 1.4430068 <= value < +infinity
Ranking effect: 65.365 percentage points; score: 63.074.
Reserved-period effect: 63.124 points; positive observed association.
Evidence: positive lower resampling bound; still selection-biased

### 10. f4z__flow__synthetic_7_rz__1h__value
Condition: 1.4557598 <= value < +infinity
Ranking effect: 65.860 percentage points; score: 62.952.
Reserved-period effect: 65.142 points; positive observed association.
Evidence: positive lower resampling bound; still selection-biased

## Interpretation limits
This is individual-feature association, not causation or a combined trading strategy.
The resampling bands are not selection-adjusted significance statements.
Multiple testing, related columns, changing coverage and dependence across blocks remain.
Data exports are not independently verified against their upstream build.
No transaction costs, drawdowns, position sizing or return expectancy were tested.