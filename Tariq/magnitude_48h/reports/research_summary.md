# 48-hour magnitude research summary

## What did we investigate?

How large is the BTCUSDT endpoint price change after 48 hours, ignoring direction? The target is absolute log return in basis points.

## What improved?

MAE improved from 182.036977 to 180.798518 log bps on the same 18,481 development rows, approximately 0.68%. The retained model combines a refitted pair with random forests. Later alternative models and shorter training windows did not improve the incumbent.

## What remains uncertain?

These are repeatedly used development results. They do not establish future trading profit or statistical significance. Feature publication timing needs more evidence. Some old losing artifacts were intentionally pruned; compact history remains.

## What should happen next?

Review the retained experiment history, verify feature availability and define a focused new experiment only when requested. Keep the incumbent until a replacement passes verification. Original data remains outside GitHub.
