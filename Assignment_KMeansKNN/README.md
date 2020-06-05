KNN algorithm was applied in diabetes classification dataset.
The test accuracy was noted at various values of k.
A maximum test accuracy of 85.13% was reported at k=46.
At this point, the test recall was 91.875% ,the test precision was 88.023% and the test F1-score was 89.91%.
At k=46 (optimal value of k), the training accuracy was 75.82%.
At this point, the training precision was 76.84%, the training recall was 87.31% and the test F1-score was 81.74%.

Steps performed:-
1)z-score normalization was applied since it gave a higher accuracy than min-max normalization.
2)Euclidean distance and Manhattan distance were applied as distance metrics. Euclidean distance gave max test accuracy of 85.13% while Manhattan distance gave a test accuracy of 72.07%.
  So,Euclidean distance was used.
3)Feature ablation study was also performed.It was observed that 'Insulin','BloodPressure' and 'SkinThickness' were the least important features.
  Hence, they were removed and accuracy increases substantially.
