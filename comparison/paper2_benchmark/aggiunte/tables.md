## Global RMSE (Interpolation Phase)

| Model | Mean Position RMSE [km] | Mean Velocity RMSE [km/s] |
|---|---|---|
| MLP | 113,889.98 | 0.08319 |
| PINN | 48,778.49 | 0.04403 |
| HPINN | 34,036.69 | 0.00851 |

## Inner vs Outer (Position and Velocity)

| Model | Inner Pos RMSE [km] | Outer Pos RMSE [km] | Inner Vel RMSE [km/s] | Outer Vel RMSE [km/s] |
|---|---|---|---|---|
| MLP | 162,270.65 | 81,865.42 | 0.16255 | 0.00478 |
| PINN | 23,361.24 | 92,730.22 | 0.06861 | 0.02431 |
| HPINN | 11,850.01 | 70,266.96 | 0.01504 | 0.00248 |

## Chronological Extrapolation (Global Position RMSE)

| Model | 2031-2032 | 2033-2034 | 2035-2036 | 2037-2038 |
|---|---|---|---|---|
| MLP | 510,159,061 | 690,359,729 | 784,484,142 | 829,152,304 |
| PINN | 382,351,678 | 555,626,482 | 683,479,081 | 771,898,490 |
| HPINN | 383,860,080 | 558,122,519 | 695,703,831 | 789,643,696 |

## Chronological Extrapolation (Earth Position Error)

| Model | 2031-2032 | 2033-2034 | 2035-2036 | 2037-2038 |
|---|---|---|---|---|
| MLP | 185,034,731 | 210,130,565 | 205,116,492 | 197,016,601 |
| PINN | 212,290,382 | 152,634,292 | 152,647,512 | 197,294,458 |
| HPINN | 216,052,463 | 146,001,661 | 155,660,996 | 209,488,714 |
