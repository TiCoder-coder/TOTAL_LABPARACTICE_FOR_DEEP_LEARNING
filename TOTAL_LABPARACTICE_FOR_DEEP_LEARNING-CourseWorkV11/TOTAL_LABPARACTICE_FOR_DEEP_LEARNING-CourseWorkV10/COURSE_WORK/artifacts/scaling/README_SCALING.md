# SCALING-v1

All StandardScaler estimators are fit once on SPLIT-v1 TRAIN rows and reused for Validation and Test transforms.

Continuous exogenous channels, historical Appliances in FS1 and FS2, and rv1 and rv2 in FS2 are standardized.

hour_sin, hour_cos, dow_sin, dow_cos and weekend pass through unchanged. Metadata never enters a scaler.

YS0 is the identity target option. YS1 uses one TRAIN-only target scaler shared by every feature variant, model, lookback and seed.

YS1 predictions must be inverse-transformed to Wh before MAE, RMSE or R2 computation.

Serialized joblib artifacts must only be loaded from this trusted local artifact set after checksum verification.

Test inspection in Phase 9 is structural only. Test distribution analysis remains locked until Phase 47.
