# Pyright Error Fixes Plan for backend/main.py
## Overview
Fix 46 Pyright type errors with type-safe None/string/pandas/torch guards.

## Steps
- [x] Step 1: Fix path/string None handling ✅\n- [ ] Step 2: Fix unbound metrics and model.predict None (348,1369)
- [ ] Step 3: Fix pandas notna/shape/fillna/values (1069,1189,1494,1789,1802,1814)
- [ ] Step 4: Fix Torch PIL unsqueeze (1610,1643)
- [ ] Step 5: Fix matplotlib jet (1658)
- [ ] Step 6: Verify with Pyright: cd backend && pyright .
- [ ] Step 7: Complete

Progress: 0/7
