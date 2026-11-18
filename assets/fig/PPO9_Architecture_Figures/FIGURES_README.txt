PPO9 architecture figures | Racer-HitchOpen-RLGPU

1. PPO9_Overview.png / .pdf
   Full input-policy-action-environment loop with independent training-only critic.
   PNG: 4800 x 2840 px. PDF: vector lines and embedded fonts.
   Recommended as the first architecture figure in the repository README.

2. PPO9_Network_Expanded.png / .pdf
   Exact original/reflected actor branches, tied weights, independent recurrent
   histories, antisymmetric beta estimate and stopped-gradient feedback, action
   reflection and mean fusion; independent mirrored critic below.
   PNG: 4800 x 2900 px. PDF: vector lines and embedded fonts.

3. PPO9_Architecture_Notes_zh.txt
   Chinese source audit, feature indexing, action semantics, profile scope,
   source locations and corrections to the reference diagram.

4. draw_ppo9_overview.py / draw_ppo9_expanded.py
   Deterministic vector drawing sources. Require reportlab and DejaVu Sans.
   Font paths near the top can be changed for another operating system.
   Run either Python script to recreate its PDF beside the script.
   PNG rendering example (Poppler):
   pdftoppm -png -singlefile -scale-to-x 4800 -scale-to-y -1 PPO9_Overview.pdf PPO9_Overview

README placement, after copying the PNG files to assets/fig/:

![PPO9 architecture](assets/fig/PPO9_Overview.png)

![PPO9 shared-mirror network](assets/fig/PPO9_Network_Expanded.png)

Suggested figure caption:
PPO9 architecture. The actor processes normalized 51-dimensional observations
and their reflections through tied GRU branches with separate recurrent
histories. A shared auxiliary head produces an antisymmetric sideslip estimate,
which is fed back with stopped gradients. Each branch concatenates its recurrent
features, observations, and signed estimate before a shared action MLP.
Reflection-aligned means define a Gaussian policy with learned, state-independent
variance. An independent privileged critic averages shared-MLP evaluations of
82-dimensional states and their reflections. Ground-truth sideslip supervises
the auxiliary estimate during training; deterministic evaluation uses the policy mean.

Scope:
These figures describe the default PPO9 profile in the supplied ZIP, combined
with the corresponding CLI defaults. They are not inferred from an unavailable
trained checkpoint. The work verifies architecture from source, not training
performance. Optional attention/road encoders are not enabled in this profile.
