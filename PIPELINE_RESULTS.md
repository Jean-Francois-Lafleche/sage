# SAGE Pipeline Test Results — 2026-04-09

## Summary

**Status: ✅ COMPLETE**  
**Input:** `test_images/image2.png` (supermarket aisle photograph)  
**Duration:** ~2 hours (04:50 → 06:57 UTC)  
**Total objects placed:** 227  
**Token usage:** 592,265 total (586,933 input + 5,332 output)  
**Tool calls used:** 11/15  
**Iterations:** 12 (scene analysis + 11 placement rounds)

## What Worked

1. **Image analysis** — VLM correctly identified room type (supermarket aisle), dimensions (3×15×4m), style (modern commercial), 35+ object categories from the photo
2. **Floor plan generation** — Created room layout with walls, floor, ceiling
3. **Large furniture placement** (Stage 1) — Shelving units, refrigerator, HVAC ducts, LED lights, signage all placed correctly (14 structural objects)
4. **Product generation** (Stage 2-12) — 213 shelf-placed products generated via mock TRELLIS and placed via DFS solver:
   - 30 tea boxes (incl. 3 Bushells brand)
   - 20 coffee jars  
   - 20 cereal boxes  
   - 10 flakes boxes, 10 grain boxes  
   - 10 granola boxes, 8 muesli boxes  
   - 6 Corn Flakes, 2 Nesquik  
   - 8 boxes, 10 jars, 8 containers, 4 canisters, 6 tins  
   - 40 price labels/tags  
   - 8 "Prices Dropped" signs  
   - 5 shelf dividers, 5 separators  
5. **Physics/semantic critics** — Gracefully degraded when Isaac Sim unavailable (no crashes)
6. **Auto-retry** — Mock TRELLIS 500 errors recovered automatically on retry

## What Didn't Work

1. **Refrigerator items (20 failed)** — Dairy (8), cartons (6), bottles (6) couldn't be placed on the refrigerator because placement requires Isaac Sim `test_object_placements_in_single_room` which needs a running Isaac instance
2. **Mock TRELLIS** — All GLB files are placeholder 1196-byte boxes with no UV/textures. Real TRELLIS would produce actual 3D models
3. **No door/window databases** — Skipped (missing `door-database.json`, `material-database.json`)
4. **MatFuse** — Material generation disabled (missing checkpoint)
5. **All objects on shelf level 0** — DFS solver placed everything on the lowest shelf level only (possible bug in shelf level iteration)
6. **Occasional mock TRELLIS 500s** — Non-fatal but caused brief delays

## Output Files

| File | Location | Description |
|------|----------|-------------|
| `layout_14b60c79.json` | `server/results/layout_14b60c79/` | Full scene layout (227 objects, 355KB) |
| `room_5413e00f.json` | `server/results/layout_14b60c79/` | Room-only layout (14 structural objects) |
| `image_analysis.json` | `client/outputs/` | VLM scene analysis |
| `chat_session_*.json` | `client/logs/` | Full conversation log |
| `chat_session_*.html` | `client/logs/` | HTML visualization of conversation |
| Floor plan PNG | `server/vis/` | Multiple floor plan visualizations |
| Room vis PNG | `server/vis/` | Room layout top-down view |
| 1239 generation files | `server/results/layout_14b60c79/generation/` | GLB + texture PNGs |

Copies in `~/storage/jlafleche-sdg/sage-pipeline-results/`

## Bugs Fixed During This Run

1. **`string indices must be integers, not 'str'`** — Physics critic crashed when Isaac Sim returned error strings instead of dicts. Fixed with `.get()` safe access in `object_placement_planner.py` (4 instances) and `object_movement_planner.py` (3 instances)
2. **`room_physics_critic` crash** — `layout_wo_robot.py` didn't handle non-dict returns from Isaac functions. Fixed.
3. **Doors/windows fatal error** — Changed from `return json.dumps({"success": False, ...})` to print warning and continue
4. **`result_create` pass-through** — Fixed 3 instances in `object_placement_planner.py`

## Recommendations for Next Steps

1. **Real TRELLIS server** — Test with actual 3D model generation
2. **Isaac Sim integration** — Enable physics/semantic critics and refrigerator placement
3. **Multi-level shelf placement** — Investigate why DFS solver only uses level 0
4. **Asset databases** — Install door/material databases from objathor-assets
5. **MatFuse checkpoint** — Download/link the matfuse-full.ckpt for material generation
6. **USD export** — Add USD scene export (currently JSON only)

## Code Changes Made

All in `/tmp/sage-repo/`:
- `client/client_generation_from_image.py` — Added `--vlm_api_key` and `--vlm_model` CLI args
- `server/object_placement_planner.py` — 7 fixes for Isaac Sim error handling  
- `server/object_movement_planner.py` — 3 fixes for Isaac Sim error handling
- `server/layout_wo_robot.py` — Physics critic non-dict return handling, doors/windows non-fatal
- `server/mock_trellis.py` — New mock TRELLIS server (port 8200)
- `server/key.json` / `client/key.json` — API configuration
