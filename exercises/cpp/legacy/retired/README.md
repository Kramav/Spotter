# Retired programs (not built)

Programs from `examples/` that a newer one replaced. They're kept here so you can read them, but **nothing
builds them**: they aren't in any CMake list, even with `-DSPOTTER_BUILD_LEGACY=ON`. Only the programs we actually
use get built (D58). They aren't maintained, and they may stop compiling as the library changes.

| Program | Replaced by | Why |
|---|---|---|
| `Demo4_Room_Twin` | `examples/demos/Demo7_Room_Twin` | It hid everything over 1.6 m above the floor. That took the ceiling off, but it cut every wall's top off too. Demo 7 finds the ceiling and hides only that (C shows it again), keeping the walls at full height. With G off, Demo 7 looks like Demo 4. |
| `Demo6_Room_Filled` | `examples/demos/Demo7_Room_Twin` | Demo 4 plus the hole filling, with the same 1.6 m cut. The fill only went up to 1.6 m. Demo 7 fills up to the ceiling it finds, and it also runs with no Kinect. |
| `13x_Room_Fill` | `examples/13v_Room_Fill` | The same 1.6 m cut and fill height. 13v finds the ceiling, fills the walls up to it, and C shows or hides it. |

Their fill code (`fill_holes.hpp`) is now in the library as `spotter/room_fill.hpp`. Their includes point there,
so they still compile if you add one back to a CMake list. `legacy/presentation/` has the older, original Demo 4
and Demo 6 (D22, D31): different programs with the same names.
