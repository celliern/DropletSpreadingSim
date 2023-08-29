# Benchmark, raw

## Threads.nthreads() = 1

- non hyperbolic: 404.078 μs (19 allocations: 640 bytes)
- hyperbolic: 592.977 μs (13 allocations: 2.62 KiB)
- explicit: 59.787816 seconds
- IMEX (implicit) 469.674910 seconds
- sundials (implicit) 140.704009 seconds

## Threads.nthreads() = 2
- non hyperbolic: 210.751 μs (41 allocations: 3.30 KiB)
- hyperbolic: 322.607 μs (42 allocations: 5.94 KiB)
- explicit: 39.982589 seconds
- IMEX (implicit): 413.464953 seconds
- sundials (implicit): 110.174566 seconds

## Threads.nthreads() = 4
- non hyperbolic: 115.825 μs (86 allocations: 8.67 KiB)
- hyperbolic: 176.008 μs (104 allocations: 12.69 KiB)
- explicit: 29.073363 seconds
- IMEX (implicit): 382.864127 seconds
- sundials (implicit): 73.060998 seconds

## Total time, simulation

- 1 thread: 8018 seconds
- 4 threads: 2771 seconds
- 8 threads: 2000.898205
- 16 threads: 2050 seconds
- GPU: 189 seconds