# Pyspice-memory
Hi, I have a problem that when using PySpice simulations, MiB keeps getting added to the RAM. Specifically, it always happens with the command analysis:
analysis = simulator.transient(**SimParams). Specifically, it is about 13 MiB, which makes it impossible to run this program, as there are so many simulations that even 16 GB of RAM is not enough. I am using the commands destroy, reset, gc.collect, but I don’t know what else to do. Thank you for any advice
