# A safer unpickle class for the new age of community-owned checkpoint!

safer_unpickle is a single-file python library which provides several functions
* A core function of loading python pickles using whitelist-guarded class path resolver which fails on forbidden paths
* A function to patch native torch.load method at runtime, thus giving a measure of security to existing pytorch applications, for example to forks of stable-diffusion
* Integrated tool to check checkpoints for class whitelist adherence
* As a bonus, the library includes a shim for pytorch_lightning to avoid requiring it at runtime for checkpoints that were trained with it in their env
