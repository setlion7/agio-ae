# AGIO

This is the repository for AGIO v0.1, from the paper `Asynchrony and GPUs: Bridging this Dichotomy for I/O with AGIO`.

## Software Requirements
* Linux kernel 6.8
* Nvidia driver 570 (open version)
* CUDA 12.8 and its nvcc compiler
* CMake 3.31 and a C/C++ compiler which supports C++17

Different versions may work, but not tested

## System Configurations
* Disable IOMMU (both in BIOS and kernel) \
  To disable in the kernel, have `iommu=off` and `intel_iommu=off` / `amd_iommu=off` in the kernel command line \
  (Check `/etc/default/grub` if your system uses `grub`, and verify using `dmesg`)
* Build kernel symbols for the Nvidia driver \
  (`sudo make` inside the Nvidia driver source directory, usually located in `/usr/src`)

## Build
```
$ mkdir build; cd build
$ cmake ..
$ make
$ cd module; make
```
You will have AGIO applications (`main-appname`) and the necessary kernel module (`libnvm.ko`) file ready.
This repository includes all dependencies as a git subtree.

## AGIO Configurations (via `.toml` files)
On initial cmake build, cmake will check if the build folder has the AGIO configuration files. If not, it will copy from the sources to the build directory, with a `_in.toml` suffix.
* `options_in.toml`: contains configurations for AGIO
* `work-appname_in.toml`: contains configurations for `appname` application

Please refer to the comments in the configuration files for additional details.

## Application Input Datasets
* All applications require the start address (on the NVMe drive) of the dataset.
* For static applications, each dataset requires a metadata `.json` file containing the information of the dataset. Please refer to the application code for details.
* For dynamic applications, each graph requires a metadata `.json` file containing the information of the CSR graph. Please refer to the `csrmat.hpp` file for additional details.
* Write the datasets to the NVMe drive (e.g., to `/dev/nvmeXn1` by using `dd`), and set its starting addresses (`Y`) in the application configuration file (`.toml`). Make sure you don't need any data on the target NVMe drive, since any writes to the drive will simply overwrite \
`$ dd if=mydata of=/dev/nvmeXn1 bs=1M oflag=seek_bytes seek=Y`

## Bind NVMe drives with AGIO
1. Unbind NVMe device(s) from the default kernel module (`nvme`). Assuming the PCIe BDF of the NVMe device is `0000:e5:00.0`, \
  `$ echo "0000:e5:00.0" | sudo tee /sys/bus/pci/drivers/nvme/unbind`
2. Load the `libnvm` module, and all NVMe devices not bound to the default kernel module will be bound to libnvm, with a device file `/dev/libnvm*` exposed. \
  `$ sudo insmod libnvm.ko`

## Bind NVMe drives back with default kernel module
1. Unload and unbind the NVMe drives from the `libnvm` kernel module \
  `$ sudo rmmod libnvm`
2. For each unbound NVMe drive, bind the drive back to its default kernel module \
  `$ echo "0000:e5:00.0" | sudo tee /sys/bus/pci/drivers/nvme/bind`

## Acknowledgement
AGIO uses libnvm[23] and BaM[36]. AGIO also includes JSON for Modern C++ (`json.hpp`) and toml++ (`toml.hpp`).
