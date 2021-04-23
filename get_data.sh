uname -r > uname.out
sudo lshw -c video > lshw.out 
apt show rocm-libs -a > apt.out
/opt/rocm/opencl/bin/clinfo > clinfo.out
/opt/rocm/bin/rocminfo > rocminfo.out
