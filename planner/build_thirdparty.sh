ROOT_DIR=$(cd $(dirname "$0"); pwd)

# build gtsam
cd ${ROOT_DIR}/lib/3rdparty/gtsam-4.1.1
rm -rf build install
mkdir build install
cd build
cmake .. -DCMAKE_INSTALL_PREFIX="../install" -DCMAKE_BUILD_TYPE=Release -DGTSAM_USE_SYSTEM_EIGEN=ON  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 
make -j6 && make install

# build osqp
cd ${ROOT_DIR}/lib/3rdparty/osqp
rm -rf build install
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX="../install" -DCMAKE_BUILD_TYPE=Release  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 
make -j4 && make install
