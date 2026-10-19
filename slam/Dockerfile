FROM --platform=linux/arm64 ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG ORB_SLAM3_COMMIT=4452a3c4ab75b1cde34e5505a36ec3f9edcdc4c4
ARG PANGOLIN_COMMIT=aff6883c83f3fd7e8268a9715e84266c42e2efe3

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        git \
        libboost-serialization-dev \
        libegl1-mesa-dev \
        libeigen3-dev \
        libepoxy-dev \
        libgl1-mesa-dev \
        libglew-dev \
        libopencv-dev \
        libssl-dev \
        libwayland-dev \
        libxkbcommon-dev \
        pkg-config \
        wayland-protocols \
    && rm -rf /var/lib/apt/lists/*

RUN git init /opt/Pangolin \
    && git -C /opt/Pangolin remote add origin https://github.com/stevenlovegrove/Pangolin.git \
    && git -C /opt/Pangolin fetch --depth 1 origin "${PANGOLIN_COMMIT}" \
    && git -C /opt/Pangolin checkout --detach FETCH_HEAD \
    && cmake -S /opt/Pangolin -B /opt/Pangolin/build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_STANDARD=17 \
        -DBUILD_EXAMPLES=OFF \
        -DBUILD_PANGOLIN_PYTHON=OFF \
        -DBUILD_TESTS=OFF \
        -DBUILD_TOOLS=OFF \
    && cmake --build /opt/Pangolin/build --parallel 4 \
    && cmake --install /opt/Pangolin/build \
    && ldconfig

RUN git init /opt/ORB_SLAM3 \
    && git -C /opt/ORB_SLAM3 remote add origin https://github.com/UZ-SLAMLab/ORB_SLAM3.git \
    && git -C /opt/ORB_SLAM3 fetch --depth 1 origin "${ORB_SLAM3_COMMIT}" \
    && git -C /opt/ORB_SLAM3 checkout --detach FETCH_HEAD \
    && sed -i 's/ -march=native//g; s/-march=native //g; s/-std=c++11/-std=c++14/g' \
        /opt/ORB_SLAM3/CMakeLists.txt \
        /opt/ORB_SLAM3/Thirdparty/DBoW2/CMakeLists.txt \
        /opt/ORB_SLAM3/Thirdparty/g2o/CMakeLists.txt \
    && cmake -S /opt/ORB_SLAM3/Thirdparty/DBoW2 \
        -B /opt/ORB_SLAM3/Thirdparty/DBoW2/build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_STANDARD=14 \
    && cmake --build /opt/ORB_SLAM3/Thirdparty/DBoW2/build --parallel 4 \
    && cmake -S /opt/ORB_SLAM3/Thirdparty/g2o \
        -B /opt/ORB_SLAM3/Thirdparty/g2o/build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_STANDARD=14 \
    && cmake --build /opt/ORB_SLAM3/Thirdparty/g2o/build --parallel 4 \
    && tar -xzf /opt/ORB_SLAM3/Vocabulary/ORBvoc.txt.tar.gz \
        -C /opt/ORB_SLAM3/Vocabulary \
    && cmake -S /opt/ORB_SLAM3 -B /opt/ORB_SLAM3/build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_STANDARD=14 \
    && cmake --build /opt/ORB_SLAM3/build --target ORB_SLAM3 --parallel 4

# The pinned upstream Shutdown() has its worker wait commented out. Exporting
# poses while local mapping or global BA still writes them races those workers.
COPY slam/orbslam3_shutdown.patch /tmp/orbslam3_shutdown.patch
RUN patch -d /opt/ORB_SLAM3 -p1 < /tmp/orbslam3_shutdown.patch \
    && cmake --build /opt/ORB_SLAM3/build --target ORB_SLAM3 --parallel 4

COPY slam/CMakeLists.txt /opt/smoke-rescue-runner/CMakeLists.txt
COPY slam/rgbd_runner.cc /opt/smoke-rescue-runner/rgbd_runner.cc

RUN cmake -S /opt/smoke-rescue-runner -B /opt/smoke-rescue-runner/build \
        -DCMAKE_BUILD_TYPE=Release \
    && cmake --build /opt/smoke-rescue-runner/build --parallel 4 \
    && cmake --install /opt/smoke-rescue-runner/build \
    && ldconfig

ENV LD_LIBRARY_PATH=/opt/ORB_SLAM3/lib:/opt/ORB_SLAM3/Thirdparty/DBoW2/lib:/opt/ORB_SLAM3/Thirdparty/g2o/lib
ENTRYPOINT ["/usr/local/bin/smoke_rescue_rgbd"]
CMD ["/recording", "/output"]
