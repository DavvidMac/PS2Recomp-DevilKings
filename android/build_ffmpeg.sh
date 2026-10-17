#!/bin/sh
# Builds the part of FFmpeg the runtime's MPEG player uses (the MPEG-1/2 video decoder, its parser
# and swscale) as static arm64 libraries for Android, with the NDK's own clang and make. Point the
# Android build at the result with -DPS2X_FFMPEG_ANDROID_DIR=<install dir>.
#
# usage: build_ffmpeg.sh <ffmpeg source dir> <install dir>
# Environment: ANDROID_NDK (default: the newest NDK under $LOCALAPPDATA/Android/Sdk or $ANDROID_HOME);
# HOST_CC, a C compiler for the build machine (configure tests one; on Windows, LLVM's clang inside
# a "x64 Native Tools" prompt, since Git Bash has no gcc).
# Get the source from https://ffmpeg.org/download.html (check it with its .asc signature).
set -e
[ $# -eq 2 ] || { echo "usage: $0 <ffmpeg source dir> <install dir>" >&2; exit 2; }
# pwd -W (Git Bash) gives C:/... paths, which the NDK's native make understands.
SRC=$(cd "$1" && { pwd -W 2>/dev/null || pwd; })
mkdir -p "$2"
PREFIX=$(cd "$2" && { pwd -W 2>/dev/null || pwd; })

case "$(uname -s)" in
    MINGW*|MSYS*) HOST=windows-x86_64; EXE=.exe; SDK="${ANDROID_HOME:-$(cygpath -m "$LOCALAPPDATA")/Android/Sdk}" ;;
    Darwin) HOST=darwin-x86_64; EXE=; SDK="${ANDROID_HOME:-$HOME/Library/Android/sdk}" ;;
    *) HOST=linux-x86_64; EXE=; SDK="${ANDROID_HOME:-$HOME/Android/Sdk}" ;;
esac
NDK="${ANDROID_NDK:-$(ls -d "$SDK"/ndk/* | sort -V | tail -1)}"
BIN="$NDK/toolchains/llvm/prebuilt/$HOST/bin"
MAKE="$NDK/prebuilt/$HOST/bin/make$EXE"
TARGET=aarch64-linux-android28

mkdir -p "$PREFIX/tmp"
# configure runs test scripts from TMPDIR; Windows' TEMP has backslashes the shell cannot use.
export TMPDIR="$PREFIX/tmp"
# Built in the source tree: the paths in the makefiles stay relative, which the NDK's native make
# needs on Windows (configure would write Git Bash /c/... paths otherwise).
cd "$SRC"
./configure --prefix="$PREFIX" --target-os=android --arch=aarch64 --cpu=armv8-a --enable-cross-compile \
    --cc="$BIN/clang$EXE" --cxx="$BIN/clang++$EXE" --ar="$BIN/llvm-ar$EXE" --nm="$BIN/llvm-nm$EXE" \
    --ranlib="$BIN/llvm-ranlib$EXE" --strip="$BIN/llvm-strip$EXE" --sysroot="$BIN/../sysroot" \
    --extra-cflags="--target=$TARGET" --extra-ldflags="--target=$TARGET" ${HOST_CC:+--host-cc="$HOST_CC"} \
    --enable-static --disable-shared --enable-pic \
    --disable-programs --disable-doc --disable-autodetect --disable-network --disable-everything \
    --disable-avdevice --disable-avformat --disable-avfilter --disable-swresample \
    --enable-avcodec --enable-avutil --enable-swscale \
    --enable-decoder=mpeg1video,mpeg2video --enable-parser=mpegvideo
"$MAKE" -j"$(nproc 2>/dev/null || echo 4)"
"$MAKE" install
echo "FFmpeg for Android in $PREFIX"
