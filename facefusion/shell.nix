{ pkgs ? import <nixpkgs> { config.allowUnfree = true; } }:

let
  cudaLibs = with pkgs.cudaPackages; [
    cuda_cudart
    cuda_nvrtc
    libcublas
    libcufft
    libcurand
    cudnn
  ];
  runtimeLibs = with pkgs; [
    stdenv.cc.cc.lib
    zlib
    glib
    libGL
    xorg.libxcb
    xorg.libX11
    xorg.libXext
    xorg.libSM
    xorg.libICE
  ] ++ cudaLibs;
in
pkgs.mkShell {
  packages = with pkgs; [
    python312
    python312Packages.pip
    python312Packages.virtualenv
    git
    ffmpeg
  ] ++ runtimeLibs;

  LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath runtimeLibs;

  shellHook = ''
    if [ ! -d .venv ]; then
      python -m venv .venv
    fi

    source .venv/bin/activate
    export PIP_DISABLE_PIP_VERSION_CHECK=1
  '';
}
