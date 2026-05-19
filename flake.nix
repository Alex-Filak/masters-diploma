{
  description = "Reproducible environment for the diploma-new face swap experiments";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
      };
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
    in {
      devShells.${system}.default = pkgs.mkShell {
        packages = with pkgs; [
          bashInteractive
          cacert
          coreutils
          ffmpeg
          git
          gnutar
          python312
          python312Packages.pip
          python312Packages.virtualenv
          python312Packages.jupyterlab
        ] ++ runtimeLibs;

        LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath runtimeLibs;

        shellHook = ''
          REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
          VENV_DIR="$REPO_ROOT/.venv"
          STAMP_PATH="$VENV_DIR/.requirements.sha256"

          requirements_hash() {
            sha256sum \
              "$REPO_ROOT/facefusion/requirements.txt" \
              "$REPO_ROOT/face-swap-experiment/requirements.txt" \
              | sha256sum | cut -d' ' -f1
          }

          if [ ! -d "$VENV_DIR" ]; then
            python -m venv "$VENV_DIR"
          fi

          source "$VENV_DIR/bin/activate"
          export PIP_DISABLE_PIP_VERSION_CHECK=1
          export PYTHONNOUSERSITE=1

          CURRENT_HASH="$(requirements_hash)"
          PREVIOUS_HASH=""
          if [ -f "$STAMP_PATH" ]; then
            PREVIOUS_HASH="$(cat "$STAMP_PATH")"
          fi

          if [ "$CURRENT_HASH" != "$PREVIOUS_HASH" ]; then
            python -m pip install \
              -r "$REPO_ROOT/facefusion/requirements.txt" \
              -r "$REPO_ROOT/face-swap-experiment/requirements.txt"
            printf '%s' "$CURRENT_HASH" > "$STAMP_PATH"
          fi

          export REPO_ROOT
          export DIPLOMA_ENV_READY=1
        '';
      };
    };
}
