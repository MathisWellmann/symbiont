{
  description = "Flake for symbiont";

  inputs = {
    nixpks.url = "github:NixOS/nixpkgs/nixos-unstable";
    rust-overlay.url = "github:oxalica/rust-overlay";
    flake-utils.url = "github:numtide/flake-utils";
  };
  nixConfig = {
    extra-substituters = [
      "https://cache.nixos-cuda.org"
    ];
    extra-trusted-public-keys = [
      "cache.nixos-cuda.org:74DUi4Ye579gUqzH4ziL9IyiJBlDpMRn9MBN8oNan9M="
    ];
  };

  outputs = {
    nixpkgs,
    rust-overlay,
    ...
  }: let
    overlays = [(import rust-overlay)];
    system = "x86_64-linux";
    pkgs = import nixpkgs {
      inherit system overlays;
      config = {
        allowUnfree = true;
        cudaSupport = true;
      };
    };
    rust = pkgs.rust-bin.selectLatestNightlyWith (
      toolchain:
        toolchain.default.override {
          extensions = [
            "rust-src"
            "rust-analyzer"
          ];
          targets = ["x86_64-unknown-linux-gnu"];
        }
    );

    buildInputs = with pkgs; [
      fontconfig
      liberation_ttf
      pkg-config
      rust
    ];
    cuda_inputs = with pkgs; [
      cudatoolkit
    ];
    lsps = with pkgs; [
      marksman # Markdown LSP
      markdown-oxide
      nixd
    ];
    tooling = with pkgs; [
      cargo-nextest
      cargo-flamegraph
      cargo-machete
      cargo-udeps
      cargo-tarpaulin
      cargo-mutants
      cargo-llvm-cov
      cargo-watch
      taplo # Toml toolkit with formatter
      yamlfmt
      mermaid-cli
      devenv
      zola
      llama-cpp
    ];
    nix_tools = with pkgs; [
      alejandra # Nix code formatter
      deadnix # Dead code detection for nix
      statix # Highlights nix antipatterns
    ];
  in {
    devShells.${system} = {
      default = pkgs.mkShell {
        buildInputs =
          buildInputs
          ++ cuda_inputs
          ++ lsps
          ++ tooling
          ++ nix_tools;
        RUST_BACKTRACE = "1";
        RUST_LOG = "info";
        LD_LIBRARY_PATH = "${pkgs.lib.makeLibraryPath (
          buildInputs
          ++ cuda_inputs
        )}:/run/opengl-driver/lib";
        CUDA_PATH = "${pkgs.cudatoolkit}";
      };
    };
  };
}
