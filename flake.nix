{
  description = "Flake for symbiont";

  inputs = {
    nixpks.url = "github:NixOS/nixpkgs/nixos-unstable";
    rust-overlay.url = "github:oxalica/rust-overlay";
    flake-utils.url = "github:numtide/flake-utils";
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
      pkg-config
      rust
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
    ];
    nix_tools = with pkgs; [
      alejandra # Nix code formatter
      deadnix # Dead code detection for nix
      statix # Highlights nix antipatterns
    ];
  in {
    nixosModules.zola-serve = import ./symbiont/modules/nixos/zola-serve.nix;

    devShells.${system} = {
      default = pkgs.mkShell {
        buildInputs =
          buildInputs
          ++ lsps
          ++ tooling
          ++ nix_tools;
        RUST_BACKTRACE = "1";
        RUST_LOG = "info";
      };
    };
  };
}
