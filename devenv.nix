# Development environment with a local llama-cpp inference server.
#
# Usage:
#   nix develop
#   devenv up        # start llama-server (downloads model on first run)
#   cargo run -p counter-example
#
# The server listens on http://127.0.0.1:8231/v1 (matching .env defaults).
# The model is auto-downloaded from HuggingFace on first launch via
# llama-server's --hf-repo flag and cached in ~/.cache/llama.cpp.
_: let
  port = 8231;
in {
  processes.llama-server = {
    exec = ''
      llama-server \
        -hf LiquidAi/LFM2.5-8B-A1B-GGUF \
        --host 127.0.0.1 \
        --port ${toString port} \
        --n-gpu-layers 999 \
        --ctx-size 32768
    '';
    ready.http.get = {
      inherit port;
      path = "/health";
    };
  };
}
