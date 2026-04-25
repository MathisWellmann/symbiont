# Development environment with a local llama-cpp inference server.
#
# Usage:
#   devenv up        # start llama-server (downloads model on first run)
#   cargo run -p counter-example
#
# The server listens on http://127.0.0.1:8231/v1 (matching .env defaults).
# The model is auto-downloaded from HuggingFace on first launch via
# llama-server's --hf-repo flag and cached in ~/.cache/llama.cpp.
{pkgs, ...}: let
  port = 8231;
in {
  env.BASE_URL = "http://127.0.0.1:${toString port}/v1";
  env.API_KEY = "";
  env.MODEL = "google/gemma-4-E2B-it";

  packages = [
    pkgs.llama-cpp
  ];

  processes.llama-server = {
    exec = ''
      llama-server \
        --hf-repo ggml-org/gemma-4-E2B-it-GGUF \
        --hf-file gemma-4-E2B-it-Q8_0.gguf \
        --host 127.0.0.1 \
        --port ${toString port} \
        --n-gpu-layers 999 \
        --ctx-size 4096
    '';
    ready.http.get = {
      inherit port;
      path = "/health";
    };
  };
}
