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

  # Concurrent decoding slots. `llama-server` serves one sequence at a time
  # unless told otherwise, so without this the concurrent lanes of
  # `Runtime::evolve_batch` would queue one behind another and the batched
  # example would demonstrate nothing. Matches the lane count of
  # `examples/batched-evolution`.
  slots = 8;

  # Context per slot. `--ctx-size` is the *total* KV budget, which
  # `llama-server` divides evenly across `--parallel` slots, so it has to be
  # scaled with the slot count to keep each slot usable. 16k is comfortably
  # above what any example needs — the largest preamble in the tree is
  # `struct-support`'s rustdoc-augmented one, and that is a few thousand
  # tokens.
  ctxPerSlot = 16384;
in {
  processes.llama-server = {
    exec = ''
      llama-server \
        -hf prism-ml/Bonsai-8B-gguf \
        --alias local \
        --host 127.0.0.1 \
        --port ${toString port} \
        --n-gpu-layers 999 \
        --parallel ${toString slots} \
        --ctx-size ${toString (slots * ctxPerSlot)}
    '';
    ready.http.get = {
      inherit port;
      path = "/health";
    };
  };
}
