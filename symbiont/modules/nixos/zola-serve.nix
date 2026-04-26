{
  config,
  lib,
  pkgs,
  ...
}: let
  cfg = config.services.zola-serve;
in {
  options.services.zola-serve = {
    enable = lib.mkEnableOption "Zola static site server";

    root = lib.mkOption {
      type = lib.types.path;
      default = "/home/m/MathisWellmann/symbiont/website";
      description = "Path to the Zola website root directory.";
    };

    port = lib.mkOption {
      type = lib.types.port;
      default = 1111;
      description = "Port to serve the site on.";
    };

    hostname = lib.mkOption {
      type = lib.types.str;
      default = "127.0.0.1";
      description = "Hostname to bind the server to.";
    };
  };

  config = lib.mkIf cfg.enable {
    systemd.services.zola-serve = {
      description = "Zola static site server";
      wantedBy = ["multi-user.target"];
      wants = ["network-online.target"];
      after = ["network-online.target"];

      serviceConfig = {
        Type = "simple";
        ExecStart = "${pkgs.zola}/bin/zola serve --port ${toString cfg.port} --interface ${cfg.hostname}";
        Restart = "on-failure";
        RestartSec = "5";
        WorkingDirectory = cfg.root;
        StandardOutput = "journal";
        StandardError = "journal";
      };
    };
    networking.firewall.allowedUDPPorts = [
      cfg.port
    ];
  };
}
