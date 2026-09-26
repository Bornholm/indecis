module github.com/bornholm/indecis/plugins/injection-detector

go 1.27

require (
	github.com/bornholm/indecis v0.0.0
	github.com/xolo-gateway/xolo/pkg/pluginsdk v0.0.0
)

require (
	github.com/fatih/color v1.13.0 // indirect
	github.com/golang/protobuf v1.5.4 // indirect
	github.com/hashicorp/go-hclog v1.6.3 // indirect
	github.com/hashicorp/go-plugin v1.7.0 // indirect
	github.com/hashicorp/yamux v0.1.2 // indirect
	github.com/mattn/go-colorable v0.1.12 // indirect
	github.com/mattn/go-isatty v0.0.17 // indirect
	github.com/oklog/run v1.1.0 // indirect
	golang.org/x/net v0.48.0 // indirect
	golang.org/x/sys v0.39.0 // indirect
	golang.org/x/text v0.32.0 // indirect
	google.golang.org/genproto/googleapis/rpc v0.0.0-20251202230838-ff82c1b0f217 // indirect
	google.golang.org/grpc v1.79.2 // indirect
	google.golang.org/protobuf v1.36.11 // indirect
)

replace (
	github.com/bornholm/indecis => ../..
	// Le SDK n'est pas encore tagué : on pointe sur le dépôt Xolo voisin.
	github.com/xolo-gateway/xolo/pkg/pluginsdk => ../../../xolo/pkg/pluginsdk
)
