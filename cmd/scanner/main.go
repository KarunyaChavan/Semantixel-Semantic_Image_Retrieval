package main

import (
	"flag"
	"log"
	"strings"
)

func main() {
	dirsFlag := flag.String("dirs", ".", "Comma-separated list of directories to scan")
	addrFlag := flag.String("addr", "localhost:50051", "gRPC server address")
	concFlag := flag.Int("concurrency", 4, "Max concurrent indexing workers")
	batchFlag := flag.Int("batch", 32, "Image batch size per IndexImages RPC")
	flag.Parse()

	var dirs []string
	if *dirsFlag != "" {
		for _, d := range strings.Split(*dirsFlag, ",") {
			d = strings.TrimSpace(d)
			if d != "" {
				dirs = append(dirs, d)
			}
		}
	}
	if len(dirs) == 0 {
		dirs = flag.Args()
	}
	if len(dirs) == 0 {
		dirs = []string{"."}
	}

	scanner, err := NewScanner(*addrFlag, *concFlag, *batchFlag)
	if err != nil {
		log.Fatalf("Failed to create scanner: %v", err)
	}
	defer scanner.Close()

	if err := scanner.WaitUntilReady(); err != nil {
		log.Fatalf("Server not ready: %v", err)
	}

	log.Printf("Scanning directories: %v (concurrency=%d, batch=%d)", dirs, *concFlag, *batchFlag)
	if err := scanner.ScanDirectories(dirs); err != nil {
		log.Fatalf("Scan failed: %v", err)
	}
}
