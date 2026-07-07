package main

import (
	"context"
	"io/fs"
	"log"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	pb "semantixel-scanner/semantixelpb"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

var imageExts = map[string]bool{".jpg": true, ".jpeg": true, ".png": true, ".webp": true, ".bmp": true, ".tiff": true}
var videoExts = map[string]bool{".mp4": true, ".mkv": true, ".avi": true, ".mov": true, ".webm": true}
var audioExts = map[string]bool{".mp3": true, ".wav": true, ".flac": true, ".m4a": true, ".ogg": true}

const (
	chunkSize   = 500
	maxMsgSize  = 512 * 1024 * 1024
)

type job struct {
	kind     string // "image_batch", "video", "audio"
	paths    []string
	metadata []*pb.IndexMediaMetadata
}

type Scanner struct {
	conn        *grpc.ClientConn
	client      pb.SemantixelInferenceClient
	concurrency int
	batchSize   int
}

func NewScanner(address string, concurrency int, batchSize int) (*Scanner, error) {
	conn, err := grpc.Dial(address,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithDefaultCallOptions(
			grpc.MaxCallRecvMsgSize(maxMsgSize),
			grpc.MaxCallSendMsgSize(maxMsgSize),
		),
	)
	if err != nil {
		return nil, err
	}
	client := pb.NewSemantixelInferenceClient(conn)
	return &Scanner{conn: conn, client: client, concurrency: concurrency, batchSize: batchSize}, nil
}

func (s *Scanner) Close() error {
	return s.conn.Close()
}

func (s *Scanner) WaitUntilReady() error {
	for i := 0; i < 30; i++ {
		resp, err := s.client.HealthCheck(context.Background(), &pb.HealthCheckRequest{})
		if err == nil && resp.Status == pb.ServingStatus_SERVING {
			log.Println("gRPC server is ready")
			return nil
		}
		log.Printf("Waiting for server to be ready (attempt %d)...", i+1)
		time.Sleep(2 * time.Second)
	}
	return nil
}

func (s *Scanner) ScanDirectories(directories []string) error {
	var files []string
	var mu sync.Mutex

	for _, dir := range directories {
		filepath.WalkDir(dir, func(path string, d fs.DirEntry, err error) error {
			if err != nil || d.IsDir() {
				return nil
			}
			ext := strings.ToLower(filepath.Ext(path))
			if imageExts[ext] || videoExts[ext] || audioExts[ext] {
				absPath, _ := filepath.Abs(path)
				mu.Lock()
				files = append(files, absPath)
				mu.Unlock()
			}
			return nil
		})
	}

	log.Printf("Found %d media files", len(files))

	if len(files) == 0 {
		return nil
	}

	// Upfront DB exist filtering
	toIndex := s.filterExistingFiles(files)
	log.Printf("%d files need indexing after DB filter", len(toIndex))

	if len(toIndex) == 0 {
		return nil
	}

	// Separate by type
	var videoPaths, audioPaths, imagePaths []string
	for _, f := range toIndex {
		ext := strings.ToLower(filepath.Ext(f))
		switch {
		case videoExts[ext]:
			videoPaths = append(videoPaths, f)
		case audioExts[ext]:
			audioPaths = append(audioPaths, f)
		default:
			imagePaths = append(imagePaths, f)
		}
	}

	// Start concurrent worker pool
	jobs := make(chan job, s.concurrency*2)
	var wg sync.WaitGroup

	for i := 0; i < s.concurrency; i++ {
		wg.Add(1)
		go s.worker(jobs, &wg)
	}

	// Enqueue video jobs
	for _, p := range videoPaths {
		jobs <- job{
			kind: "video",
			metadata: []*pb.IndexMediaMetadata{{
				Source: "local", Locator: p, DisplayPath: p,
			}},
		}
	}

	// Enqueue audio jobs
	for _, p := range audioPaths {
		jobs <- job{
			kind: "audio",
			metadata: []*pb.IndexMediaMetadata{{
				Source: "local", Locator: p, DisplayPath: p,
			}},
		}
	}

	// Enqueue image batch jobs
	for i := 0; i < len(imagePaths); i += s.batchSize {
		end := i + s.batchSize
		if end > len(imagePaths) {
			end = len(imagePaths)
		}
		batch := imagePaths[i:end]
		metas := make([]*pb.IndexMediaMetadata, len(batch))
		for j, p := range batch {
			metas[j] = &pb.IndexMediaMetadata{
				Source: "local", Locator: p, DisplayPath: p,
			}
		}
		jobs <- job{
			kind:  "image_batch",
			paths: batch,
			metadata: metas,
		}
	}

	close(jobs)
	wg.Wait()

	log.Println("Scanning completed")
	return nil
}

func (s *Scanner) worker(jobs <-chan job, wg *sync.WaitGroup) {
	defer wg.Done()
	for j := range jobs {
		switch j.kind {
		case "image_batch":
			s.indexImagesBatch(j.paths, j.metadata)
		case "video":
			s.indexVideo(j.metadata[0])
		case "audio":
			s.indexAudio(j.metadata[0])
		}
	}
}

func (s *Scanner) filterExistingFiles(files []string) []string {
	log.Printf("Checking %d files against DB...", len(files))
	exists := make([]bool, len(files))
	var mu sync.Mutex
	var wg sync.WaitGroup

	for start := 0; start < len(files); start += chunkSize {
		end := start + chunkSize
		if end > len(files) {
			end = len(files)
		}
		chunk := files[start:end]
		wg.Add(1)

		go func(idx int, paths []string) {
			defer wg.Done()
			metas := make([]*pb.IndexMediaMetadata, len(paths))
			for i, p := range paths {
				metas[i] = &pb.IndexMediaMetadata{
					Source: "local", Locator: p, DisplayPath: p,
				}
			}
			resp, err := s.client.CheckExists(context.Background(), &pb.CheckExistsRequest{Media: metas})
			if err != nil {
				log.Printf("CheckExists chunk failed: %v", err)
				return
			}
			mu.Lock()
			for i, ex := range resp.Exists {
				exists[idx+i] = ex
			}
			mu.Unlock()
		}(start, chunk)
	}
	wg.Wait()

	var toIndex []string
	for i, ex := range exists {
		if !ex {
			toIndex = append(toIndex, files[i])
		}
	}
	return toIndex
}

func (s *Scanner) indexVideo(meta *pb.IndexMediaMetadata) {
	_, err := s.client.IndexVideo(context.Background(), &pb.IndexVideoRequest{Metadata: meta})
	if err != nil {
		log.Printf("Error indexing video %s: %v", meta.Locator, err)
	} else {
		log.Printf("Indexed video: %s", meta.Locator)
	}
}

func (s *Scanner) indexAudio(meta *pb.IndexMediaMetadata) {
	_, err := s.client.IndexAudio(context.Background(), &pb.IndexAudioRequest{Metadata: meta})
	if err != nil {
		log.Printf("Error indexing audio %s: %v", meta.Locator, err)
	} else {
		log.Printf("Indexed audio: %s", meta.Locator)
	}
}

func (s *Scanner) indexImagesBatch(paths []string, metas []*pb.IndexMediaMetadata) {
	var images [][]byte
	var validMetas []*pb.IndexMediaMetadata

	for i, p := range paths {
		data, err := os.ReadFile(p)
		if err != nil {
			log.Printf("Failed to read %s: %v", p, err)
			continue
		}
		images = append(images, data)
		validMetas = append(validMetas, metas[i])
	}

	if len(images) == 0 {
		return
	}

	_, err := s.client.IndexImages(context.Background(), &pb.IndexImagesRequest{
		Images:   images,
		Metadata: validMetas,
	})
	if err != nil {
		log.Printf("Failed to index image batch of %d: %v", len(images), err)
	} else {
		log.Printf("Indexed batch of %d images", len(images))
	}
}
