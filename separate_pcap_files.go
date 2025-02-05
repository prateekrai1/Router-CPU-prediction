package main

import (
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"
)

// Date to filter in filenames
const targetDate = "2023-10-31"

func separatePcapFiles(inputDir, outputDir string) {
	// Ensure the output directory exists
	err := os.MkdirAll(outputDir, os.ModePerm)
	if err != nil {
		log.Fatalf("Error creating output directory: %v", err)
	}

	// Get all .pcap files in the input directory
	files, err := filepath.Glob(filepath.Join(inputDir, "*.pcap"))
	if err != nil {
		log.Fatalf("Error scanning directory: %v", err)
	}

	for _, file := range files {
		filename := filepath.Base(file)
		if strings.Contains(filename, targetDate) {
			destPath := filepath.Join(outputDir, filename)
			
			// Move the file
			err := os.Rename(file, destPath)
			if err != nil {
				log.Printf("Error moving file %s: %v", filename, err)
			} else {
				fmt.Printf("Moved: %s -> %s\n", file, destPath)
			}
		}
	}
}

func main() {
	inputDir := "/home/umd-user/Downloads/archive/pcap_combined/Combined_Pcaps/Benign_Pcaps/rewritten_pcaps" // Change this to your input directory
	outputDir := "./benign1" // New directory to store filtered files

	separatePcapFiles(inputDir, outputDir)
}

